"""Train and evaluate the MFCC + GMM-HMM baseline.

Usage from the repository root:
  python -m mfcc_gmm_hmm.run --train --limit 8 --dev-limit 2
  python -m mfcc_gmm_hmm.run --train
  python -m mfcc_gmm_hmm.run --evaluate-test --confirm-frozen
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.fft import dct
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare import normalize_text  # noqa: E402

PACKAGE = ROOT / "mfcc_gmm_hmm"
CONFIG_PATH = PACKAGE / "config.json"
AUDIO_ROOT = ROOT / "data" / "cleaned" / "samromur-21.05-20h-v1"
MANIFESTS = ROOT / "output"
MODEL_PATH = ROOT / "models" / "mfcc_gmm_hmm" / "model.joblib"
RESULTS = ROOT / "results" / "mfcc_gmm_hmm"


def read_manifest(split: str) -> list[dict]:
    path = MANIFESTS / f"{split}.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mfcc_features(path: Path, config: dict) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != config["sample_rate"]:
        raise ValueError(f"Expected {config['sample_rate']} Hz audio, got {sample_rate}: {path}")
    signal = audio.mean(axis=1)
    frame = round(sample_rate * config["frame_ms"] / 1000.0)
    hop = round(sample_rate * config["hop_ms"] / 1000.0)
    if signal.size < frame:
        signal = np.pad(signal, (0, frame - signal.size))
    count = 1 + max(0, (signal.size - frame) // hop)
    starts = np.arange(count) * hop
    frames = np.stack([signal[start:start + frame] for start in starts])
    frames *= np.hamming(frame).astype(np.float32)[None, :]
    spectrum = np.abs(np.fft.rfft(frames, n=config["n_fft"], axis=1)) ** 2

    n_mels = 40
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2]))[0], n_mels + 2)
    bins = np.floor((config["n_fft"] + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    filters = np.zeros((n_mels, config["n_fft"] // 2 + 1), dtype=np.float32)
    for band in range(n_mels):
        left, center, right = bins[band:band + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        filters[band, left:center] = np.arange(left, center) - left
        filters[band, center:right] = right - np.arange(center, right)
        filters[band] /= max(float(right - left), 1.0)
    log_mel = np.log(np.maximum(spectrum @ filters.T, 1e-10))
    static = dct(log_mel, type=2, axis=1, norm="ortho")[:, :config["n_mfcc"]]
    static -= static.mean(axis=0, keepdims=True)
    static /= np.maximum(static.std(axis=0, keepdims=True), 1e-4)

    def delta(values: np.ndarray) -> np.ndarray:
        radius = config["delta_window"]
        padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
        denom = 2.0 * sum(i * i for i in range(1, radius + 1))
        result = np.zeros_like(values)
        for offset in range(1, radius + 1):
            result += offset * (padded[radius + offset:radius + offset + len(values)] - padded[radius - offset:radius - offset + len(values)])
        return result / denom

    first = delta(static)
    second = delta(first)
    features = np.concatenate((static, first, second), axis=1)
    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-3)
    return features.astype(np.float32)


def read_text(row: dict) -> str:
    text = normalize_text(row["text"])
    return text.replace(" ", "|")


def reservoir_append(store: dict[str, list[np.ndarray]], token: str, values: np.ndarray,
                     cap: int, rng: np.random.Generator) -> None:
    if len(values) == 0:
        return
    bucket = store.setdefault(token, [])
    bucket.append(values.astype(np.float32, copy=False))
    total = sum(len(part) for part in bucket)
    if total > cap * 1.5:
        merged = np.concatenate(bucket, axis=0)
        keep = rng.choice(len(merged), size=cap, replace=False)
        store[token] = [merged[keep]]


def fit_model(rows: list[dict], config: dict) -> dict:
    vocab_file = MANIFESTS / "ctc_vocab.json"
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    tokens = [token for token, index in sorted(vocab.items(), key=lambda item: item[1]) if token not in {"[PAD]", "[UNK]"}]
    token_to_index = {token: index for index, token in enumerate(tokens)}
    rng = np.random.default_rng(config["seed"])
    by_token: dict[str, list[np.ndarray]] = {}
    blank: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    transition_counts = np.full((len(tokens), len(tokens)), config["transition_smoothing"], dtype=np.float64)
    initial_counts = np.full(len(tokens), config["transition_smoothing"], dtype=np.float64)
    total_chars = 0
    total_frames = 0

    for number, row in enumerate(rows, 1):
        features = mfcc_features(AUDIO_ROOT / row["audio_path"], config)
        text = read_text(row)
        sequence = [char for char in text if char in token_to_index]
        if not sequence:
            continue
        total_frames += len(features)
        total_chars += len(sequence)
        for token in sequence:
            initial_counts[token_to_index[token]] += 1.0
        for left, right in zip(sequence, sequence[1:]):
            transition_counts[token_to_index[left], token_to_index[right]] += 1.0

        edges = np.linspace(0, len(features), len(sequence) + 1, dtype=int)
        for position, token in enumerate(sequence):
            start, end = int(edges[position]), int(edges[position + 1])
            if end <= start:
                continue
            reservoir_append(by_token, token, features[start:end], config["max_frames_per_character"], rng)
        if len(sequence) > 1:
            boundary = np.clip(edges[1:-1], 0, len(features) - 1)
            if len(boundary):
                blank.extend([features[index:index + 1] for index in boundary])
        if sum(len(part) for part in global_parts) < config["max_global_frames"] * 1.5:
            global_parts.append(features)
        if number % 500 == 0:
            print(f"aligned {number}/{len(rows)} training utterances", flush=True)

    if not total_frames:
        raise RuntimeError("No training frames were loaded. Fetch the required cleaned train audio with Git LFS.")
    global_pool = np.concatenate(global_parts, axis=0)
    if len(global_pool) > config["max_global_frames"]:
        global_pool = global_pool[rng.choice(len(global_pool), config["max_global_frames"], replace=False)]
    blank_pool = np.concatenate(blank, axis=0) if blank else global_pool
    if len(blank_pool) > config["max_global_frames"]:
        blank_pool = blank_pool[rng.choice(len(blank_pool), config["max_global_frames"], replace=False)]

    models = []
    component_counts = {}
    for token in tokens + ["<boundary>"]:
        pool = np.concatenate(by_token[token], axis=0) if token in by_token else (blank_pool if token == "<boundary>" else global_pool)
        if len(pool) > config["max_frames_per_character"]:
            pool = pool[rng.choice(len(pool), config["max_frames_per_character"], replace=False)]
        components = min(config["n_mixtures"], max(1, len(pool) // 20))
        mixture = GaussianMixture(
            n_components=components,
            covariance_type=config["covariance_type"],
            max_iter=config["gmm_max_iter"],
            reg_covar=config["gmm_reg_covar"],
            random_state=config["seed"],
            n_init=1,
        )
        mixture.fit(pool)
        models.append(mixture)
        component_counts[token] = components

    transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial_probabilities = initial_counts / initial_counts.sum()
    mean_frames_per_char = total_frames / max(total_chars, 1)
    self_loop = float(np.clip(1.0 - 1.0 / max(mean_frames_per_char, 1.1), 0.50, 0.98))
    return {
        "tokens": tokens,
        "models": models,
        "transition_probabilities": transition_probabilities,
        "initial_probabilities": initial_probabilities,
        "self_loop": self_loop,
        "training_utterances": len(rows),
        "training_frames": total_frames,
        "component_counts": component_counts,
        "config": config,
    }


def decode(features: np.ndarray, model: dict, lm_scale: float) -> str:
    tokens = model["tokens"]
    models = model["models"]
    n_tokens = len(tokens)
    boundary_model = models[-1]
    emissions = np.column_stack([gmm.score_samples(features) for gmm in models[:-1]] + [boundary_model.score_samples(features)])
    char_emission = emissions[:, :n_tokens]
    boundary_emission = emissions[:, n_tokens]
    log_lm = np.log(np.maximum(model["transition_probabilities"], 1e-12)) * lm_scale
    start = np.log(np.maximum(model["initial_probabilities"], 1e-12)) * lm_scale
    self_loop = model["self_loop"]
    log_stay = math.log(self_loop)
    log_exit = math.log(1.0 - self_loop)
    frames = len(features)

    char_scores = start + char_emission[0]
    boundary_scores = np.full(n_tokens, -np.inf)
    back = np.full((frames, 2 * n_tokens), -1, dtype=np.int32)
    for frame in range(1, frames):
        next_char_sources = boundary_scores[:, None] + log_lm
        best_boundary = np.argmax(next_char_sources, axis=0)
        best_boundary_scores = next_char_sources[best_boundary, np.arange(n_tokens)]
        stay_scores = char_scores + log_stay
        choose_stay = stay_scores >= best_boundary_scores
        next_char_scores = np.maximum(stay_scores, best_boundary_scores) + char_emission[frame]
        back[frame, :n_tokens] = np.where(choose_stay, np.arange(n_tokens), n_tokens + best_boundary)

        next_boundary_scores = char_scores + log_exit + boundary_emission[frame]
        back[frame, n_tokens:] = np.arange(n_tokens)
        char_scores = next_char_scores
        boundary_scores = next_boundary_scores

    state = int(np.argmax(char_scores))
    states = [state]
    for frame in range(frames - 1, 0, -1):
        state = int(back[frame, state])
        states.append(state)
    states.reverse()
    output = []
    previous = None
    for state in states:
        if state < n_tokens and state != previous:
            output.append(tokens[state])
        previous = state
    return "".join(output).replace("|", " ").strip()


def edit_distance(left: list, right: list) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score_rows(rows: list[dict], predictions: list[str]) -> dict:
    word_errors = word_count = char_errors = char_count = 0
    for row, prediction in zip(rows, predictions):
        reference = normalize_text(row["text"])
        hypothesis = normalize_text(prediction)
        ref_words, hyp_words = reference.split(), hypothesis.split()
        word_errors += edit_distance(ref_words, hyp_words)
        word_count += len(ref_words)
        char_errors += edit_distance(list(reference), list(hypothesis))
        char_count += len(reference)
    return {
        "wer": word_errors / max(word_count, 1),
        "cer": char_errors / max(char_count, 1),
        "word_errors": word_errors,
        "reference_words": word_count,
        "char_errors": char_errors,
        "reference_characters": char_count,
    }


def run_decode(rows: list[dict], model: dict, scale: float) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    predictions = []
    skipped = []
    for row in rows:
        path = AUDIO_ROOT / row["audio_path"]
        if not path.exists():
            skipped.append(row["id"])
            continue
        raw = decode(mfcc_features(path, model["config"]), model, scale)
        predictions.append({
            "id": row["id"], "split": row["split"], "speaker_session_id": row["speaker_session_id"],
            "duration_sec": row["duration_sec"], "reference": normalize_text(row["text"]),
            "reference_raw": row["text_raw"], "hypothesis_raw": raw, "hypothesis": normalize_text(raw),
        })
    wall = time.perf_counter() - started
    scored = score_rows([{"text": row["reference"]} for row in predictions], [row["hypothesis"] for row in predictions])
    audio_seconds = sum(float(row["duration_sec"]) for row in predictions)
    return predictions, {
        "utterances": len(predictions), "skipped_missing_audio": skipped,
        "audio_hours": audio_seconds / 3600, "wall_seconds_incl_audio_loading": wall,
        "rtf_incl_audio_loading": wall / max(audio_seconds, 1e-9), **scored,
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "split", "speaker_session_id", "duration_sec", "reference", "reference_raw", "hypothesis_raw", "hypothesis"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit train utterances for a smoke run")
    parser.add_argument("--dev-limit", type=int, help="Limit dev utterances for a smoke run")
    parser.add_argument("--confirm-frozen", action="store_true", help="Confirm model and decoding settings were selected on dev")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.train:
        all_train = read_manifest("train")
        train_rows = all_train[:args.limit] if args.limit else all_train
        started = time.perf_counter()
        model = fit_model(train_rows, config)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH, compress=3)
        train_seconds = time.perf_counter() - started
        dev_rows = read_manifest("dev")
        if args.dev_limit:
            dev_rows = dev_rows[:args.dev_limit]
        candidates = []
        decoded = {}
        for scale in config["lm_scale_candidates"]:
            predictions, metrics = run_decode(dev_rows, model, scale)
            candidates.append({"lm_scale": scale, **metrics})
            decoded[scale] = predictions
            print(f"dev lm_scale={scale}: WER={metrics['wer']:.4f} CER={metrics['cer']:.4f} ({metrics['utterances']} clips)", flush=True)
        available = [entry for entry in candidates if entry["utterances"]]
        if not available:
            raise RuntimeError("No dev audio was available. Fetch the needed cleaned dev audio through Git LFS.")
        best = min(available, key=lambda item: (item["wer"], item["cer"], item["lm_scale"]))
        best_scale = best["lm_scale"]
        write_predictions(RESULTS / "predictions" / "mfcc_gmm_hmm_dev.csv", decoded[best_scale])
        record = {
            "model_key": "mfcc_gmm_hmm", "family": "MFCC + character GMM-HMM",
            "split": "dev", "train_utterances": len(train_rows), "training_seconds": train_seconds,
            "training_frames": model["training_frames"], "model_sha256": sha256(MODEL_PATH),
            "config": config, "dev_candidates": candidates, "selected_lm_scale": best_scale,
            "selected_dev_metrics": best, "text_normalization": "prepare.normalize_text",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scikit_learn": __import__("sklearn").__version__, "soundfile": sf.__version__, "device": "CPU"},
            "train_is_smoke_limit": bool(args.limit),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "model_info").mkdir(parents=True, exist_ok=True)
        (RESULTS / "model_info" / "mfcc_gmm_hmm_dev.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RESULTS / "effective_config.json").write_text(json.dumps({**config, "selected_lm_scale": best_scale}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved model {MODEL_PATH}; best dev WER={best['wer']:.4f} CER={best['cer']:.4f}")
        return

    if not args.confirm_frozen:
        parser.error("test evaluation requires --confirm-frozen after selecting settings on dev")
    if not MODEL_PATH.exists():
        parser.error("trained model is missing; train first")
    dev_record_path = RESULTS / "model_info" / "mfcc_gmm_hmm_dev.json"
    if not dev_record_path.exists():
        parser.error("dev selection record is missing; train and select settings on dev first")
    dev_record = json.loads(dev_record_path.read_text(encoding="utf-8"))
    if dev_record.get("train_is_smoke_limit"):
        parser.error("the current model was fitted on a smoke subset; run full training before test evaluation")
    model = joblib.load(MODEL_PATH)
    test_rows = read_manifest("test")
    predictions, metrics = run_decode(test_rows, model, dev_record["selected_lm_scale"])
    write_predictions(RESULTS / "predictions" / "mfcc_gmm_hmm_test.csv", predictions)
    record = {
        "model_key": "mfcc_gmm_hmm", "split": "test", "test_evaluation": "single evaluation after dev freeze",
        "selected_lm_scale": dev_record["selected_lm_scale"], "config": model["config"],
        "model_sha256": sha256(MODEL_PATH), "text_normalization": "prepare.normalize_text", **metrics,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": __import__("sklearn").__version__, "soundfile": sf.__version__, "device": "CPU"},
    }
    (RESULTS / "model_info" / "mfcc_gmm_hmm_test.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"test WER={metrics['wer']:.4f} CER={metrics['cer']:.4f}")


if __name__ == "__main__":
    main()
