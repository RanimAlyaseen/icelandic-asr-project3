"""Train a separate, transcript-constrained GMM-HMM alignment experiment.

Run from the repository root after fetching the baseline model and cleaned audio:
  python -m mfcc_gmm_hmm.iterative_alignment --limit 8 --dev-limit 3
  python -m mfcc_gmm_hmm.iterative_alignment

Only train audio and transcripts refit the acoustic models. Dev selects the
alignment iteration and decoder LM weight. This command never reads test data.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from . import run


def write_predictions_lf(path: Path, rows: list[dict]) -> None:
    """Use UTF-8 and LF for reproducible committed prediction tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "split", "speaker_session_id", "duration_sec", "reference",
               "reference_raw", "hypothesis_raw", "hypothesis"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def forced_align(features: np.ndarray, sequence: str, model: dict) -> np.ndarray:
    """Viterbi path through transcript positions with stay/advance arcs.

    Each transcript position emits at least one frame. Equal adjacent letters
    remain separate positions; boundaries are inferred rather than evenly cut.
    """
    tokens = model["tokens"]
    lookup = {token: index for index, token in enumerate(tokens)}
    if not sequence or any(char not in lookup for char in sequence):
        raise ValueError("Transcript contains an unknown token or is empty")
    frames, states = len(features), len(sequence)
    if frames < states:
        raise ValueError("Fewer acoustic frames than transcript positions")
    columns = [lookup[char] for char in sequence]
    emissions = np.column_stack([model["models"][index].score_samples(features)
                                  for index in sorted(set(columns))])
    mapping = {index: position for position, index in enumerate(sorted(set(columns)))}
    emissions = emissions[:, [mapping[index] for index in columns]]
    stay = np.log(model["self_loop"])
    advance = np.log1p(-model["self_loop"])
    scores = np.full(states, -np.inf)
    scores[0] = emissions[0, 0]
    advanced = np.zeros((frames, states), dtype=np.bool_)
    for frame in range(1, frames):
        from_stay = scores + stay
        from_advance = np.r_[-np.inf, scores[:-1] + advance]
        advanced[frame] = from_advance > from_stay
        scores = np.maximum(from_stay, from_advance) + emissions[frame]
        # Paths that cannot still reach the final state are never useful.
        scores[:max(0, states - (frames - frame))] = -np.inf
    if not np.isfinite(scores[-1]):
        raise ValueError("No valid transcript-constrained alignment")
    path = np.empty(frames, dtype=np.int32)
    state = states - 1
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        if frame and advanced[frame, state]:
            state -= 1
    if state != 0 or np.any(np.diff(path) < 0) or np.any(np.diff(path) > 1):
        raise AssertionError("Invalid forced alignment path")
    return path


def refit(rows: list[dict], previous: dict, iteration: int) -> tuple[dict, dict]:
    """Refit character mixtures on train-only Viterbi segments."""
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Acoustic refitting accepts train rows only")
    config = previous["config"]
    rng = np.random.default_rng(config["seed"] + iteration)
    buckets: dict[str, list[np.ndarray]] = {}
    boundary: list[np.ndarray] = []
    boundary_seen = 0
    skipped: list[str] = []
    for number, row in enumerate(rows, 1):
        features = run.mfcc_features(run.AUDIO_ROOT / row["audio_path"], config)
        sequence = "".join(char for char in run.read_text(row) if char in previous["tokens"])
        try:
            path = forced_align(features, sequence, previous)
        except ValueError:
            skipped.append(row["id"])
            continue
        cuts = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
        for position, token in enumerate(sequence):
            run.reservoir_append(buckets, token, features[cuts[position]:cuts[position + 1]],
                                 config["max_frames_per_character"], rng)
        for at in cuts[1:-1]:
            boundary_seen += 1
            sample = features[at:at + 1]
            if len(boundary) < config["max_frames_per_character"]:
                boundary.append(sample)
            else:
                replacement = rng.integers(boundary_seen)
                if replacement < len(boundary):
                    boundary[replacement] = sample
        if number % 500 == 0:
            print(f"iteration {iteration}: aligned {number}/{len(rows)} train clips", flush=True)
    if not buckets:
        raise RuntimeError("No train utterances could be aligned; check Git LFS audio and transcripts")
    if boundary:
        buckets["<boundary>"] = boundary
    fitted = []
    components = {}
    for index, token in enumerate(previous["tokens"] + ["<boundary>"]):
        if token not in buckets or sum(map(len, buckets[token])) < 2:
            fitted.append(previous["models"][index])
            components[token] = previous["component_counts"][token]
            continue
        pool = np.concatenate(buckets[token], axis=0)
        if len(pool) > config["max_frames_per_character"]:
            pool = pool[rng.choice(len(pool), config["max_frames_per_character"], replace=False)]
        count = min(config["n_mixtures"], max(1, len(pool) // 20))
        mixture = GaussianMixture(
            n_components=count, covariance_type=config["covariance_type"],
            max_iter=config["gmm_max_iter"], reg_covar=config["gmm_reg_covar"],
            random_state=config["seed"] + iteration, n_init=1,
        )
        mixture.fit(pool)
        fitted.append(mixture)
        components[token] = count
    result = {**previous, "models": fitted, "component_counts": components,
              "alignment_iteration": iteration}
    return result, {"aligned": len(rows) - len(skipped), "skipped_ids": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="First N train clips for a smoke run")
    parser.add_argument("--dev-limit", type=int, help="First N dev clips for a smoke run")
    parser.add_argument("--iterations", type=int, default=2, help="Number of train-only realignment passes")
    args = parser.parse_args()
    if args.iterations < 1 or (args.limit is not None and args.limit < 1) or (args.dev_limit is not None and args.dev_limit < 1):
        parser.error("iterations and sample limits must be positive")
    if not run.MODEL_PATH.exists() or run.MODEL_PATH.read_bytes()[:7] == b"version":
        parser.error("Fetch models/mfcc_gmm_hmm/model.joblib with Git LFS first")
    baseline_record = json.loads((run.RESULTS / "model_info" / "mfcc_gmm_hmm_dev.json").read_text(encoding="utf-8"))
    baseline_hash = run.sha256(run.MODEL_PATH)
    if baseline_hash != baseline_record["model_sha256"]:
        parser.error("Baseline model checksum differs from the frozen dev run")
    train = run.read_manifest("train")
    dev = run.read_manifest("dev")
    train = train[:args.limit] if args.limit else train
    dev = dev[:args.dev_limit] if args.dev_limit else dev
    baseline = joblib.load(run.MODEL_PATH)
    if len(baseline["models"]) != len(baseline["tokens"]) + 1:
        parser.error("Unexpected baseline model format")
    smoke = args.limit is not None or args.dev_limit is not None
    name = "mfcc_gmm_hmm_aligned_smoke" if smoke else "mfcc_gmm_hmm_aligned"
    target = run.ROOT / "results" / name
    model_path = run.ROOT / "models" / name / "model.joblib"
    started = time.perf_counter()
    candidate_metrics = []
    best = None
    current = baseline
    for iteration in range(1, args.iterations + 1):
        current, alignment = refit(train, current, iteration)
        for scale in current["config"]["lm_scale_candidates"]:
            predictions, metrics = run.run_decode(dev, current, scale)
            candidate = {"iteration": iteration, "lm_scale": scale, "alignment": alignment, **metrics}
            candidate_metrics.append(candidate)
            print(f"iteration={iteration}, lm_scale={scale}: dev WER={metrics['wer']:.4f} CER={metrics['cer']:.4f}", flush=True)
            if metrics["utterances"] == len(dev) and (best is None or
                    (metrics["wer"], metrics["cer"], iteration, scale) <
                    (best["wer"], best["cer"], best["iteration"], best["lm_scale"])):
                best = candidate
                target.mkdir(parents=True, exist_ok=True)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(current, model_path, compress=3)
                write_predictions_lf(target / "dev_predictions.csv", predictions)
    if best is None:
        raise RuntimeError("Full requested dev subset unavailable; fetch cleaned dev audio via Git LFS")
    record = {
        "experiment": name, "selection_split": "dev", "test_evaluated": False,
        "baseline_model_sha256": baseline_hash,
        "model_sha256": run.sha256(model_path), "seed": current["config"]["seed"],
        "train_utterances": len(train), "dev_utterances": len(dev),
        "smoke": smoke, "wall_seconds_including_dev": time.perf_counter() - started,
        "frozen_baseline_full_dev": None if smoke else baseline_record["selected_dev_metrics"],
        "candidates": candidate_metrics, "selected": best,
        "normalization": "prepare.normalize_text", "config": current["config"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": __import__("sklearn").__version__, "device": "CPU"},
    }
    (target / "dev_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected iteration {best['iteration']}, lm_scale {best['lm_scale']}, dev WER {best['wer']:.4f}; model {model_path}")


if __name__ == "__main__":
    main()
