"""Training-only memorization check and transcript-constrained re-estimation experiment.

This script deliberately does not read the test manifest, save over the published
model, or write over the initial dev/test predictions and run records.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from . import run as asr


def constrained_viterbi_alignment(features: np.ndarray, sequence: str, model: dict) -> np.ndarray:
    """Align frames to a known character sequence with optional one-frame boundaries.

    The trellis has one state per transcript character and a boundary state
    between adjacent characters. Character states can self-loop; transitions
    must follow the supplied transcript. Returned state IDs are character
    states 0, 2, ... and intervening boundary states 1, 3, ... .
    """
    tokens = model["tokens"]
    token_to_index = {token: index for index, token in enumerate(tokens)}
    sequence = [char for char in sequence if char in token_to_index]
    if not sequence:
        raise ValueError("Transcript contains no characters from the train vocabulary.")

    # If there are too few frames for an explicit boundary between every
    # character, retain a character-only forced path for this recording.
    has_boundaries = len(features) >= 2 * len(sequence) - 1
    state_count = 2 * len(sequence) - 1 if has_boundaries else len(sequence)
    state_to_token = sequence
    token_emissions = {}
    for token in set(sequence):
        token_emissions[token] = model["models"][token_to_index[token]].score_samples(features)
    boundary_emission = model["models"][-1].score_samples(features)

    dp = np.full((len(features), state_count), -np.inf, dtype=np.float64)
    back = np.full((len(features), state_count), -1, dtype=np.int32)
    dp[0, 0] = token_emissions[sequence[0]][0]
    stay = float(np.log(model["self_loop"]))
    leave = float(np.log(1.0 - model["self_loop"]))

    for frame in range(1, len(features)):
        previous = dp[frame - 1]
        for char_position, token in enumerate(state_to_token):
            char_state = 2 * char_position if has_boundaries else char_position
            stay_score = previous[char_state] + stay
            if char_position > 0:
                if has_boundaries:
                    predecessor = char_state - 1
                    advance_score = previous[predecessor]
                else:
                    predecessor = char_state - 1
                    advance_score = previous[predecessor] + leave
                if advance_score > stay_score:
                    dp[frame, char_state] = advance_score + token_emissions[token][frame]
                    back[frame, char_state] = predecessor
                else:
                    dp[frame, char_state] = stay_score + token_emissions[token][frame]
                    back[frame, char_state] = char_state
            else:
                dp[frame, char_state] = stay_score + token_emissions[token][frame]
                back[frame, char_state] = char_state

            if has_boundaries and char_position < len(sequence) - 1:
                boundary_state = char_state + 1
                dp[frame, boundary_state] = previous[char_state] + leave + boundary_emission[frame]
                back[frame, boundary_state] = char_state

    final_state = (2 * len(sequence) - 2) if has_boundaries else (len(sequence) - 1)
    if not np.isfinite(dp[-1, final_state]):
        raise RuntimeError("No valid transcript-constrained alignment was found.")
    path = np.empty(len(features), dtype=np.int32)
    state = final_state
    for frame in range(len(features) - 1, -1, -1):
        path[frame] = state
        if frame:
            state = int(back[frame, state])
            if state < 0:
                raise RuntimeError("Alignment backtrace is incomplete.")
    return path


def reestimate_from_alignments(rows: list[dict], model: dict, config: dict) -> tuple[dict, dict]:
    """Run one forced-alignment pass and refit GMMs from aligned train frames."""
    rng = np.random.default_rng(config["seed"] + 1)
    token_to_index = {token: index for index, token in enumerate(model["tokens"])}
    by_token: dict[str, list[np.ndarray]] = {}
    boundary_parts: list[np.ndarray] = []
    stays = exits = aligned_frames = aligned_utterances = 0
    skipped = []

    for row in rows:
        features = asr.mfcc_features(asr.AUDIO_ROOT / row["audio_path"], config)
        sequence = asr.read_text(row)
        path = constrained_viterbi_alignment(features, sequence, model)
        tokens = [char for char in sequence if char in token_to_index]
        has_boundaries = len(path) >= 2 * len(tokens) - 1
        if has_boundaries:
            for char_position, token in enumerate(tokens):
                char_frames = features[path == 2 * char_position]
                if len(char_frames):
                    asr.reservoir_append(by_token, token, char_frames, config["max_frames_per_character"], rng)
                if char_position < len(tokens) - 1:
                    boundary_frames = features[path == 2 * char_position + 1]
                    if len(boundary_frames):
                        boundary_parts.append(boundary_frames)
            for previous, current in zip(path[:-1], path[1:]):
                if previous == current and previous % 2 == 0:
                    stays += 1
                elif current == previous + 1 and current % 2 == 1:
                    exits += 1
        else:
            for char_position, token in enumerate(tokens):
                char_frames = features[path == char_position]
                if len(char_frames):
                    asr.reservoir_append(by_token, token, char_frames, config["max_frames_per_character"], rng)
        aligned_frames += len(features)
        aligned_utterances += 1

    old_models = model["models"]
    new_models = []
    fitted_tokens = []
    component_counts = dict(model["component_counts"])
    for index, token in enumerate(model["tokens"]):
        parts = by_token.get(token, [])
        if not parts:
            new_models.append(old_models[index])
            skipped.append(token)
            continue
        pool = np.concatenate(parts, axis=0)
        components = min(config["n_mixtures"], max(1, len(pool) // 20))
        mixture = GaussianMixture(
            n_components=components, covariance_type=config["covariance_type"],
            max_iter=config["gmm_max_iter"], reg_covar=config["gmm_reg_covar"],
            random_state=config["seed"] + 1, n_init=1,
        ).fit(pool)
        new_models.append(mixture)
        fitted_tokens.append(token)
        component_counts[token] = components

    if boundary_parts:
        pool = np.concatenate(boundary_parts, axis=0)
        if len(pool) > config["max_global_frames"]:
            pool = pool[rng.choice(len(pool), config["max_global_frames"], replace=False)]
        components = min(config["n_mixtures"], max(1, len(pool) // 20))
        boundary_model = GaussianMixture(
            n_components=components, covariance_type=config["covariance_type"],
            max_iter=config["gmm_max_iter"], reg_covar=config["gmm_reg_covar"],
            random_state=config["seed"] + 1, n_init=1,
        ).fit(pool)
        component_counts["<boundary>"] = components
    else:
        boundary_model = old_models[-1]
        skipped.append("<boundary>")

    updated = dict(model)
    updated["models"] = new_models + [boundary_model]
    updated["component_counts"] = component_counts
    updated["training_frames"] = aligned_frames
    updated["training_utterances"] = aligned_utterances
    updated["alignment_method"] = "transcript-constrained-viterbi"
    if stays + exits:
        updated["self_loop"] = float(np.clip(stays / (stays + exits), 0.50, 0.98))
    stats = {
        "alignment_method": "transcript-constrained-viterbi",
        "aligned_utterances": aligned_utterances,
        "aligned_frames": aligned_frames,
        "character_self_transitions": stays,
        "character_exits": exits,
        "updated_self_loop": updated["self_loop"],
        "refitted_character_models": fitted_tokens,
        "kept_initial_models_for_unseen_tokens": skipped,
    }
    return updated, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--dev-limit", type=int, default=8)
    parser.add_argument("--lm-scale", type=float, default=None)
    args = parser.parse_args()
    if args.train_limit < 2 or args.dev_limit < 1:
        parser.error("use at least two train utterances and one dev utterance")

    config = json.loads((asr.PACKAGE / "config.json").read_text(encoding="utf-8"))
    train_rows = asr.read_manifest("train")[:args.train_limit]
    dev_rows = asr.read_manifest("dev")[:args.dev_limit]
    missing = [row["audio_path"] for row in train_rows + dev_rows if not (asr.AUDIO_ROOT / row["audio_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Fetch the selected cleaned train/dev audio with Git LFS; first missing path: {missing[0]}")

    selected_config = json.loads((asr.RESULTS / "effective_config.json").read_text(encoding="utf-8"))
    lm_scale = args.lm_scale if args.lm_scale is not None else selected_config["selected_lm_scale"]
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        started = time.perf_counter()
        initial = asr.fit_model(train_rows, config)
        initial_fit_seconds = time.perf_counter() - started
        uniform_train_predictions, uniform_train_metrics = asr.run_decode(train_rows, initial, lm_scale)
        uniform_dev_predictions, uniform_dev_metrics = asr.run_decode(dev_rows, initial, lm_scale)

        started = time.perf_counter()
        realigned, alignment_stats = reestimate_from_alignments(train_rows, initial, config)
        realignment_seconds = time.perf_counter() - started
        realigned_train_predictions, realigned_train_metrics = asr.run_decode(train_rows, realigned, lm_scale)
        realigned_dev_predictions, realigned_dev_metrics = asr.run_decode(dev_rows, realigned, lm_scale)

    out = asr.RESULTS / "diagnostics"
    asr.write_predictions(out / "uniform_train_memorization.csv", uniform_train_predictions)
    asr.write_predictions(out / "realigned_train_memorization.csv", realigned_train_predictions)
    asr.write_predictions(out / "uniform_dev_sample.csv", uniform_dev_predictions)
    asr.write_predictions(out / "realigned_dev_sample.csv", realigned_dev_predictions)
    report = {
        "experiment": "training-only memorization plus one transcript-constrained realignment/re-estimation pass",
        "test_data_read": False,
        "train_limit": len(train_rows), "train_ids": [row["id"] for row in train_rows],
        "dev_limit": len(dev_rows), "dev_ids": [row["id"] for row in dev_rows],
        "selected_lm_scale_from_initial_dev_run": lm_scale,
        "uniform_initial_fit_seconds": initial_fit_seconds,
        "realignment_reestimation_seconds": realignment_seconds,
        "alignment": alignment_stats,
        "uniform_train_memorization": uniform_train_metrics,
        "realigned_train_memorization": realigned_train_metrics,
        "uniform_dev_sample": uniform_dev_metrics,
        "realigned_dev_sample": realigned_dev_metrics,
        "warnings": [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught_warnings],
        "interpretation": "Compare train memorization to diagnose learnability; compare the dev sample to assess whether realignment helps beyond the fitted recordings. Do not use this small experiment as a final benchmark.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": __import__("sklearn").__version__, "device": "CPU"},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "memorization_and_realignment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"uniform train: WER={uniform_train_metrics['wer']:.4f}, CER={uniform_train_metrics['cer']:.4f}")
    print(f"realigned train: WER={realigned_train_metrics['wer']:.4f}, CER={realigned_train_metrics['cer']:.4f}")
    print(f"uniform dev sample: WER={uniform_dev_metrics['wer']:.4f}, CER={uniform_dev_metrics['cer']:.4f}")
    print(f"realigned dev sample: WER={realigned_dev_metrics['wer']:.4f}, CER={realigned_dev_metrics['cer']:.4f}")
    print(f"diagnostics saved to {out}")


if __name__ == "__main__":
    main()
