"""Fair dev-only LM sweep for baseline and selected aligned GMM-HMM.

Run after the full iterative_alignment experiment. Never reads test data.
"""
from __future__ import annotations

import argparse
import csv
import json

import joblib
import numpy as np

from . import run
from .iterative_alignment import write_predictions_lf

SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0)
OUTPUT = run.ROOT / "results" / "mfcc_gmm_hmm_aligned" / "dev_comparison.json"


def paired_dev_uncertainty(baseline_csv, aligned_csv, seed: int = 2105, draws: int = 5000) -> dict:
    """Paired utterance bootstrap of WER difference; descriptive on tuned dev."""
    with baseline_csv.open(encoding="utf-8", newline="") as stream:
        baseline = list(csv.DictReader(stream))
    with aligned_csv.open(encoding="utf-8", newline="") as stream:
        aligned = list(csv.DictReader(stream))
    if not baseline or len(baseline) != len(aligned):
        raise ValueError("Missing or unequal dev predictions")
    for left, right in zip(baseline, aligned):
        if (left["id"], left["reference"]) != (right["id"], right["reference"]):
            raise ValueError("Dev predictions are not paired by ID and reference")
    ref_words = np.array([len(row["reference"].split()) for row in baseline])
    base_errors = np.array([run.edit_distance(row["reference"].split(), row["hypothesis"].split()) for row in baseline])
    aligned_errors = np.array([run.edit_distance(row["reference"].split(), row["hypothesis"].split()) for row in aligned])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(baseline), size=(draws, len(baseline)))
    difference = (base_errors[indexes].sum(axis=1) - aligned_errors[indexes].sum(axis=1)) / ref_words[indexes].sum(axis=1)
    return {"method": "paired utterance bootstrap on selected dev predictions",
            "seed": seed, "draws": draws, "difference_definition": "baseline WER minus aligned WER",
            "observed_difference": float((base_errors.sum() - aligned_errors.sum()) / ref_words.sum()),
            "percentile_95_interval": np.percentile(difference, [2.5, 97.5]).tolist(),
            "limitation": "Both settings were selected on dev, so this interval is exploratory and is not an unbiased test comparison."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-only", action="store_true", help="Recompute uncertainty from already saved paired dev predictions")
    args = parser.parse_args()
    aligned_path = run.ROOT / "models" / "mfcc_gmm_hmm_aligned" / "model.joblib"
    selected_path = OUTPUT.parent / "dev_run.json"
    prior = json.loads(selected_path.read_text(encoding="utf-8"))
    if prior["smoke"] or prior["train_utterances"] != len(run.read_manifest("train")):
        raise RuntimeError("A full train-only alignment run is required")
    if run.sha256(aligned_path) != prior["model_sha256"]:
        raise RuntimeError("Aligned model checksum differs from the dev run")
    baseline_record = json.loads((run.RESULTS / "model_info" / "mfcc_gmm_hmm_dev.json").read_text(encoding="utf-8"))
    if run.sha256(run.MODEL_PATH) != baseline_record["model_sha256"]:
        raise RuntimeError("Baseline model checksum differs from its dev run")
    if args.stats_only:
        record = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if record["models"]["iterative_alignment"]["model_sha256"] != run.sha256(aligned_path):
            raise RuntimeError("Saved comparison belongs to another aligned model")
    else:
        rows = run.read_manifest("dev")
        comparison = {}
        for name, path in (("uniform_baseline", run.MODEL_PATH), ("iterative_alignment", aligned_path)):
            model = joblib.load(path)
            candidates = []
            winner = None
            for scale in SCALES:
                predictions, metrics = run.run_decode(rows, model, scale)
                if metrics["utterances"] != len(rows):
                    raise RuntimeError(f"Incomplete dev audio for {name}")
                entry = {"lm_scale": scale, **metrics}
                candidates.append(entry)
                print(f"{name} LM={scale}: dev WER={metrics['wer']:.4f} CER={metrics['cer']:.4f}", flush=True)
                if winner is None or (entry["wer"], entry["cer"], scale) < (winner["wer"], winner["cer"], winner["lm_scale"]):
                    winner = entry
                    write_predictions_lf(OUTPUT.parent / f"{name}_dev_predictions.csv", predictions)
            comparison[name] = {"model_sha256": run.sha256(path), "candidates": candidates, "selected": winner,
                                "predictions": f"{name}_dev_predictions.csv"}
        record = {"selection_split": "dev", "test_evaluated": False, "dev_utterances": len(rows),
                  "selection": "minimum WER, then CER, then decoder weight on same candidate grid",
                  "candidate_lm_scales": SCALES, "alignment_iteration": prior["selected"]["iteration"],
                  "seed": prior["seed"], "normalization": "prepare.normalize_text", "models": comparison,
                  "interpretation": "Exploratory dev model comparison after tuning on dev; does not establish test improvement or statistical significance."}
    record["paired_dev_uncertainty"] = paired_dev_uncertainty(
        OUTPUT.parent / "uniform_baseline_dev_predictions.csv",
        OUTPUT.parent / "iterative_alignment_dev_predictions.csv",
    )
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base, aligned = (record["models"][key]["selected"] for key in ("uniform_baseline", "iterative_alignment"))
    print(f"Dev WER baseline {base['wer']:.4f}; aligned {aligned['wer']:.4f}; do not use test to choose settings.")


if __name__ == "__main__":
    main()
