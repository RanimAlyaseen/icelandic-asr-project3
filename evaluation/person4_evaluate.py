#!/usr/bin/env python3
"""Person 4: reproducible ASR evaluation from saved predictions.

Run: python evaluation/person4_evaluate.py --person3-csv PATH --gmm-csv PATH --rnnt-csv PATH --ctc-csv PATH --attention-csv PATH --hybrid-csv PATH [--output-dir PATH]
Dependencies: numpy pandas jiwer==4.0.0 matplotlib; git required.
No GPU or training required. The four original baseline inputs are fetched from a pinned Git commit; added-model CSVs are supplied explicitly.
"""
import argparse
import hashlib
import os

parser = argparse.ArgumentParser(description="Evaluate Icelandic ASR predictions")
parser.add_argument("--person3-csv", type=str, default=None, help="Person 3 test_predictions.csv")
parser.add_argument("--gmm-csv", type=str, default=None, help="Person 5 MFCC+GMM-HMM test predictions CSV")
parser.add_argument("--rnnt-csv", type=str, default=None, help="Person 2 RNN-T test predictions CSV")
parser.add_argument("--ctc-csv", type=str, default=None, help="Person 2 scratch CTC encoder test predictions CSV")
parser.add_argument("--attention-csv", type=str, default=None, help="Person 2 scratch attention-only test predictions CSV")
parser.add_argument("--hybrid-csv", type=str, default=None, help="Person 2 hybrid CTC/attention test predictions CSV")
parser.add_argument("--output-dir", default="person4_evaluation", help="Output directory")
args = parser.parse_args()


# ---- Notebook section 3 ----

import csv
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

source_commit = "e9cd03331ef7f2295a5baf197c8c74f1ce41417d"
repo_url = "https://github.com/RanimAlyaseen/icelandic-asr-project3.git"
inputs = Path.cwd() / "person4_inputs"
results_dir = Path(args.output_dir).resolve()
extra_predictions = {}

pretrained = "results/pretrained/predictions"
model_sources = {
    "hubert_ctc_test.csv": "results/hubert_ctc/predictions/hubert_ctc_test.csv",
    "wav2vec2_is_test.csv": f"{pretrained}/wav2vec2_is_test.csv",
    "whisper_small_is_test_beam5.csv": f"{pretrained}/whisper_small_is_test_beam5.csv",
    "whisper_small_test_beam5.csv": f"{pretrained}/whisper_small_test_beam5.csv",
}
inputs.mkdir(exist_ok=True)
git_dir = inputs / "repository"
git_dir.mkdir(exist_ok=True)

def git(*args, **kwargs):
    return subprocess.run(["git", "-C", str(git_dir), *args], check=True, **kwargs)

git("init", "-q")
git("fetch", "-q", "--depth=1", "--no-tags", repo_url, source_commit)

source_files = {
    "evaluation_references.tsv": "output/evaluation_references.tsv",
    "test.jsonl": "output/test.jsonl",
    "prepare.py": "prepare.py",
    "audit.py": "audit.py",
    **model_sources,
}
for filename, repo_path in source_files.items():
    blob = git("show", f"{source_commit}:{repo_path}", capture_output=True).stdout
    (inputs / filename).write_bytes(blob)

model_paths = {name: inputs / name for name in model_sources}
if model_paths.keys() & extra_predictions.keys():
    raise ValueError("An added model duplicates a baseline name.")
model_paths.update({label: Path(path) for label, path in extra_predictions.items()})
model_files = list(model_paths)

# Use the same prepare.py and its local dependency from the pinned commit.
sys.path.insert(0, str(inputs))
try:
    normalize_text = runpy.run_path(str(inputs / "prepare.py"))["normalize_text"]
finally:
    sys.path.pop(0)

def read_rows(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))

print(f"Source commit: {source_commit}")
# Person 3 CSV has a different column schema; handled below after baseline validation.


# ---- Notebook section 5 ----

references = [
    row for row in read_rows(inputs / "evaluation_references.tsv", "\t")
    if row["split"] == "test"
]
official = {row["id"]: row for row in references}
assert len(references) == len(official) == 1884, "Check the frozen test references."

with (inputs / "test.jsonl").open(encoding="utf-8") as file:
    manifest = [json.loads(line) for line in file if line.strip()]
manifest_by_id = {row["id"]: row for row in manifest}
assert len(manifest) == len(manifest_by_id) == len(official)
assert manifest_by_id.keys() == official.keys(), "Test clip IDs differ."
assert all(row["split"] == "test" and row["speaker_session_id"] for row in manifest)
session_by_clip = {
    clip_id: str(row["speaker_session_id"]) for clip_id, row in manifest_by_id.items()
}
assert len(set(session_by_clip.values())) == 1026, "Unexpected session count."

def validate_predictions(rows, frozen, sessions):
    counts = Counter(row["id"] for row in rows)
    return {
        "duplicate IDs": sum(count > 1 for count in counts.values()),
        "missing IDs": len(frozen.keys() - counts.keys()),
        "unexpected IDs": len(counts.keys() - frozen.keys()),
        "wrong split": sum(row["split"] != "test" for row in rows),
        "reference mismatches": sum(
            row["id"] in frozen and (
                row["reference"] != frozen[row["id"]]["text"]
                or row["reference_raw"] != frozen[row["id"]]["text_raw"]
            )
            for row in rows
        ),
        "session mismatches": sum(
            row["id"] in sessions and row["speaker_session_id"] != sessions[row["id"]]
            for row in rows
        ),
        "malformed hypothesis fields": sum(row["hypothesis_raw"] is None for row in rows),
    }

# Verify exact matching, preservation of empty predictions, and duplicate detection.
example_ref = {"0001": {"text": "ég", "text_raw": "Ég."}}
example_sessions = {"0001": "s1"}
example = {
    "id": "0001", "split": "test", "speaker_session_id": "s1",
    "reference": "ég", "reference_raw": "Ég.", "hypothesis_raw": "",
}
assert not any(validate_predictions([example], example_ref, example_sessions).values())
assert validate_predictions([example, example], example_ref, example_sessions)["duplicate IDs"] == 1
bad_example = {**example, "reference": "þú", "speaker_session_id": "s2"}
issues = validate_predictions([bad_example], example_ref, example_sessions)
assert issues["reference mismatches"] == issues["session mismatches"] == 1

rows_by_model = {name: read_rows(path) for name, path in model_paths.items()}
if args.person3_csv:
        p3_path = Path(args.person3_csv)
        expected_sha256 = "506f0b277372ebdbba9e08c990e1b691773de6539a2a803b83544d57ac20bf3b"
        actual_sha256 = hashlib.sha256(p3_path.read_bytes()).hexdigest()

        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Recovered Whisper prediction CSV SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}. "
                "Do not evaluate a modified prediction file."
            )
        p3_rows = read_rows(p3_path)
        if len(p3_rows) != 1884 or any(not all(k in row for k in ("id", "split", "reference", "prediction")) for row in p3_rows):
            raise ValueError("Person 3 file must contain 1,884 rows and id/split/reference/prediction")
        # Canonicalize schema and compare with frozen official references.
        p3_converted = []
        for row in p3_rows:
            clip_id = row["id"]
            if clip_id not in official:
                raise ValueError(f"Unknown Person 3 clip ID: {clip_id}")
            p3_converted.append({
                "id": clip_id, "split": row["split"],
                "speaker_session_id": session_by_clip[clip_id],
                "reference": row["reference"],
                "reference_raw": official[clip_id]["text_raw"],
                "hypothesis_raw": row["prediction"],
                "hypothesis": normalize_text(row["prediction"]),
            })
        rows_by_model["whisper_finetuned_step1000_test.csv"] = p3_converted
        model_files.append("whisper_finetuned_step1000_test.csv")
if args.gmm_csv:
    gmm_path = Path(args.gmm_csv)
    gmm_rows = read_rows(gmm_path)
    if len(gmm_rows) != 1884:
        raise ValueError(f"GMM-HMM file should have 1,884 rows; found {len(gmm_rows)}")
    required = {"id", "split", "speaker_session_id", "reference", "reference_raw", "hypothesis_raw"}
    if any(not required.issubset(row) for row in gmm_rows):
        raise ValueError(f"GMM-HMM file is missing required columns: {required}")
    rows_by_model["mfcc_gmm_hmm_test.csv"] = gmm_rows
    model_files.append("mfcc_gmm_hmm_test.csv")
# Additional test prediction CSVs follow the team-standard baseline schema.
# Preserve their original, potentially empty, hypothesis_raw strings.
additional_model_csvs = (
    ("rnnt_test.csv", args.rnnt_csv),
    ("ctc_scratch_test.csv", args.ctc_csv),
    ("attention_scratch_test.csv", args.attention_csv),
    ("hybrid_ctc_attention_test.csv", args.hybrid_csv),
)
required_model_columns = {
    "id", "split", "speaker_session_id", "reference", "reference_raw", "hypothesis_raw"
}
for model_filename, csv_path in additional_model_csvs:
    if csv_path is None:
        continue
    new_rows = read_rows(Path(csv_path))
    if len(new_rows) != 1884:
        raise ValueError(f"{model_filename}: expected 1,884 rows, found {len(new_rows)}")
    if any(not required_model_columns.issubset(row) for row in new_rows):
        raise ValueError(f"{model_filename}: missing one or more required columns: {required_model_columns}")
    rows_by_model[model_filename] = new_rows
    model_files.append(model_filename)

for name, rows in rows_by_model.items():
    issues = validate_predictions(rows, official, session_by_clip)
    failures = {issue: count for issue, count in issues.items() if count}
    if failures:
        raise ValueError(f"{name}: {failures}")
    print(f"PASS {name} | {len(rows):,} rows")
print(f"Verified {len(set(session_by_clip.values())):,} speaker/session groups.")

# ---- Notebook section 7 ----

import jiwer

reported = {
    "hubert_ctc_test.csv": (68.46, 23.31),
    "wav2vec2_is_test.csv": (7.76, 1.86),
    "whisper_small_is_test_beam5.csv": (21.75, 5.60),
    "whisper_small_test_beam5.csv": (86.82, 36.35),
    "whisper_finetuned_step1000_test.csv": (36.06, 10.16),
    "rnnt_test.csv": (68.95, 30.54),
    "ctc_scratch_test.csv": (74.50, 22.88),
    "hybrid_ctc_attention_test.csv": (55.57, 17.77),
    # Person 2 reported attention-only WER rounded to one decimal (153.8%).
    # Recalculate its exact WER/CER rather than enforcing two-decimal equality.
}

for filename, rows in rows_by_model.items():
    expected = reported.get(filename)
    refs = [normalize_text(official[row["id"]]["text"]) for row in rows]
    preds = [normalize_text(row["hypothesis_raw"]) for row in rows]

    wer = round(100 * jiwer.wer(refs, preds), 2)
    cer = round(100 * jiwer.cer(refs, preds), 2)
    if expected is not None and (
        wer != expected[0] or (expected[1] is not None and cer != expected[1])
    ):
        raise ValueError(f"{filename}: recalculated {(wer, cer)} vs reported {expected}")
    status = "MATCH" if expected is not None else "SCORE"

    print(f"{status:5}  {filename:34} WER {wer:6.2f}%  CER {cer:6.2f}%")

# ---- Notebook section 9 ----

import numpy as np

sessions = sorted(set(session_by_clip.values()))
session_index = {session: i for i, session in enumerate(sessions)}
normalized_refs = {
    clip_id: normalize_text(row["text"]) for clip_id, row in official.items()
}

# First two columns are reference word and character counts.
# Each model then gets two columns for word and character errors.
session_totals = np.zeros((len(sessions), 2 + 2 * len(model_files)), dtype=np.int64)

for clip_id, reference in normalized_refs.items():
    i = session_index[session_by_clip[clip_id]]
    session_totals[i, :2] += [len(reference.split()), len(reference)]

for model_index, filename in enumerate(model_files):
    column = 2 + 2 * model_index
    for row in rows_by_model[filename]:
        clip_id = row["id"]
        i = session_index[session_by_clip[clip_id]]
        reference = normalized_refs[clip_id]
        prediction = normalize_text(row["hypothesis_raw"])

        words = jiwer.process_words(reference, prediction)
        chars = jiwer.process_characters(reference, prediction)
        session_totals[i, column:column + 2] += [
            words.substitutions + words.deletions + words.insertions,
            chars.substitutions + chars.deletions + chars.insertions,
        ]

seed = 2105
n_resamples = 10_000
rng = np.random.default_rng(seed)
bootstrap_totals = np.empty((n_resamples, session_totals.shape[1]), dtype=np.int64)

for start in range(0, n_resamples, 250):
    stop = min(start + 250, n_resamples)
    drawn = rng.integers(0, len(sessions), size=(stop - start, len(sessions)))
    bootstrap_totals[start:stop] = session_totals[drawn].sum(axis=1)

observed_totals = session_totals.sum(axis=0)
ci_rows = []

for model_index, filename in enumerate(model_files):
    for offset, metric in enumerate(("WER", "CER")):
        column = 2 + 2 * model_index + offset
        point = 100 * observed_totals[column] / observed_totals[offset]
        bootstrap_rates = 100 * bootstrap_totals[:, column] / bootstrap_totals[:, offset]
        low, high = np.percentile(bootstrap_rates, [2.5, 97.5])

        if filename in reported:
            if reported[filename][offset] is not None:
                assert round(point, 2) == reported[filename][offset]
        ci_rows.append({
            "model_file": filename,
            "metric": metric,
            "score_percent": point,
            "ci_low_percent": low,
            "ci_high_percent": high,
        })
        print(f"{filename:34} {metric} {point:6.2f}%  [{low:.2f}%, {high:.2f}%]")

# ---- Notebook section 11 ----

from itertools import combinations

comparisons = []
for a, b in combinations(range(len(model_files)), 2):
    for offset, metric in enumerate(("WER", "CER")):
        comparisons.append((a, b, offset, metric))

# Positive differences mean model A has the higher error rate.
group_differences = np.column_stack([
    session_totals[:, 2 + 2 * a + offset]
    - session_totals[:, 2 + 2 * b + offset]
    for a, b, offset, metric in comparisons
])
observed_differences = group_differences.sum(axis=0)

permutation_seed = 2106
n_permutations = 10_000
rng = np.random.default_rng(permutation_seed)
extreme_counts = np.zeros(len(comparisons), dtype=np.int64)

for start in range(0, n_permutations, 250):
    batch_size = min(250, n_permutations - start)
    signs = 2 * rng.integers(0, 2, size=(batch_size, len(sessions))) - 1
    permuted = signs @ group_differences
    extreme_counts += (
        np.abs(permuted) >= np.abs(observed_differences)
    ).sum(axis=0)

p_values = (extreme_counts + 1) / (n_permutations + 1)

# Holm adjustment across all WER and CER tests in this run.
order = np.argsort(p_values)
holm_p = np.empty_like(p_values)
holm_p[order] = np.minimum(
    1, np.maximum.accumulate(p_values[order] * np.arange(len(order), 0, -1))
)

paired_rows = []
for i, (a, b, offset, metric) in enumerate(comparisons):
    a_column = 2 + 2 * a + offset
    b_column = 2 + 2 * b + offset
    difference = 100 * observed_differences[i] / observed_totals[offset]
    bootstrap_differences = 100 * (
        bootstrap_totals[:, a_column] - bootstrap_totals[:, b_column]
    ) / bootstrap_totals[:, offset]
    low, high = np.percentile(bootstrap_differences, [2.5, 97.5])

    paired_rows.append({
        "model_a": model_files[a],
        "model_b": model_files[b],
        "metric": metric,
        "difference_pp": difference,
        "ci_low_pp": low,
        "ci_high_pp": high,
        "p_value": p_values[i],
        "p_holm": holm_p[i],
    })

    print(
        f"{Path(model_files[a]).stem} vs {Path(model_files[b]).stem} | "
        f"{metric}: {difference:+.2f} pp [{low:+.2f}, {high:+.2f}] | "
        f"p={p_values[i]:.5f}, Holm p={holm_p[i]:.5f}"
    )

# ---- Notebook section 13 ----

from itertools import groupby

error_types = {
    "substitute": "substitution",
    "delete": "deletion",
    "insert": "insertion",
}

error_summary = []
error_examples = []

for filename in model_files:
    totals = Counter()
    examples = {}

    for row in rows_by_model[filename]:
        clip_id = row["id"]
        reference = normalized_refs[clip_id]
        prediction = normalize_text(row["hypothesis_raw"])
        aligned = jiwer.process_words(reference, prediction)

        totals["substitutions"] += aligned.substitutions
        totals["deletions"] += aligned.deletions
        totals["insertions"] += aligned.insertions

        ref_words = aligned.references[0]
        pred_words = aligned.hypotheses[0]
        chunks = aligned.alignments[0]

        def record_example(category, ref_tokens, pred_tokens):
            examples.setdefault(category, {
                "model_file": filename,
                "clip_id": clip_id,
                "category": category,
                "reference_span": " ".join(ref_tokens),
                "prediction_span": " ".join(pred_tokens),
                "reference": reference,
                "prediction": prediction,
            })

        for chunk in chunks:
            if chunk.type in error_types:
                record_example(
                    error_types[chunk.type],
                    ref_words[chunk.ref_start_idx:chunk.ref_end_idx],
                    pred_words[chunk.hyp_start_idx:chunk.hyp_end_idx],
                )

        for is_equal, block in groupby(chunks, key=lambda chunk: chunk.type == "equal"):
            if is_equal:
                continue
            edits = list(block)
            ref_span = ref_words[edits[0].ref_start_idx:edits[-1].ref_end_idx]
            pred_span = pred_words[edits[0].hyp_start_idx:edits[-1].hyp_end_idx]

            if (ref_span and pred_span
                    and len(ref_span) != len(pred_span)
                    and "".join(ref_span) == "".join(pred_span)):
                totals["exact_word_boundary_cases"] += 1
                record_example("word boundary", ref_span, pred_span)

    error_summary.append({
        "model_file": filename,
        **{key: totals[key] for key in (
            "substitutions", "deletions", "insertions", "exact_word_boundary_cases"
        )},
    })
    error_examples.extend(examples.values())

    print(f"\n{filename}")
    print(
        f"  Substitutions: {totals['substitutions']:,} | "
        f"Deletions: {totals['deletions']:,} | "
        f"Insertions: {totals['insertions']:,} | "
        f"Exact word-boundary cases: {totals['exact_word_boundary_cases']:,}"
    )
    for category, example in examples.items():
        ref = example["reference_span"] or "∅"
        pred = example["prediction_span"] or "∅"
        print(f"  {example['clip_id']} | {category}: {ref!r} → {pred!r}")

# ---- Notebook section 15 ----

import pandas as pd

results_dir.mkdir(parents=True, exist_ok=True)

tables = {
    "model_scores_and_confidence_intervals.csv": ci_rows,
    "paired_comparisons.csv": paired_rows,
    "error_counts.csv": error_summary,
    "error_examples.csv": error_examples,
}

for filename, rows in tables.items():
    pd.DataFrame(rows).to_csv(results_dir / filename, index=False)
    print(f"Saved {filename} ({len(rows)} rows)")

# Generate the primary fine-tuning comparison from the same computed,
# speaker/session-clustered paired statistics (never hard-code results).
original_model = "whisper_small_test_beam5.csv"
finetuned_model = "whisper_finetuned_step1000_test.csv"
if finetuned_model in rows_by_model:
    score_lookup = {
        (row["model_file"], row["metric"]): row["score_percent"]
        for row in ci_rows
    }
    comparison_rows = []
    for metric in ("WER", "CER"):
        matches = [
            row for row in paired_rows
            if row["model_a"] == original_model
            and row["model_b"] == finetuned_model
            and row["metric"] == metric
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one original-vs-finetuned {metric} comparison")
        paired = matches[0]
        comparison_rows.append({
            "metric": metric,
            "original_percent": round(score_lookup[(original_model, metric)], 2),
            "finetuned_percent": round(score_lookup[(finetuned_model, metric)], 2),
            "improvement_pp": round(paired["difference_pp"], 2),
            "ci_low_pp": round(paired["ci_low_pp"], 2),
            "ci_high_pp": round(paired["ci_high_pp"], 2),
            "p_value": paired["p_value"],
        })
    comparison_path = results_dir / "whisper_finetuning_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"Saved {comparison_path} ({len(comparison_rows)} rows)")

methods = {
    "source_commit": source_commit,
    "split": "test",
    "test_clips": len(official),
    "models": model_files,
    "speaker_session_groups": len(sessions),
    "normalization": "prepare.normalize_text from the source commit",
    "confidence_intervals": {
        "method": "95% percentile bootstrap resampling whole speaker/session groups",
        "seed": seed,
        "resamples": n_resamples,
    },
    "paired_tests": {
        "method": "Two-sided paired randomization with model labels swapped by speaker/session group",
        "seed": permutation_seed,
        "random_swaps": n_permutations,
        "p_value_correction": f"Plus-one Monte Carlo correction, then Holm across {len(paired_rows)} comparisons",
        "difference_direction": "Model A minus model B, in percentage points",
    },
    "caveat": (
        "The published Icelandic wav2vec 2.0 and Whisper Small checkpoints "
        "may have seen some Samrómur test clips during external training."
    ),
}

(results_dir / "methods.json").write_text(
    json.dumps(methods, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Saved methods.json in {results_dir}")

# ---- Notebook section 16 ----

import matplotlib.pyplot as plt

labels = {
    "hubert_ctc_test.csv": "HuBERT + CTC",
    "wav2vec2_is_test.csv": "Icelandic wav2vec 2.0*",
    "whisper_small_is_test_beam5.csv": "Icelandic Whisper Small*",
    "whisper_small_test_beam5.csv": "Original Whisper Small",
    "whisper_finetuned_step1000_test.csv": "Fine-tuned Whisper Small",
    "mfcc_gmm_hmm_test.csv": "MFCC + GMM-HMM",
    "rnnt_test.csv": "RNN-T (scratch)",
    "ctc_scratch_test.csv": "CTC encoder (scratch)",
    "attention_scratch_test.csv": "Attention-only (scratch)",
    "hybrid_ctc_attention_test.csv": "Hybrid CTC/attention (scratch)",
}

scores = pd.DataFrame(ci_rows)
model_order = (
    scores[scores["metric"] == "WER"]
    .sort_values("score_percent")["model_file"]
    .tolist()
)

fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5.5, 0.53 * len(model_order))), sharey=True)
positions = list(range(len(model_order)))

for ax, metric in zip(axes, ("WER", "CER")):
    subset = scores[scores["metric"] == metric].set_index("model_file").loc[model_order]
    values = subset["score_percent"].to_numpy()
    errors = [
        values - subset["ci_low_percent"].to_numpy(),
        subset["ci_high_percent"].to_numpy() - values,
    ]

    ax.errorbar(values, positions, xerr=errors, fmt="o", capsize=4, color="#145A72")
    ax.set_title(metric)
    ax.set_xlabel("Error rate (%)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

axes[0].set_yticks(positions, labels=[labels.get(name, name) for name in model_order])
axes[0].invert_yaxis()
fig.suptitle("Icelandic ASR test results with 95% confidence intervals")
fig.text(0.02, 0.03, "* Published checkpoints may have seen some Samrómur test clips.", fontsize=9)
fig.tight_layout(rect=(0, 0.10, 1, 0.92))

chart_path = results_dir / "test_error_rates.png"
fig.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {chart_path}")
