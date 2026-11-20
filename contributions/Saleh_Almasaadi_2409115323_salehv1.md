# Saleh Almasaadi — Evaluation and Statistical Analysis

- Verified that all four baseline prediction files matched the 1,884 frozen test clips, reference transcripts, and speaker/session IDs.
- Recalculated WER and CER using the shared text normalization and reproduced the reported scores.
- Computed speaker/session-level bootstrap confidence intervals and paired randomization tests with Holm adjustment.
- Analyzed substitutions, deletions, insertions, and word-boundary errors.
- Added the reproducible evaluation notebook in `notebooks/` and exported the tables, methods, and chart to `results/person4_evaluation/`.

The comparison between the team's fine-tuned Whisper and the original Whisper Small is pending the official test predictions.
