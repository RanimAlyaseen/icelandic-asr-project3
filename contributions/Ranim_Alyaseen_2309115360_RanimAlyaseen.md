# Ranim Alyaseen

- Student ID **2309115360**
- GitHub **RanimAlyaseen**
- Role **Project coordinator and integration reviewer**

## Dataset preparation

Built the shared data foundation for the Samrómur 21.05 Icelandic ASR comparison. This work covered the corpus audit, seed-2105 split selection, frozen evaluation references, a train-only character vocabulary, and audio and text preprocessing for Whisper and wav2vec 2.0. The manifests preserve the official train, dev, and test partitions and Icelandic characters.

Defined the bounded raw input, verified each selected recording against the release's partition and session metadata, and implemented the checksum-verified raw-to-cleaned pipeline. The [split integrity report](../output/split_integrity.json) and [raw subset removal log](../output/raw_subset_removals.tsv) document the outcome.

## Reproducibility and delivery

Made the preprocessing outputs reproducible through UTF-8 LF artifacts, a compressed audit, independent correctness tests, and generated quality and exclusion reports. Prepared the handoff and storage documentation, configured Git LFS, and documented licensing and data access. The [release verification](../output/verification.json) and [handoff verification](../output/handoff_verification.json) record the automated checks.

Published the raw and cleaned audio and verified an [anonymous Git LFS round trip](../output/lfs_verification.json) from a fresh checkout.

## Quality review and integration

Completed audio-quality listening for all 60 recordings in the [focused manual review](../review/focused_manual_review.tsv), drawn from the [focused review queue](../output/focused_review.tsv). The review marked 16 usable, 43 with audibility issues, and one uncertain. All recordings were available. The log records a recommendation for each clip while the frozen dataset remains unchanged. Icelandic transcript agreement was not assessed.

Coordinated task ownership and progress across the team, reviewed pull requests for completeness and reproducibility, and checked reported scores against committed predictions and frozen references. Maintained the shared README, coordinated corrections and merge follow-ups, and preserved each teammate's attribution and commit history. Baseline applicability was checked against official model cards.

Model training and benchmark result generation are documented in teammates' individual contributions.
