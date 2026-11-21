# Ranim Alyaseen

- Student ID **2309115360**
- GitHub **RanimAlyaseen**

My project contribution covers dataset preprocessing for the shared Samrómur 21.05 Icelandic ASR comparison. I prepared the corpus audit, seed-2105 split selection, fixed evaluation references, train-only character vocabulary, and audio and text preprocessing for Whisper and wav2vec 2.0. The split manifests preserve the official partitions and Icelandic text.

My preprocessing work also includes reproducibility fixes for UTF-8 LF artifacts and the compressed audit, independent correctness tests, generated quality and exclusion reports, and handoff and storage documentation. The [release verification](../output/verification.json) and [handoff verification](../output/handoff_verification.json) record the automated checks. The [focused review queue](../output/focused_review.tsv) collects questionable recordings for follow-up.

I defined the bounded raw input, verified each frozen recording against the official partition and session metadata, and implemented the checksum-verified raw-to-cleaned pipeline. The [split verification report](../output/split_integrity.json) and [subset removal log](../output/raw_subset_removals.tsv) record the results. I also configured Git LFS, checked baseline applicability against official model cards, and documented licensing and data access.

I listened to all 60 recordings in the [focused audio-quality review](../review/focused_manual_review.tsv). I marked 16 usable, 43 with audibility issues, and one uncertain. All 60 recordings were available. I recorded audio-quality recommendations for each clip and kept the frozen dataset unchanged. Icelandic transcript agreement was not assessed.

Model training and benchmark results are outside this contribution. Each teammate documents and commits their own work.

I published the raw and cleaned audio and verified an [anonymous Git LFS round trip](../output/lfs_verification.json) from a fresh checkout.
