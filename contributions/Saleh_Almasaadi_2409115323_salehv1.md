# Saleh Almasaadi

- Student ID 2409115323
- GitHub salehv1

My contribution covers evaluation and statistical analysis for the Icelandic ASR project. I verified that the four saved model prediction files matched all 1,884 frozen test clips, reference transcripts, and speaker/session IDs. I recalculated WER and CER using the shared text normalization and reproduced the reported scores.

I calculated 95% bootstrap confidence intervals and paired randomization tests with Holm adjustment, keeping recordings from the same speaker/session together. I also analyzed substitutions, deletions, insertions, and word-boundary errors. The results flag possible Samrómur training/test overlap for the two published Icelandic checkpoints.

The [evaluation notebook](../notebooks/Person4_Icelandic_ASR_Evaluation.ipynb) reproduces the analysis. The [results folder](../results/person4_evaluation/) contains the tables, methods, and comparison chart. The comparison between our team's fine-tuned Whisper and the original Whisper Small is pending the official test predictions.
