# Saleh Almasaadi

- Student ID 2409115323
- GitHub salehv1

My contribution covers evaluation and statistical analysis for the Icelandic ASR project. I evaluated five ASR models on the same 1,884 frozen test clips from 1,026 speaker/session groups. I verified that the saved model predictions matched the official clip IDs and reference transcripts. I recalculated WER and CER using the shared text normalization and reproduced the reported scores.

I calculated 95% confidence intervals using 10,000 speaker/session-clustered bootstrap resamples and performed 20 paired statistical comparisons using 10,000 session-level randomization tests with Holm adjustment. I also analyzed substitutions, deletions, insertions, and word-boundary errors, and created a comparison chart. The results flag possible Samrómur training/test overlap for the two published Icelandic checkpoints.

I additionally evaluated our team's fine-tuned Whisper Small checkpoint 1000 on the development and test sets using a Tesla T4 GPU. On the frozen test set, it achieved 36.06% WER and 10.16% CER, compared with 86.82% WER and 36.35% CER for the original Whisper Small baseline. This represents a 50.76-percentage-point WER reduction (95% CI: 47.31–54.85). The original baseline used five-beam decoding, while the fine-tuned model used greedy decoding, so the comparison also reflects different decoding configurations.

I created and tested the standalone Python script [`evaluation/person4_evaluate.py`](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/evaluation/person4_evaluate.py), which reproduces the five-model statistical evaluation on a CPU without retraining. The [evaluation notebook](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/notebooks/Person4_Icelandic_ASR_Evaluation.ipynb) contains the original analysis. The [results folder](https://github.com/RanimAlyaseen/icelandic-asr-project3/tree/person4-evaluation/results/person4_evaluation) contains the predictions, statistical tables, error examples, methods, and comparison chart. I documented the results and reproduction instructions in the README and submitted the work through [Pull Request #3](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/3).
