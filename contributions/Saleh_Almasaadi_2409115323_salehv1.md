# Saleh Almasaadi

- Student ID 2409115323
- GitHub salehv1

My contribution covers evaluation and statistical analysis for the Icelandic ASR project. I evaluated seven ASR models on the same 1,884 frozen test clips from 1,026 speaker/session groups, including the pretrained baselines, HuBERT + CTC, our fine-tuned Whisper Small, RNN-T, and MFCC + GMM-HMM. I verified the clip IDs, reference transcripts, and speaker/session groups and recalculated WER and CER using the shared `prepare.normalize_text` function.

I calculated 95% confidence intervals using 10,000 speaker/session-clustered bootstrap resamples and performed 42 paired WER/CER statistical comparisons using 10,000 session-level randomization tests with Holm correction. I also analyzed substitutions, deletions, insertions, and word-boundary errors, generated comparison graphs, and examined the 39 original Whisper test predictions containing digits without changing the frozen scoring policy.

I evaluated Person 3's recovered Whisper Small checkpoint 1000 on the development and test sets using a Tesla T4 GPU. The model achieved 34.51% WER and 9.69% CER on the development set and 36.06% WER and 10.16% CER on the test set. Compared with the original Whisper Small baseline, this represents an observed WER reduction of 50.76 percentage points (95% CI: 47.31–54.85). The two models used different decoding configurations, which is documented as a limitation.

I additionally evaluated Person 5's MFCC + GMM-HMM model, obtaining 101.47% WER and 87.74% CER, and Person 2's RNN-T model, obtaining 68.95% WER and 30.54% CER. The RNN-T and HuBERT + CTC models had no statistically significant difference in WER after Holm correction (adjusted p = 0.50725), although HuBERT + CTC achieved significantly lower CER.

I created and successfully tested the standalone Python script [`evaluation/person4_evaluate.py`](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/evaluation/person4_evaluate.py), which reproduces the evaluation using saved predictions on a CPU. I also updated the [evaluation notebook](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/notebooks/Person4_Icelandic_ASR_Evaluation.ipynb) to include all seven models and documented the exact dependency versions.

The [results folder](https://github.com/RanimAlyaseen/icelandic-asr-project3/tree/person4-evaluation/results/person4_evaluation) contains the confidence intervals, paired statistical comparisons, error analysis, methods, and graphs. I updated the README with the findings, limitations, and reproduction instructions and submitted the work through [Pull Request #3](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/3).
