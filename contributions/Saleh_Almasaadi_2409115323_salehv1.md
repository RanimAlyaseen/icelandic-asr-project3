# Saleh Almasaadi

- Student ID 2409115323
- GitHub salehv1

My contribution covers evaluation and statistical analysis for the Icelandic ASR project. I independently evaluated **ten ASR models** on the same 1,884 frozen test clips from 1,026 speaker/session groups. These included four pretrained and fine-tuned baseline systems, our recovered fine-tuned Whisper Small checkpoint, Person 2's RNN-T, scratch-trained CTC encoder, attention-only and hybrid CTC/attention models, and Person 5's MFCC + GMM-HMM model. I verified clip IDs, reference transcripts, and speaker/session groups and recalculated WER and CER using the shared `prepare.normalize_text` function.

I calculated **95% confidence intervals** using 10,000 speaker/session-clustered bootstrap resamples and performed **90 paired WER/CER statistical comparisons** using 10,000 session-level randomization tests with Holm correction. I also analyzed substitutions, deletions, insertions, and word-boundary errors, generated comparison graphs, and examined the 39 original Whisper test predictions containing digits without changing the frozen scoring policy.

I evaluated Person 3's recovered Whisper Small checkpoint 1000 on the development and test sets using a Tesla T4 GPU. It achieved 34.51% WER and 9.69% CER on the development set and 36.06% WER and 10.16% CER on the test set. Compared with the original Whisper Small baseline, the observed WER reduction was 50.76 percentage points (95% CI: 47.31–54.85). I documented the different decoding configurations used by the two models as a limitation.

I independently evaluated Person 5's MFCC + GMM-HMM model, obtaining 101.47% WER and 87.74% CER. I also evaluated Person 2's four additional models:

- RNN-T: 68.95% WER, 30.54% CER.
- Scratch-trained CTC encoder: 74.50% WER, 22.88% CER.
- Attention-only: 153.82% WER, 92.79% CER.
- Hybrid CTC/attention: 55.57% WER, 17.77% CER.

The hybrid CTC/attention model achieved the lowest WER among the models trained entirely from scratch by our group. It outperformed HuBERT + CTC by 12.89 percentage points in WER (95% CI: 11.63–14.15; Holm-adjusted p approximately 0.00900).

HuBERT + CTC and RNN-T did not differ significantly in WER after Holm correction (adjusted p = 0.55114), although HuBERT + CTC achieved significantly lower CER (adjusted p approximately 0.00900).

I updated and successfully tested the standalone Python evaluation script, [`evaluation/person4_evaluate.py`](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/evaluation/person4_evaluate.py), and the [evaluation notebook](https://github.com/RanimAlyaseen/icelandic-asr-project3/blob/person4-evaluation/notebooks/Person4_Icelandic_ASR_Evaluation.ipynb). Both support ten models without requiring retraining or GPU inference.

For reproducibility, the original baseline predictions and frozen references are retrieved from a fixed Git commit. The notebook also checks the recovered Whisper test predictions against a fixed SHA-256 checksum. Evaluation dependencies are pinned in `evaluation/requirements-person4.txt`.

The [evaluation results folder](https://github.com/RanimAlyaseen/icelandic-asr-project3/tree/person4-evaluation/results/person4_evaluation) contains all 20 WER/CER confidence-interval rows, 90 paired comparisons, error analysis, methods, and comparison chart.

I updated the shared README with the ten-model findings, corrected the outdated statements about Whisper development evaluation, and documented the evaluation methods, limitations, and reproduction instructions. My work is submitted through [Pull Request #3](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/3).