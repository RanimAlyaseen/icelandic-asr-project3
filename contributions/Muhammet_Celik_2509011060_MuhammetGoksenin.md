# Muhammet Çelik

- Student ID **2509011060**
- GitHub **MuhammetGoksenin**
- Hugging Face **GokseninMuhammet**

## Pretrained baselines (Person 2)

I wrote [baselines/run_pretrained.py](../baselines/run_pretrained.py), which runs three published checkpoints on the frozen v1 dev and test manifests without any training by our group: `openai/whisper-small` (multilingual, zero-shot Icelandic), `language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h` (wav2vec 2.0 with a CTC head), and `Valdimarb13/whisper-small-icelandic`. Each run records the exact checkpoint revision, decoding settings, precision, batch size, GPU and package versions next to its per-clip predictions in [results/pretrained/](../results/pretrained/).

I chose the Whisper beam size on the dev split (beam 5 won for both Whisper models) and decoded the test split once with the frozen setting. The [notebook](../notebooks/person2_pretrained_baselines.ipynb) pulls the audio from this repository through Git LFS and needs no manual uploads; I ran it on Kaggle (one Tesla T4). [tests/test_pretrained.py](../tests/test_pretrained.py) checks that predictions join to the manifest by ID, keep leading zeros and Icelandic characters, and use the shared text normalization. I fixed the decoding of `Valdimarb13/whisper-small-icelandic`, whose old generation config has no language tables.

I documented which course baselines do not support Icelandic (Canary, Parakeet, Qwen3-ASR) and the possible training-data overlap of the published Icelandic checkpoints with our test clips.

## HuBERT + CTC (our training)

No published Icelandic HuBERT ASR checkpoint was available, so I fine-tuned the multilingual `utter-project/mHuBERT-147` encoder with a new CTC head over our train-only character vocabulary ([training/train_hubert_ctc.py](../training/train_hubert_ctc.py), [notebook](../notebooks/hubert_ctc_training.ipynb)). The best epoch was chosen by dev WER and the test split was decoded once: **test WER 68.46 %, CER 23.31 %**.

The first two full runs collapsed to all-blank output. I wrote [training/diagnose_hubert_ctc.py](../training/diagnose_hubert_ctc.py), which showed that full fine-tuning from a random head could not even memorize 32 clips, while a two-stage schedule (frozen encoder first, then unfrozen) could. I also added a plateau guard that stops a run whose dev WER stays near 100 %, which kept the failed runs to about one hour of GPU each. [tests/test_hubert_ctc.py](../tests/test_hubert_ctc.py) checks label padding, attention-mask handling and greedy CTC decoding.

The fine-tuned weights are published at [GokseninMuhammet/mhubert-147-icelandic-ctc-samromur](https://huggingface.co/GokseninMuhammet/mhubert-147-icelandic-ctc-samromur) (revision `6cf95c5810843ceeac3044038760f40eeb992879`), with a model card that lists the data, the training recipe, the results and the limitations.

## RNN-T (our training, from scratch)

The only transducer checkpoint on the course list does not support Icelandic, so I implemented an RNN-Transducer and trained it from random initialization on our 15 h train split ([training/train_rnnt.py](../training/train_rnnt.py), [notebook](../notebooks/rnnt_training.ipynb)): log-mel features, a convolutional subsampler and Transformer encoder, an LSTM prediction network, a joiner, `torchaudio`'s RNN-T loss and batched greedy decoding. Dev WER chose epoch 56 of 60 and the test split was decoded once: **test WER 68.95 %, CER 30.54 %**.

Learning from the HuBERT runs, the notebook first checks that the model memorizes 32 clips and stops if it cannot, and the plateau guard stops a run that does not learn; the full run then took about one hour on one T4 in Google Colab. [tests/test_rnnt.py](../tests/test_rnnt.py) checks the vocabulary, the features, the batching, the model shapes, greedy decoding against a scripted joiner, and `torchaudio`'s loss against a plain forward-algorithm implementation.

The weights are published at [GokseninMuhammet/rnnt-icelandic-samromur-scratch](https://huggingface.co/GokseninMuhammet/rnnt-icelandic-samromur-scratch) (revision `35089afe2355e035c184d0f12afe034cc2a20b4b`) with a model card.

## CTC encoder, attention encoder-decoder and hybrid CTC/attention (our training, from scratch)

Because the course lists these families separately from the self-supervised and pretrained models, I implemented and trained all three from scratch on the RNN-T pipeline, so the four neural families share features, encoder, data and training budget ([training/train_ctc_attention.py](../training/train_ctc_attention.py), [notebook](../notebooks/ctc_attention_training.ipynb)). The CTC mode decodes greedily; the attention mode uses beam search; the hybrid mode trains with 0.3 × CTC + 0.7 × attention and rescores the beam's n-best list with the CTC head.

Test results (decoded once each): **hybrid CTC/attention WER 55.57 %, CER 17.77 %**; CTC encoder WER 74.50 %, CER 22.88 %; the attention-only configuration did not learn a stable alignment in this experiment and was stopped by the plateau guard at epoch 16 (test WER 153.82 %). This attention-only result is a limitation of this experiment and does not imply that attention models generally cannot learn alignment. Hybrid CTC/attention achieved the lowest test WER among our models trained from scratch (55.57%). Our recovered fine-tuned Whisper checkpoint achieved a lower test WER overall (36.06%). I documented this result and the memorization-check issues honestly in the README and notebook. [tests/test_ctc_attention.py](../tests/test_ctc_attention.py) covers teacher forcing, CTC decoding, beam search, CTC rescoring and per-mode learning.

The weights are published at [GokseninMuhammet/ctc-attention-icelandic-samromur-scratch](https://huggingface.co/GokseninMuhammet/ctc-attention-icelandic-samromur-scratch) (revision `61ec3fd06db751fae48a56a94aef370bd63744b8`) with a model card.

## Documentation

I wrote the README section "Pretrained baselines, HuBERT + CTC, and neural models trained from scratch", including the comparison table of all neural models our group trained, the justification of WER and CER and the license note for the model weights, and added the new folders to the repository contents table.
