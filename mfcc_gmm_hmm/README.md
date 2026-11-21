# MFCC + GMM-HMM Icelandic ASR

This package implements a character-level Icelandic recognizer using 39-dimensional MFCC, delta, and delta-delta features, one diagonal-covariance Gaussian mixture per character, and a left-to-right character HMM with a train-only smoothed character bigram language model. Whitespace is a character in the Icelandic orthographic inventory. A boundary state permits repeated characters.

The method uses weak, uniform transcript-to-frame alignment to initialize the per-character acoustic models. This is a reproducible classical baseline, not a phonetic lexicon or a manually aligned pronunciation model. The model card records this limitation.

## Setup

From the repository root, with Python 3.10–3.13 and Git LFS installed:

```powershell
git lfs install
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r mfcc_gmm_hmm\requirements.txt
```

For a one-utterance smoke run, fetch only that example instead of the full cleaned tree:

```powershell
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/train/007478/007478-0072202.flac" --exclude=""
```

## Run

Smoke fit on the first frozen train example and decode the first dev example:

```powershell
python -m mfcc_gmm_hmm.run --train --limit 1 --dev-limit 1
```

Train the configured model on the complete frozen train split, then select and record dev performance:

```powershell
python -m mfcc_gmm_hmm.run --train
```

Freeze the configuration and model revision before the single test evaluation:

```powershell
python -m mfcc_gmm_hmm.run --evaluate-test --confirm-frozen
```

The trainer refuses test evaluation without the explicit freeze confirmation and a completed dev-selected model. It reads references and IDs directly from the project's frozen JSONL manifests. Audio is loaded from the cleaned v1 LFS root.

Outputs are written under `results/mfcc_gmm_hmm/`: per-utterance CSV predictions, JSON run metadata with WER/CER and timing, a copied effective configuration, and the model card. Model weights are stored under `models/mfcc_gmm_hmm/` and are Git LFS tracked. They remain downloadable from this repository using ordinary `git lfs pull`; every run record includes the SHA-256 model hash and full configuration. Do not change settings after looking at test results.

The default configuration is intentionally modest: 2 mixture components, 20,000 maximum training frames per character, 25 ms frames, 10 ms hop, and a 13-coefficient MFCC stack with deltas. Model selection on dev compares WER and CER; the saved dev run is the configuration record used for the final test run.

## Frozen run

The full CPU fit used all 10,488 frozen train utterances (5,379,049 feature frames) and took 410.4 seconds. Dev selected language-model weight 1.5 from 0.5, 1.0, and 1.5. On dev (1,314 utterances), WER was 101.46%, CER 87.51%, and decoding took 89.4 seconds (RTF 0.0124). The single frozen test run (1,884 utterances) scored WER 101.47%, CER 87.74%, and took 135.9 seconds (RTF 0.0126).

These error rates are very high. The recognizer uses approximate uniform transcript-to-frame alignment and no curated Icelandic phonetic lexicon, so this run is a weak classical reference point rather than a competitive recognizer. See `results/mfcc_gmm_hmm/` for the exact predictions and timing records and `models/mfcc_gmm_hmm/model.joblib` for the LFS model artifact.
