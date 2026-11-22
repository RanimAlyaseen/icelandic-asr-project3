# MFCC + GMM-HMM Icelandic ASR

This package implements a character-level Icelandic recognizer using 39-dimensional MFCC, delta, and delta-delta features, one diagonal-covariance Gaussian mixture per character, and a left-to-right character HMM with a train-only smoothed character bigram language model. Whitespace is a character in the Icelandic orthographic inventory. A boundary state permits repeated characters.

The method uses weak, uniform transcript-to-frame alignment to initialize the per-character acoustic models. This is a reproducible classical baseline, not a phonetic lexicon or a manually aligned pronunciation model. The model card records this limitation.

## Setup

From the repository root, with Python 3.10–3.13 and Git LFS installed:

```powershell
git lfs install
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r mfcc_gmm_hmm\requirements.txt
```

For a one-utterance smoke run, fetch only that example instead of the full cleaned tree:

```powershell
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/train/007478/007478-0072202.flac,data/cleaned/samromur-21.05-20h-v1/dev/004485/004485-0034220.flac" --exclude=""
```

## Run

Smoke fit on the first frozen train example and decode the first dev example:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.run --train --limit 1 --dev-limit 1
```

Train the configured model on the complete frozen train split, then select and record dev performance:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.run --train
```

Freeze the configuration and model revision before the single test evaluation:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.run --evaluate-test --confirm-frozen
```

The trainer refuses test evaluation without the explicit freeze confirmation and a completed dev-selected model. It reads references and IDs directly from the project's frozen JSONL manifests. Audio is loaded from the cleaned v1 LFS root.

Outputs are written under `results/mfcc_gmm_hmm/`: per-utterance CSV predictions, JSON run metadata with WER/CER and timing, a copied effective configuration, and the model card. Model weights are stored under `models/mfcc_gmm_hmm/` and are Git LFS tracked. They remain downloadable from this repository using ordinary `git lfs pull`; every run record includes the SHA-256 model hash and full configuration. Do not change settings after looking at test results.

The default configuration is intentionally modest: 2 mixture components, 20,000 maximum training frames per character, 25 ms frames, 10 ms hop, and a 13-coefficient MFCC stack with deltas. Model selection on dev compares WER and CER; the saved dev run is the configuration record used for the final test run.

## Training-only memorization and realignment diagnostic

The baseline initializes transcript-to-frame labels with equal-width character segments and keeps them fixed. To measure whether it can learn examples from its own training data, and to compare one transcript-constrained Viterbi realignment/re-estimation pass, run:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.diagnose_realign --train-limit 16 --dev-limit 8
```

This diagnostic fits a fresh model on the first 16 frozen train utterances, scores those same train utterances (memorization check), re-aligns only those train utterances to their known transcripts, re-estimates the character and boundary GMMs, and scores train plus an untouched 8-utterance dev sample. It never reads the test manifest or writes over the initial model or its original results. Diagnostic predictions and metrics go to `results/mfcc_gmm_hmm/diagnostics/`.

On the recorded 16-train/8-dev diagnostic, the uniform-initialized model did not memorize the train subset (WER 107.32%, CER 80.30%). One transcript-constrained realignment/re-estimation pass also did not help the held-out dev sample: WER rose from 108.06% to 116.13%, and CER from 91.03% to 98.21%. Scikit-learn reported one GMM convergence warning in this small run; it is recorded in the JSON report. This small experiment is evidence that one simple re-estimation pass is insufficient; it is not a final benchmark or a replacement for the original full-split model.

## Frozen run

The full CPU fit used all 10,488 frozen train utterances (5,379,049 feature frames) and took 410.4 seconds. Dev selected language-model weight 1.5 from 0.5, 1.0, and 1.5. On dev (1,314 utterances), WER was 101.46%, CER 87.51%, and decoding took 89.4 seconds (RTF 0.0124). The single frozen test run (1,884 utterances) scored WER 101.47%, CER 87.74%, and took 135.9 seconds (RTF 0.0126).

These error rates are very high. The recognizer uses approximate uniform transcript-to-frame alignment and no curated Icelandic phonetic lexicon, so this run is a weak classical reference point rather than a competitive recognizer. See `results/mfcc_gmm_hmm/` for the exact predictions and timing records and `models/mfcc_gmm_hmm/model.joblib` for the LFS model artifact.

## Separate iterative-alignment experiment

`iterative_alignment.py` uses the frozen baseline weights as an initialization. For each *train* clip it finds an ordered Viterbi path constrained by its reference characters (each occurrence occupies at least one frame), refits the Gaussian mixtures on these inferred frames, and repeats. This changes the audio alignment instead of merely changing the data format. It is character-based and does **not** use an Icelandic pronunciation dictionary; improvements are empirical, not guaranteed. The original baseline and its test predictions above remain unchanged.

After fetching the baseline model and cleaned train/dev audio with Git LFS, in the Python environment from the setup above run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gmm_alignment.py
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.iterative_alignment --limit 8 --dev-limit 3
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.iterative_alignment --iterations 2
```

The first training command is a smoke run and writes to separate `models/mfcc_gmm_hmm_aligned_smoke/` and `results/mfcc_gmm_hmm_aligned_smoke/` directories. The full command selects alignment iteration and decoder language-model weight by dev WER, then CER. It writes the selected model, dev predictions, and all candidate scores to `models/mfcc_gmm_hmm_aligned/` and `results/mfcc_gmm_hmm_aligned/`. Its run JSON records the baseline and selected model SHA-256 hashes, seed, configuration, environment, and whether the run was a smoke test. These generated directories are experiment outputs; before reporting a full result, commit their small JSON and prediction files and track the model weight with Git LFS. The script does not open the test set. Keep the existing test result as the original baseline; only evaluate an aligned model on test once the team has frozen a dev-selected full-training run.

For a fair dev comparison with a wider language-model weight grid, run this *after* the full alignment command:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.compare_alignment
```

It decodes all 1,314 dev clips with **both** the frozen uniform-init baseline and the selected aligned acoustic model, on identical decoder weights (0.5 through 24). `results/mfcc_gmm_hmm_aligned/dev_comparison.json` stores every candidate and the independently dev-selected settings. Separate per-clip dev prediction CSVs are saved for each model. This is exploratory model selection on dev, not evidence of test performance or statistical significance. The aligned model has not been evaluated on test.

### Full dev outcome

Both alignment passes fitted all 10,488 train clips with no alignment skips. The selected second-pass model has SHA-256 `5c94f91e0d4a0abbcda04d5a5740d5a623c9abda0b8dcdb5fc78a8af2c61fa8d`. The dev comparison is:

| Acoustic model | Dev decoder grid | Selected weight | Dev WER | Dev CER |
| --- | --- | ---: | ---: | ---: |
| Original uniform baseline | 0.5, 1.0, 1.5 | 1.5 | 101.46% | 87.51% |
| Aligned, two passes | 0.5, 1.0, 1.5 | 1.5 | 106.59% | 83.16% |
| Original uniform baseline | matched expanded grid | 6.0 | 99.94% | 93.28% |
| Aligned, two passes | matched expanded grid | 24.0 | 99.47% | 84.42% |

The expanded grid is `[0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24]`. It uses exactly the same dev clips, references, and normalization for both models. The paired utterance bootstrap of **baseline WER minus aligned WER** on these dev-selected predictions is **0.48 percentage points**, with an exploratory 95% interval of **0.30–0.66 points** (5,000 draws, seed 2105). Because decoder settings were selected on this same dev split, this interval cannot establish a test improvement or a confirmatory significance claim. WER near 100% remains unacceptable for practical ASR. The original frozen baseline test result is not replaced.

For a cheap reproducible uncertainty check from the committed predictions without decoding again:

```powershell
.\.venv\Scripts\python.exe -m mfcc_gmm_hmm.compare_alignment --stats-only
```
