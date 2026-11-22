# Samrómur 21.05 Icelandic ASR dataset preparation

**Dataset.** Use **samromur-21.05-20h-v1**, the frozen seed-2105 selection verified at commit 55f5a68. Its 15/2/3-hour targets, selected IDs, and references are unchanged. The professor accepts the bounded raw subset and Git LFS in this public repository. [Download the audio](#download-the-audio) · [Raw scope and storage](DATA_STORAGE.md)

This repository contains the shared 20-hour Samrómur selection for Whisper and wav2vec 2.0. Seed **2105** selected clips within the official train, dev, and test partitions. The dataset is Icelandic audio-to-text ASR.

| Split | Speaker/session IDs | Utterances | Decoded hours |
|---|---:|---:|---:|
| [Train](output/train.jsonl) | 2,796 | 10,488 | 15.00002 |
| [Dev](output/dev.jsonl) | 963 | 1,314 | 2.00088 |
| [Test](output/test.jsonl) | 1,026 | 1,884 | 2.99971 |

The [split integrity report](output/split_integrity.json) confirms **zero shared speaker/session IDs** for train/dev, train/test, and dev/test. The selected test set has **zero utterances** whose exact original sentence is in selected train and **zero** whose normalized scoring text is in selected train. These are separate comparisons.

The [publisher's README](data/raw/samromur_21.05/README) documents the official partitions as speaker-separated and uses session IDs as speaker IDs. The cleaner checks each selected utterance against the original partition and session in the release metadata. It selects within those partitions. The identifiers establish separation of recorded sessions and cannot identify a person across sessions. Any future new split design must keep each complete speaker/session group together.

Use **dev** to select checkpoints, settings, and hyperparameters. Use **test** after freezing those choices for the final model comparison with shared v1 references. [Required baseline checklist](BASELINES.md)

## Download the audio

With Git LFS installed, clone without downloading audio and then fetch the paths you need. The example uses PowerShell.

~~~powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/RanimAlyaseen/icelandic-asr-project3.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location icelandic-asr-project3

# For model preprocessing, fetch the cleaned v1 audio
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""

# For raw-to-cleaned reproduction or review of shipped exclusions, fetch the raw subset
git lfs pull --include="data/raw/samromur_21.05/**" --exclude=""
~~~

For a single split, append its name before the wildcard, for example **data/cleaned/samromur-21.05-20h-v1/dev/**. Git LFS reuses cached objects shared by the raw and cleaned roots. The [storage notes](DATA_STORAGE.md) give byte counts and quota details.

A fresh public clone on Windows fetched every object with credential helpers disabled. The checkout decoded and verified all 17,307 raw FLACs, matched the 13,686 cleaned files to their release bytes, and reproduced the scoped reports and six v1 ASR artifacts. All 12 independent tests and the 12-artifact gzip replay also passed there. [Verification record](output/lfs_verification.json)

## Get started

Use the hydrated cleaned LFS root, the local full release, or the **audio/** directory of the optional LF handoff. Each has train, dev, and test directories and works with the frozen manifests. The Windows verification environment used Python 3.14.10, NumPy 2.5.5, SoundFile 0.14.0 with libsndfile 1.2.2, and Transformers 5.23.0.

Use a Python **3.14** environment for the data pipeline and its tests. Run from the repository root on Windows.

```powershell
py -3.14 -m venv .venv
$venvPy = '.\.venv\Scripts\python.exe'
& $venvPy -m pip install --only-binary=:all: -r requirements-windows-py314.lock.txt
& $venvPy -m pip check
```

On macOS or Linux with Python 3.14, use `python3.14 -m venv .venv`, `.venv/bin/python`, and [requirements.txt](requirements.txt). See the [Windows dependency lock](requirements-windows-py314.lock.txt).

Use a separate [Python 3.13 modeling environment](#modeling-environment-and-tests) for the pretrained and HuBERT tests.

## Repository contents

| Path | Purpose |
|---|---|
| [README.md](README.md) | Setup, review, and reproduction instructions |
| [BASELINES.md](BASELINES.md) | Required modeling checklist and official language support evidence |
| [DATA_STORAGE.md](DATA_STORAGE.md) | Measured data sizes, availability, and storage decision |
| [LICENSE](LICENSE) | MIT terms for project code and CC BY 4.0 dataset attribution |
| [.gitignore](.gitignore) and [.gitattributes](.gitattributes) | Local-file exclusions and byte-preserving checkout of generated artifacts |
| [audit.py](audit.py) and [prepare.py](prepare.py) | Release audit, cleaning, and seeded subset generation |
| [subset.py](subset.py) | Bounded raw membership and raw-to-cleaned processing with split checks |
| [preprocess.py](preprocess.py) | Audio loader and Whisper and wav2vec 2.0 preprocessing |
| [verify.py](verify.py) and [verify_handoff.py](verify_handoff.py) | Verification against the release or the extracted handoff |
| [requirements.txt](requirements.txt) and [requirements-windows-py314.lock.txt](requirements-windows-py314.lock.txt) | Pinned dependencies and tested Windows environment |
| [requirements-modeling.txt](requirements-modeling.txt) | Person 2's pinned environment for pretrained baselines, HuBERT, and their tests. Person 3's Whisper dependencies are in [training/requirements-person3.txt](training/requirements-person3.txt). |
| [NLP_Project3_Person3_Whisper_Clean.ipynb](NLP_Project3_Person3_Whisper_Clean.ipynb) | Person 3's Whisper Small fine-tuning notebook and saved training outputs |
| [training_metrics.json](training_metrics.json) | Historical Whisper training and validation summary transcribed from the notebook output. Its evaluation status describes that snapshot. |
| [tests/](tests/) | Separate data-pipeline, pretrained/HuBERT/RNN-T/CTC-attention, and classical-alignment unit tests |
| [output/](output/) | Frozen manifests and references, vocabulary, audit, review queues, and verification records |
| [data/](data/) | Raw subset and original documents under **raw/samromur_21.05/** and cleaned v1 FLACs under **cleaned/samromur-21.05-20h-v1/** |
| [review/](review/) | Manual listening decisions kept outside the generated outputs |
| [contributions/](contributions/) | Individual work records authored and committed by team members |
| [baselines/](baselines/) | Inference code for published pretrained checkpoints (baselines, no training by the group) |
| [training/](training/) | HuBERT training and diagnostics, RNN-T, CTC, attention and hybrid CTC/attention training from scratch, recovered Whisper evaluation, and Person 3's Whisper dependencies |
| [mfcc_gmm_hmm/](mfcc_gmm_hmm/) | MFCC features, GMM-HMM training and inference, configuration, dependencies, and model card |
| [models/mfcc_gmm_hmm/](models/mfcc_gmm_hmm/) | Reproducible trained GMM-HMM weights tracked with Git LFS |
| [models/mfcc_gmm_hmm_aligned/](models/mfcc_gmm_hmm_aligned/) | Separately trained transcript-constrained alignment experiment weights, tracked with Git LFS |
| [notebooks/](notebooks/) | Person 2's pretrained-baseline, HuBERT, RNN-T and CTC/attention/hybrid Kaggle/Colab notebooks |
| [results/](results/) | Per-clip predictions, run records, and training logs |

### MFCC + GMM-HMM baseline

The classical Icelandic character recognizer uses 13 MFCCs with delta and delta-delta features, diagonal Gaussian mixtures, a left-to-right character HMM, and a train-only character bigram language model. Its initialization uses uniform transcript-to-frame alignment; it does not use a curated phonetic lexicon. The completed CPU run trained on 10,488 utterances, selected its language-model weight on 1,314 dev utterances, then evaluated test once on 1,884 utterances. Test WER was **101.47%** and CER **87.74%**, so the current recipe is a weak reference point. See [`mfcc_gmm_hmm/README.md`](mfcc_gmm_hmm/README.md) for setup, model access, full metrics, and predictions in [`results/mfcc_gmm_hmm/`](results/mfcc_gmm_hmm/).

A follow-up diagnostic checked memorization on 16 train utterances and tried one transcript-constrained realignment/re-estimation pass, without reading test data or replacing the initial model. The model did not memorize the train sample (WER 107.32%, CER 80.30%), and the realignment pass worsened the 8-utterance dev sample (WER 108.06% to 116.13%, CER 91.03% to 98.21%). See the [diagnostic script](mfcc_gmm_hmm/diagnose_realign.py) and [separate diagnostic outputs](results/mfcc_gmm_hmm/diagnostics/).

A separate two-pass, transcript-constrained alignment experiment trained on the same 10,488 train clips and selected settings on the same 1,314 dev clips. At the original decoder weight grid (0.5, 1.0, 1.5), its best dev WER was **106.59%**, worse than the original baseline's **101.46%**. A wider *matched* dev decoder grid selected **99.94% WER / 93.28% CER** for the original model and **99.47% WER / 84.42% CER** for the aligned model. The WER difference is just 0.48 percentage points, both models remain very weak, and the exploratory paired dev interval is optimistic because settings were selected on dev. The aligned model has **not** been run on test; the frozen baseline test score above is unchanged. Full code, the candidate grid, model hashes, dev predictions, and caveats are in [`mfcc_gmm_hmm/README.md`](mfcc_gmm_hmm/README.md) and [`results/mfcc_gmm_hmm_aligned/`](results/mfcc_gmm_hmm_aligned/).



## Load and preprocess a clip

The multilingual Whisper preprocessing checkpoint is **openai/whisper-tiny** at revision **169d4a4341b33bc18d8881c4b69c2e104e1cc0af**.

```python
from pathlib import Path

from transformers import Wav2Vec2FeatureExtractor

from preprocess import (
    ctc_tokenizer,
    iter_examples,
    wav2vec2_example,
    whisper_components,
    whisper_example,
)

audio_root = Path("data/cleaned/samromur-21.05-20h-v1")
output = Path("output")
example = next(iter_examples(audio_root, output / "dev.jsonl"))

extractor, tokenizer = whisper_components()
whisper = whisper_example(example, extractor, tokenizer)

ctc_extractor = Wav2Vec2FeatureExtractor(
    sampling_rate=16_000, do_normalize=True, return_attention_mask=True
)
wav2vec = wav2vec2_example(example, ctc_extractor, ctc_tokenizer(output / "ctc_vocab.json"))
```

[preprocess.py](preprocess.py) loads each FLAC as a mono 16 kHz float32 waveform. Whisper uses the pinned multilingual feature configuration and tokenizer, with Icelandic as the language and `transcribe` as the task. The first call fetches the public preprocessing assets. The wav2vec 2.0 helper uses the [CTC vocabulary derived from train](output/ctc_vocab.json), which covers all characters in the frozen dev/test references. For a wav2vec 2.0 checkpoint with a fixed CTC vocabulary, pass its matching extractor and tokenizer.

Use the shared [evaluation references (TSV)](output/evaluation_references.tsv) or [evaluation references (JSONL)](output/evaluation_references.jsonl) for both models. Join predictions by `split` and `id`, preserving leading zeros. Each manifest preserves the raw and release-normalized prompts. The scoring text lowercases Unicode, applies NFC, treats punctuation and symbols as spaces, and collapses whitespace. Apply `normalize_text` from [prepare.py](prepare.py) to model hypotheses for comparable WER or CER. Icelandic characters and diacritics are preserved.

## Focused audio-quality review

Ranim Alyaseen completed audio-quality listening for all 60 recordings in the focused queue on 21 November 2026. All recordings were available. [Manual review log](review/focused_manual_review.tsv)

| Automatic flag | Usable | Hard to hear | Uncertain |
|---|---:|---:|---:|
| Near-silence exclusions | 10 | 43 | 0 |
| Retained possible-clipping cases | 6 | 0 | 1 |
| **Total** | **16** | **43** | **1** |

Clip `0049034` needs a second audio-quality opinion. Icelandic transcript agreement was unassessed for all 60 clips and is recorded as `uncertain`. The recommendations leave frozen v1 unchanged.

For follow-up reviews, preserve leading zeros when importing clip IDs. Preprocessing stores the [automatic measurements](output/focused_review.tsv) separately from the manual decisions.

1. **Find and play the FLAC.** Append the row's **audio_path** to the hydrated raw subset or to the root of your full local release. The raw subset includes all seven retained clipping cases and two of the near-silence cases, **0008401** and **0076273**. Use the full release for the other 51 near-silence recordings.
2. **Record audio quality.** Any teammate can listen for audible speech, gaps, competing voices, clipping, noise, and abrupt cutoffs. Enter your name or initials as reviewer and record **usable**, **issue**, or **uncertain**. Add a timestamp and what you heard where useful. A low RMS score can occur in intelligible speech.
3. **Check transcript agreement.** A reviewer who understands Icelandic can find the original prompt in the [release metadata](data/raw/samromur_21.05/metadata.tsv) under the same ID in the **sentence** column. Record **match**, **mismatch**, or **uncertain**, and note the phrase or timestamp. Keep each unreviewed decision **pending**.
4. **Recommend an action.** Note whether to retain the clip, keep it excluded, or propose a change. Keep v1 frozen. A verified membership or transcript correction needs a documented reason and a new dataset version before its use in benchmark comparisons.

For example, run this from the repository root in PowerShell and replace the release path and ID for each clip.

~~~powershell
$release = 'C:\path\to\samromur_21.05'
$id = '0025341'
$row = Import-Csv -LiteralPath './review/focused_manual_review.tsv' -Delimiter ([char]9) |
    Where-Object { $_.id -eq $id }
if (-not $row) { throw "Review ID not found: $id" }
Invoke-Item -LiteralPath (Join-Path $release $row.audio_path)
~~~

The broader listening review remains unassessed outside these 60 clips. The 541-clip evaluation checklist covers 315 test and 226 dev recordings. Two were covered by the focused review, leaving 539 unassessed. Its paths start with **audio/** so they resolve from the handoff root. For the original release root, drop that prefix. [Evaluation checklist](output/flagged_dev_test_review.tsv) · [Full listening queue, including flagged train clips and unflagged controls](output/listening_queue.tsv)

The [focused review queue](output/focused_review.tsv) documents **53 near-silence exclusions and seven selected possible-clipping flags**. All 53 met the RMS threshold, and 15 also met the quiet-fraction threshold. The seven selected clips have 0.1095% to 0.3238% full-scale samples. Quiet speech may still be intelligible and full-scale samples may be inaudible.

The exclusion log contains **150 unique utterances**, with 127 train, 18 dev, and five test. Of those, 94 were withheld only for the release validation flag, one for both validation and digital silence, two for the metadata empty status, and 53 for near-silence.

## Cleaning policy

- The audit checks the FLAC bytes against the release MD5 list and fully decodes each recording. It records format, sample count, sound levels, and a SHA-256 of the PCM that includes sample rate and channel count. The hash detects exact duplicates. Edited or near-duplicate recordings need other checks.
- Selection excludes absent metadata or prompt text, validation values other than **1.0**, empty values other than **0**, missing or unreadable audio, decode failures, checksum mismatches, audio outside mono 16 kHz FLAC PCM16, or decoded durations outside 0.5–30 seconds. A missing release checksum is a review flag.
- A quiet window has RMS below −50 dBFS over 20 ms, with the trailing partial window weighted by its sample count. Digital silence means zero frames or all-zero PCM. Near-silence means whole-clip RMS at or below −55 dBFS or at least 98% quiet time. These exclusions are automated decisions pending listening.
- Among otherwise usable signals, flag RMS at or below −35 dBFS, at least 75% quiet time, a quiet edge of at least 1.5 seconds, or at least 0.1% full-scale samples. Flag invalid duration metadata, a measured duration difference greater than 0.05 seconds, or a sample-rate disagreement.
- Withhold exact PCM groups with conflicting scoring text. For matching labels, retain one eligible copy in test/dev/train priority order, breaking ties by seeded order. The same split priority screens later selections against selected session IDs, raw prompts, and scoring references. The IDs establish session separation. A person could have recorded in several sessions.
- The sampler sorts by SHA-256 of **seed/audio_path** and takes the prefix closest to each duration target from its official partition. This revision preserves the existing IDs and references. Any later membership or reference change needs a dataset version and a documented reason before model comparisons use it.

## Regenerate cleaned audio from the raw subset

The raw input is the frozen v1 membership plus the first release utterance by numeric ID from every represented split and session. Membership uses the original metadata and ignores audio flags. It does not construct new splits. The raw files and full release metadata preserve the original bytes. The publisher says human listeners verified the published audio against the prompts. Every selected v1 row carries the release validation flag and its original text alongside derived scoring text. [Upstream release README](data/raw/samromur_21.05/README)

With the raw LFS files hydrated, run:

~~~powershell
& $venvPy -X utf8 subset.py clean
~~~

This reads and decodes all raw inputs, verifies release MD5s, applies the same automated cleaning rules, and fails if a frozen clip or artifact changes. It writes the **13,686** cleaned FLACs under **reproduced/cleaned/samromur-21.05-20h-v1/**, plus rebuilt manifests, references, vocabulary, and logs under **reproduced/subset/**. The output is independent of the committed compressed audio audit. The [scoped removal log](output/raw_subset_removals.tsv) records two near-silence exclusions and one unvalidated utterance, while **3,618** eligible session examples stay outside frozen v1. The 150 historical exclusions describe the entire release.

To reconstruct the inputs from a local full release, run **subset.py stage --source-release** followed by the release directory path. It verifies the three documents and each original FLAC checksum before copying the bounded membership into the raw data directory.

## Replay the frozen artifacts from a fresh checkout

The [upstream README](data/raw/samromur_21.05/README), [metadata.tsv](data/raw/samromur_21.05/metadata.tsv), and [md5s.txt](data/raw/samromur_21.05/md5s.txt) are committed byte for byte. The release MD5 list confirms the README and metadata. All three SHA-256 hashes match the committed audit header.

After installing the Python 3.14 data dependencies, run these commands from the repository root. Select [test_pipeline.py](tests/test_pipeline.py) explicitly so discovery stays in the data suite.

~~~powershell
& $venvPy -X utf8 -m unittest discover -s tests -p 'test_pipeline.py' -v
& $venvPy -X utf8 verify.py --replay-only --root data/raw/samromur_21.05 --output output
~~~

On macOS or Linux with the data environment set up, use:

~~~bash
.venv/bin/python -X utf8 -m unittest discover -s tests -p 'test_pipeline.py' -v
.venv/bin/python -X utf8 verify.py --replay-only --root data/raw/samromur_21.05 --output output
~~~

The 12 data tests use small fixtures. They cover decoding failures, quality exclusions, split integrity, raw-to-cleaned reproduction, and deterministic artifacts. They run without PyTorch, release audio, or a GPU.

Replay uses the committed gzip audit and the three release documents. It compares regenerated manifests, references, vocabulary, and quality reports byte for byte in a temporary folder. It leaves the existing real-audio verification record intact. The decoded audio measurements come from the stored audit.

To keep a copy of the rebuilt artifacts, use a new output directory.

~~~powershell
New-Item -ItemType Directory -Force reproduced/replay
Copy-Item output/audio_audit.jsonl.gz reproduced/replay/
& $venvPy -X utf8 prepare.py --root data/raw/samromur_21.05 --output reproduced/replay --seed 2105
~~~

## Re-audit the original audio

Use a clean extraction of the release. The [data inventory](DATA_STORAGE.md) documents extra copies found outside the official paths in the local extraction.

To repeat the full audit and seed-2105 selection, use the full Samrómur 21.05 release. Write fresh results to a separate folder so they can be compared with the tracked outputs.

```powershell
$release = 'C:\path\to\samromur_21.05'
& $venvPy -X utf8 audit.py --root $release --output reproduced
& $venvPy -X utf8 prepare.py --root $release --output reproduced --seed 2105
& $venvPy -X utf8 verify.py --root $release --output reproduced
```

[audit.py](audit.py) inventories and decodes the full release. [prepare.py](prepare.py) applies exclusion rules, assesses overlap, and builds the seeded subsets and references. [verify.py](verify.py) checks the result against the full release. [verify_handoff.py](verify_handoff.py) is the verifier used in the extracted team handoff with its bundled audio and Whisper assets.

- [Quality report](output/quality_report.md), [machine-readable summary](output/quality_summary.json), and [prompt overlap](output/prompt_overlap.tsv)
- [Per-recording audio audit (gzip-compressed JSONL)](output/audio_audit.jsonl.gz). Python's `gzip.open` reads the full audit directly.
- [Preparation verification](output/verification.json) and [portable handoff verification](output/handoff_verification.json)

Re-decoding the entire release needs all 100,000 local FLACs. Running **verify.py** without **--replay-only** also opens the selected audio and checks features with the pinned Whisper assets. Both **prepare.py** and **verify.py** accept a plain or gzip audit, with a plain audit taking precedence. Generated artifacts use UTF-8 with LF line endings. Pass **--offline** to **verify.py** once the pinned assets are already local.

Run the handoff verifier against the inner folder extracted from the LF ZIP. It needs the packaged **audio/** split directories, **output/**, and **assets/whisper/** with its checksum file.

~~~powershell
& $venvPy -X utf8 verify_handoff.py --package-root 'C:\path\to\Samromur_ASR_Handoff'
~~~

Earlier Windows verification re-decoded all 100,000 official paths and matched the committed decompressed audit. It checked all selected clips and labels and ran both feature extractors on 30 clips. The earlier handoff check used a fresh extraction and local Whisper assets. The fresh-checkout replay above is a separate check of regeneration from stored audit results.

## Pretrained baselines, HuBERT + CTC, and neural models trained from scratch

The eight systems below have committed full test predictions on `main` for all 1,884 frozen v1 utterances. They use the shared [evaluation references](output/evaluation_references.tsv) and [text normalization](prepare.py). The dev columns report full development decoding with the chosen settings. See also the [recovered Whisper Small checkpoint evaluations](#recovered-checkpoint-evaluation).

Per-clip predictions and run records are in [results/pretrained/](results/pretrained/), [results/hubert_ctc/](results/hubert_ctc/), [results/rnnt/](results/rnnt/), [results/ctc_scratch/](results/ctc_scratch/), [results/attention_scratch/](results/attention_scratch/) and [results/hybrid_ctc_attention/](results/hybrid_ctc_attention/). The pretrained inference records and final HuBERT decoding record a Tesla T4. HuBERT training used two T4s. RNN-T, CTC, attention and hybrid training and decoding used one T4 on Google Colab.

Every Hugging Face checkpoint is pinned to a full commit hash in the code (`REVISIONS` in
[`baselines/run_pretrained.py`](baselines/run_pretrained.py), `Config.base_revision` in
[`training/train_hubert_ctc.py`](training/train_hubert_ctc.py)), including the `openai/whisper-small`
generation config used as a fallback for `Valdimarb13/whisper-small-icelandic`. A script refuses to run with an
unpinned checkpoint, and [`tests/test_pretrained.py`](tests/test_pretrained.py) checks the pins.

### Modeling environment and tests

Person 2's pretrained and HuBERT runs used Python 3.13.16, PyTorch 2.11.0 with CUDA 13.0, Transformers 5.23.0, and Accelerate 1.15.0. Use [requirements-modeling.txt](requirements-modeling.txt) in a separate Python **3.13** environment. The two unit test files work with the CPU build of PyTorch. Person 3's Whisper dependencies are recorded in [training/requirements-person3.txt](training/requirements-person3.txt).

~~~powershell
py -3.13 -m venv reproduced/venv-modeling
$modelPy = './reproduced/venv-modeling/Scripts/python.exe'
& $modelPy -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& $modelPy -m pip install -r requirements-modeling.txt
& $modelPy -m pip check
& $modelPy -X utf8 -m unittest discover -s tests -p 'test_pretrained.py' -v
& $modelPy -X utf8 -m unittest discover -s tests -p 'test_hubert_ctc.py' -v
~~~

For Linux CPU tests, create the environment with `python3.13 -m venv reproduced/venv-modeling` and use `reproduced/venv-modeling/bin/python` for the same commands. The six [pretrained tests](tests/test_pretrained.py) and four [HuBERT tests](tests/test_hubert_ctc.py) cover checkpoint pins, ordered predictions, text normalization, and CTC vocabulary, padding, masking, and decoding. Unit tests need no audio or downloaded model weights.

For Person 2's GPU runs, use the [pretrained notebook](notebooks/person2_pretrained_baselines.ipynb) or [HuBERT training notebook](notebooks/hubert_ctc_training.ipynb). They pin the data and code revisions, fetch the required audio through Git LFS, and install the modeling dependencies. For local GPU runs, install the PyTorch 2.11.0 CUDA build from the official wheel index that matches the runtime and hydrate the required split. [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2110)

For example, to rerun the wav2vec 2.0 dev baseline on Linux with CUDA 13.0, use a separate checkout because the runner writes to `results/pretrained/`.

~~~bash
python3.13 -m venv reproduced/venv-modeling-gpu
reproduced/venv-modeling-gpu/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
reproduced/venv-modeling-gpu/bin/python -m pip install -r requirements-modeling.txt
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/dev/**" --exclude=""
reproduced/venv-modeling-gpu/bin/python baselines/run_pretrained.py --model wav2vec2_is --split dev
~~~

### Baselines: published checkpoints, no training by our group

Code: [`baselines/run_pretrained.py`](baselines/run_pretrained.py), notebook:
[`notebooks/person2_pretrained_baselines.ipynb`](notebooks/person2_pretrained_baselines.ipynb).

| System | Checkpoint | Revision | Decoding | Dev WER | Test WER | Test CER |
|---|---|---|---|---:|---:|---:|
| Whisper small, multilingual, zero-shot | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | `973afd24965f72e36ca33b3055d56a652f456b4d` | beam 5, `language="is"` | 85.77 % | 86.82 % | 36.35 % |
| wav2vec 2.0 XLSR-53 + CTC, fine-tuned for Icelandic by LVL | [`language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h`](https://huggingface.co/language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h) | `f8de0cdc7a09ef45362f32c3514afb2edefca68d` | greedy CTC, no LM | 7.11 % | **7.76 %** | 1.86 % |
| Whisper small, fine-tuned for Icelandic by a third party | [`Valdimarb13/whisper-small-icelandic`](https://huggingface.co/Valdimarb13/whisper-small-icelandic) | `e078378ec8dfc6ebd1860d549f70bf132f030f64` | beam 5, `language="is"` | 16.42 % | 21.75 % | 5.60 % |

Beam size for both Whisper models was chosen on dev from {1, 5}; beam 5 had the lower dev WER for both
(whisper-small 89.36 → 85.77 %, whisper-small-icelandic 17.95 → 16.42 %).
`Valdimarb13/whisper-small-icelandic` ships an old generation config without language tables, so its decoding
uses the generation config of `openai/whisper-small` (recorded as `generation_config_from` in its run record).

**Possible test overlap.** The two Icelandic checkpoints were trained by others on Samrómur data:
the LVL wav2vec 2.0 model on 967 h of Samrómur Milljón, and `Valdimarb13/whisper-small-icelandic` on
`language-and-voice-lab/samromur_asr` (its card does not name the training split). Our test clips come from
Samrómur 21.05. Overlap with the external training sets is possible and has not been established at the clip level.
Read these published-checkpoint scores with that caveat when comparing them with models trained on our 15 h split.

### Course baselines that do not support Icelandic (not run)

As the course instructions require, these were not run with a wrong language setting:

| Model | Model card | Icelandic |
|---|---|---|
| `nvidia/canary-1b-v2` | https://huggingface.co/nvidia/canary-1b-v2 | not in the supported languages (25 European languages) |
| `nvidia/parakeet-tdt-0.6b-v3` | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | not in the supported languages (25 European languages) |
| `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | [1.7B card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [0.6B card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Icelandic is absent from the documented language lists |

### HuBERT + CTC, fine-tuned for Icelandic by our group

No published Icelandic HuBERT ASR checkpoint was found, and a bare HuBERT encoder cannot transcribe, so we
fine-tuned one. This is a course baseline family that we trained ourselves.

- Base encoder: [`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147) (HuBERT base, 95M parameters, 147 languages), revision `7ad3fc0bc5106c58c9c13526abccad527150d135`.
- A new CTC head over the train-only character vocabulary ([`output/ctc_vocab.json`](output/ctc_vocab.json), 38 symbols).
- Two-stage fine-tuning on the 15 h train split: encoder frozen and head trained alone for 1,000 updates (head LR 1e-3), then the transformer encoder unfrozen (LR 5e-5). 30 epochs, best epoch chosen by dev WER (epoch 29), seed 2105, 2 × T4, about 5.8 h.
- Code: [`training/train_hubert_ctc.py`](training/train_hubert_ctc.py), notebook: [`notebooks/hubert_ctc_training.ipynb`](notebooks/hubert_ctc_training.ipynb), tests: [`tests/test_hubert_ctc.py`](tests/test_hubert_ctc.py).
- Fine-tuned weights (378 MB, too large for the repository's LFS quota): [`GokseninMuhammet/mhubert-147-icelandic-ctc-samromur`](https://huggingface.co/GokseninMuhammet/mhubert-147-icelandic-ctc-samromur), revision `6cf95c5810843ceeac3044038760f40eeb992879`, `model.safetensors` SHA-256 `a9fffe14358a03465dca6243a4598047c01ff1c4180c789c4e8ebeedaa424075`.

| System | Decoding | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| mHuBERT-147 + CTC (ours, 15 h) | greedy CTC, no LM | 66.30 % | 22.55 % | **68.46 %** | **23.31 %** |

[Full dev run](results/hubert_ctc/model_info/hubert_ctc_dev.json) · [Full test run](results/hubert_ctc/model_info/hubert_ctc_test.json)

How we got here: full fine-tuning from a random CTC head collapsed to all-blank output (dev WER 100 % for 6 epochs, in two runs).
[`training/diagnose_hubert_ctc.py`](training/diagnose_hubert_ctc.py) showed the same model could not even memorize 32 clips
that way, while the two-stage schedule memorized them within 800 updates. Both failed runs were stopped automatically by a
plateau guard (dev WER still ≥ 98 % at epoch 6), which saved GPU quota.

**Two dev WER numbers for the same checkpoint.** Dev WER is 61.31 % during checkpoint selection and 66.30 % in the
final decoding. Both come from the same weights (epoch 29, `checkpoint-9512`) on the same 1,314 dev clips with the same
normalization and greedy decoding. They differ only in how clips are batched:

| | Checkpoint selection (Trainer evaluation) | Final decoding (`decode_split`) |
|---|---|---|
| Where | [`training_log.json`](results/hubert_ctc/training_log.json), `eval_wer` at epoch 29 | [`hubert_ctc_dev.json`](results/hubert_ctc/model_info/hubert_ctc_dev.json) |
| Batching | manifest order, 16 clips per GPU on 2 GPUs | sorted by duration, 16 clips, 1 GPU |
| Dev WER / CER | 61.31 % / 19.61 % | 66.30 % / 22.55 % |

The mHuBERT-147 feature encoder uses group normalization and takes no attention mask, so the zero padding added to
shorter clips in a batch changes the features the model sees. In manifest order, clips of very different lengths share a
batch, as they did during training; sorted by duration, clips get almost no padding. The model was trained with padded
batches, which is the most likely reason it scores better under the matching condition; we did not test this separately. The 61.31 % was used only to choose the epoch. We report
the final-decoding numbers (66.30 % dev, 68.46 % test) from the same HuBERT decoding path. A batch-size-one comparison remains untested.
Dev WER was still falling slowly at epoch 30, so longer training would probably help. CER is far below WER: most errors are
merged or split words and single-letter spelling mistakes that a language model would fix.

### RNN-T (transducer), trained from scratch by our group

The course's only transducer checkpoint, `nvidia/parakeet-tdt-0.6b-v3`, does not support Icelandic, so we implemented and
trained one from random initialization. It uses no pretrained weights and no language model.

- Model: 80-dim log-mel features → two stride-2 convolutions (4× subsampling) → 8-layer pre-norm Transformer encoder (d=256);
  LSTM prediction network (320); additive joiner over the same 38-symbol train-only vocabulary as HuBERT. 9.19M parameters.
- Loss: `torchaudio.functional.rnnt_loss` (torchaudio 2.11.0). Decoding: batched greedy search, at most 5 symbols per frame.
- Training: 10,486 train utterances (14.99 h; 2 clips over 16 s left out), SpecAugment, AdamW, peak LR 1e-3 with 2,000 warmup
  steps and cosine decay, fp16, seed 2105. 60 epochs (13,740 updates); dev WER every 2 epochs chose epoch 56.
  One Tesla T4 on Google Colab, 1 h 1 min of training, peak GPU memory 2.6 GB.
- Before full training, the notebook checks that the model memorizes 32 train clips (loss 431 → 0.018, exact transcripts) and
  stops if it cannot; a plateau guard would also stop a run whose dev WER stays ≥ 98 % at epoch 16.
- Code: [`training/train_rnnt.py`](training/train_rnnt.py) (code commit `6f702ce9160c443ef18f6b378453fafd743bb50a`), notebook:
  [`notebooks/rnnt_training.ipynb`](notebooks/rnnt_training.ipynb), tests: [`tests/test_rnnt.py`](tests/test_rnnt.py), including a check of
  torchaudio's loss against a plain forward-algorithm implementation.
- Weights (37 MB): [`GokseninMuhammet/rnnt-icelandic-samromur-scratch`](https://huggingface.co/GokseninMuhammet/rnnt-icelandic-samromur-scratch),
  revision `35089afe2355e035c184d0f12afe034cc2a20b4b`, `best.pt` SHA-256 `ad9392a444809e2f47875b354f44fb6eeac7c7af7b23fcb99eb21293ce312b53`.
  Released under CC BY 4.0, following the Samrómur training data; it is not derived from another model.

| System | Decoding | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| RNN-T from scratch (ours, 15 h) | greedy, no LM | 65.50 % | 28.61 % | **68.95 %** | **30.54 %** |

[Full dev run](results/rnnt/model_info/rnnt_dev.json) · [Full test run](results/rnnt/model_info/rnnt_test.json) · [Training curves](results/rnnt/training_curves.png)

Checkpoint selection and the final dev decoding use the same decoding function and batching, so the dev WER during training
equals the reported dev WER. Trained only on 15 h without any pretraining, the RNN-T reaches about the same WER as the
pretrained HuBERT + CTC (68.95 % vs 68.46 %) but a higher CER (30.54 % vs 23.31 %). Dev WER was still falling slowly at
epoch 60, so longer training would probably help.

### CTC encoder, attention encoder-decoder and hybrid CTC/attention, trained from scratch by our group

The remaining neural course families are trained from random initialization with the RNN-T pipeline: the same log-mel
features, encoder (two stride-2 convolutions and an 8-layer Transformer, d=256), train-only vocabulary, data, batching,
optimizer schedule (AdamW, peak LR 1e-3, 2,000 warmup steps, cosine decay), SpecAugment, seed 2105 and up to 60 epochs.
Only the output side differs, so the four neural families compare directly.

| Mode | Output side | Training loss | Final decoding |
|---|---|---|---|
| CTC encoder | linear CTC head | CTC | greedy CTC |
| Attention encoder-decoder | 4-layer Transformer decoder with cross-attention | cross-entropy, label smoothing 0.1 | beam 5, length-normalized |
| Hybrid CTC/attention | both | 0.3 × CTC + 0.7 × cross-entropy | beam 5, n-best rescored with 0.3 × CTC + 0.7 × attention |

Greedy dev WER every 2 epochs selects each checkpoint; test was decoded once per model.
Code: [`training/train_ctc_attention.py`](training/train_ctc_attention.py) (code commit `4912e53b396d82f17d9f71243d57a28fec353014`),
notebook: [`notebooks/ctc_attention_training.ipynb`](notebooks/ctc_attention_training.ipynb), tests:
[`tests/test_ctc_attention.py`](tests/test_ctc_attention.py) (teacher forcing, CTC greedy collapse, beam search on a scripted
decoder, CTC n-best rescoring, and a one-clip learning check per mode).
Weights: [`GokseninMuhammet/ctc-attention-icelandic-samromur-scratch`](https://huggingface.co/GokseninMuhammet/ctc-attention-icelandic-samromur-scratch),
revision `61ec3fd06db751fae48a56a94aef370bd63744b8` (CC BY 4.0). SHA-256: `ctc_scratch.pt`
`e863b46fc83ebde66b11abb56c1b7f4ba8e67bb80a175bbc558fe31dd8d7c8a8`, `attention_scratch.pt`
`6593506d55d67f99714ac9d27f1c0d33487e23b42f6a0bfe8f5b213a0ce14b47`, `hybrid_ctc_attention.pt`
`987794ae2efc6748c229d488e3fd715fc7d70d600a0e62473d2aab66c4cbbcb5`.

#### All neural models trained by our group on the 15 h train split

| System | Pretraining | Params | Best epoch | Training | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid CTC/attention** | none | 12.40 M | 60 / 60 | 79 min | **52.51 %** | **16.37 %** | **55.57 %** | **17.77 %** |
| mHuBERT-147 + CTC | 90k h, 147 languages | 94.4 M | 29 / 30 | 5.8 h (2 T4) | 66.30 % | 22.55 % | 68.46 % | 23.31 % |
| RNN-T | none | 9.19 M | 56 / 60 | 61 min | 65.50 % | 28.61 % | 68.95 % | 30.54 % |
| CTC encoder | none | 8.17 M | 54 / 60 | 54 min | 70.81 % | 21.26 % | 74.50 % | 22.88 % |
| Attention encoder-decoder | none | 12.39 M | 16 (stopped) | 23 min | 173.36 %* | n/a | 153.82 % | 92.79 % |

\* Greedy dev WER during training; the plateau guard stopped the run before a final dev decoding.
Run records: [CTC](results/ctc_scratch/model_info/ctc_scratch_test.json) ·
[attention](results/attention_scratch/model_info/attention_scratch_test.json) ·
[hybrid](results/hybrid_ctc_attention/model_info/hybrid_ctc_attention_test.json); training curves are next to each record.

**What this shows.** In this experiment, the attention-only configuration did not learn a stable audio-text alignment
from 15 h: it repeated and inserted words (WER above 100 %), and the plateau guard stopped it at epoch 16 (dev WER still
≥ 98 %). This is a limitation of this experiment and does not imply that attention models generally cannot learn
alignment. Adding the CTC branch improved this setup: the hybrid model passed the guard at epoch 16 (dev WER 92 %) and
kept improving to epoch 60. Hybrid CTC/attention achieved the lowest test WER among our models trained from scratch
(55.57%). Our recovered fine-tuned Whisper checkpoint achieved a lower test WER overall (36.06%). This result is
consistent with the motivation for hybrid CTC/attention (Watanabe et al., IEEE JSTSP 2017): CTC's monotonic alignment
regularizes the attention decoder. Beam search with CTC
rescoring improved the hybrid's dev WER from 53.67 % (greedy) to 52.51 %. The CTC encoder has a lower CER than the RNN-T
but a higher WER: it spells characters well but splits and merges words more.

**Memorization checks.** CTC copied its 32 clips exactly. For attention and hybrid the check was first written as "loss
below 10 % of its start", which label smoothing prevents; it now reports the transcript and requires a 70 % loss drop.
Hybrid cut its loss from 5.45 to 0.52 but did not copy the transcript exactly at step 600; we continued because the
loss fell and the plateau guard protects the run. The notebook's section 3 records these steps.

### Why WER and CER

We report word error rate as the main metric because it is the standard ASR metric: it counts substitutions,
deletions and insertions at the word level and divides by the number of reference words, so lower is better and
a misspelled word counts as a full error. Icelandic is highly inflected, so one wrong ending turns a word into an
error even when the sound was recognized almost correctly; we therefore also report character error rate, which
uses the same edit operations at the character level and gives partial credit for partly correct words.
Both are computed on normalized text (lowercase, no punctuation), as the Hugging Face audio course recommends,
so that systems are not penalized for casing or punctuation conventions (Whisper outputs them, CTC models do not).
Reference: [Hugging Face audio course, chapter 5, "Evaluation"](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

The scoring references contain no digits. A numeric hypothesis can disagree with a reference that spells out the number under the shared normalizer. The prediction files retain raw outputs for error analysis.

### License of model weights

The project's code is under the repository's MIT license. The fine-tuned HuBERT weights are a derivative of
`utter-project/mHuBERT-147` and are released under its license, CC BY-NC-SA 4.0. The RNN-T, CTC, attention and hybrid weights were
trained from scratch on Samrómur 21.05 and are released under CC BY 4.0.

## Remaining work

- Icelandic transcript verification and listening beyond the 60 focused clips remain pending. Clip `0049034` is the one uncertain audio-quality decision in the [completed focused review](#focused-audio-quality-review).
- Every course baseline family that supports Icelandic now has committed results: GMM-HMM, CTC encoder, attention encoder-decoder, hybrid CTC/attention, RNN-T, wav2vec 2.0 + CTC, HuBERT + CTC and multilingual Whisper (checked 22 November 2026). The [baseline checklist](BASELINES.md) records the course requirements and open applicability decisions. The professor's minimum training duration remains unconfirmed.
- Saleh completed full standardized dev and test evaluations of the recovered Whisper checkpoint in [PR #3](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/3), which is still open and has not been merged into `main`.

Each teammate documents and commits their own contribution from their own GitHub account.

## Source and attribution

The repository's original code and documentation use [MIT](LICENSE). The Samrómur audio, prompts, metadata, and derived scoring references use CC BY 4.0.

Samrómur Icelandic Speech 21.05 by David Erik Mollberg et al. is distributed under CC BY 4.0. This repository selects IDs and derives normalized scoring references from that release. Cite Mollberg et al., “Samrómur: Crowd-sourcing Data Collection for Icelandic Speech Recognition,” LREC 2020. [CLARIN release](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/189) · [OpenSLR 112](https://openslr.org/112/) · [LREC paper](https://aclanthology.org/2020.lrec-1.425/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


## (Saleh Almasaadi) — Evaluation and Statistical Analysis

I evaluated ten ASR models on the same 1,884 frozen test clips from 1,026 speaker/session groups. I verified clip IDs, reference transcripts, and speaker/session assignments and independently recalculated word error rate (WER) and character error rate (CER) using the shared `prepare.normalize_text` function.

### Test Results

| Model | Test WER | Test CER |
| --- | ---: | ---: |
| Icelandic wav2vec 2.0* | 7.76% | 1.86% |
| Published Icelandic Whisper Small* | 21.75% | 5.60% |
| Fine-tuned Whisper Small (checkpoint 1000) | 36.06% | 10.16% |
| Hybrid CTC/attention (trained from scratch) | 55.57% | 17.77% |
| HuBERT + CTC | 68.46% | 23.31% |
| RNN-T (trained from scratch) | 68.95% | 30.54% |
| CTC encoder (trained from scratch) | 74.50% | 22.88% |
| Original Whisper Small | 86.82% | 36.35% |
| MFCC + GMM-HMM | 101.47% | 87.74% |
| Attention-only (trained from scratch) | 153.82% | 92.79% |

\* The published Icelandic checkpoints may have encountered some Samrómur test clips during external training. Their scores should therefore be interpreted cautiously.

WER can exceed 100% when substitutions, deletions, and insertions collectively outnumber the reference words.

The hybrid CTC/attention model achieved the lowest WER among the models trained entirely from scratch by our group. Its 55.57% WER was also lower than the HuBERT + CTC baseline's 68.46% WER. The attention-only model struggled to learn an alignment and performed poorly, illustrating the benefit of combining CTC and attention.

### Statistical Analysis

I calculated 95% confidence intervals using 10,000 bootstrap resamples of whole speaker/session groups (random seed 2105).

I performed 90 paired WER/CER comparisons using 10,000 speaker/session-level randomization tests (random seed 2106), with plus-one Monte Carlo p-values and Holm correction across all 90 comparisons.

For example:

- Hybrid CTC/attention achieved a WER 12.89 percentage points lower than HuBERT + CTC (95% CI for HuBERT minus hybrid: 11.63–14.15 percentage points; Holm-adjusted p approximately 0.00900).
- HuBERT + CTC and RNN-T differed by only 0.49 percentage points in WER. Their paired WER difference was not statistically significant (95% CI: -1.94 to +0.93 percentage points; Holm-adjusted p = 0.55114).
- HuBERT + CTC had a CER 7.24 percentage points lower than RNN-T, with Holm-adjusted p approximately 0.00900.
- HuBERT + CTC and the scratch-trained CTC encoder showed no statistically significant CER difference after Holm correction (adjusted p = 0.55114).

The [evaluation results](results/person4_evaluation/) contain all model scores, confidence intervals, paired comparisons, error counts, representative transcription errors, methods, and the comparison chart.

The [evaluation notebook](notebooks/Person4_Icelandic_ASR_Evaluation.ipynb) reproduces the ten-model analysis on a CPU.

### Fine-Tuned Whisper Small Evaluation

I evaluated Person 3's recovered Whisper Small checkpoint 1000 on the development and frozen test sets using a Tesla T4 GPU.

- Development set (1,314 clips): 34.51% WER and 9.69% CER.
- Test set (1,884 clips): 36.06% WER and 10.16% CER.

The original Whisper Small baseline achieved 86.82% WER and 36.35% CER.

The observed WER reduction was 50.76 percentage points (95% CI: 47.31–54.85). CER decreased by 26.19 percentage points (95% CI: 23.48–29.17).

The paired randomization p-values were approximately 0.00010 before correction. In the final ten-model analysis, the Holm-adjusted p-values for these comparisons were approximately 0.00900.

Test-set inference took 892.32 seconds on a Tesla T4 GPU.

The original Whisper baseline used five-beam decoding, while the recovered fine-tuned checkpoint used greedy decoding. Consequently, the observed comparison reflects both model fine-tuning and different decoding configurations.

The full development and test prediction CSVs and run metadata are committed under `results/person4_evaluation/person3_whisper/`. These evaluations are complete; no additional test decoding is needed.

The evaluation script automatically generates `results/person4_evaluation/whisper_finetuning_comparison.csv` when the fine-tuned Whisper predictions are supplied.

### Reproducibility and Input Verification

The frozen reference data, normalization implementation, and four original baseline prediction files are retrieved from Git commit:

`e9cd03331ef7f2295a5baf197c8c74f1ce41417d`

The evaluation notebook verifies the recovered Whisper test prediction input against this fixed SHA-256 checksum:

`506f0b277372ebdbba9e08c990e1b691773de6539a2a803b83544d57ac20bf3b`

The notebook rejects a modified or mismatched Whisper prediction file, rather than silently accepting predictions from a moving branch.

Evaluation dependencies are pinned in `evaluation/requirements-person4.txt`.

### Additional Model Comparisons

I evaluated Person 5's MFCC + GMM-HMM predictions and independently obtained 101.47% WER and 87.74% CER. Subsequent training-only diagnostics found that this model struggled to memorize a small set of training recordings. One transcript-constrained realignment attempt worsened the development sample, so the original model and test results were retained without test-based tuning.

I also independently evaluated Person 2's additional predictions:

- RNN-T: 68.95% WER, 30.54% CER.
- Scratch CTC encoder: 74.50% WER, 22.88% CER.
- Scratch attention-only model: 153.82% WER, 92.79% CER.
- Hybrid CTC/attention: 55.57% WER, 17.77% CER.

All ten models were checked against the same frozen 1,884 test clips and 1,026 speaker/session groups before comparison.

### Transcription Error Analysis

I analyzed substitutions, deletions, insertions, and exact word-boundary errors across all ten models. The final evaluation generated ten model-level error summaries and 38 representative transcription error examples.

I also examined digit-versus-word errors in the original Whisper Small predictions. There were 39 test predictions containing digits. The notebook preserves representative examples and generates `original_whisper_digit_examples.csv`.

The official frozen normalization and scoring rules were not changed for this additional analysis.

### Reproducing the Evaluation

The standalone script `evaluation/person4_evaluate.py` reproduces the ten-model analysis from saved predictions without retraining or GPU inference.

Install dependencies from the repository root:

```bash
pip install -r evaluation/requirements-person4.txt
```

Run the complete evaluation:

```bash
python evaluation/person4_evaluate.py \
  --person3-csv results/person4_evaluation/person3_whisper/test_predictions.csv \
  --gmm-csv results/mfcc_gmm_hmm/predictions/mfcc_gmm_hmm_test.csv \
  --rnnt-csv results/rnnt/predictions/rnnt_test.csv \
  --ctc-csv results/ctc_scratch/predictions/ctc_scratch_test.csv \
  --attention-csv results/attention_scratch/predictions/attention_scratch_test.csv \
  --hybrid-csv results/hybrid_ctc_attention/predictions/hybrid_ctc_attention_test.csv \
  --output-dir results/person4_evaluation_reproduced
```

The RNN-T, scratch CTC, attention-only, hybrid CTC/attention, GMM-HMM, and recovered Whisper prediction files are included in the shared repository.

The script validates all 1,884 test clips and 1,026 speaker/session groups, recalculates WER/CER, computes 95% confidence intervals, performs 90 paired statistical comparisons with Holm correction, and generates the transcription error analysis and comparison chart.

The script exports:

- `model_scores_and_confidence_intervals.csv` (20 rows)
- `paired_comparisons.csv` (90 rows)
- `error_counts.csv` (10 rows)
- `error_examples.csv` (38 rows)
- `whisper_finetuning_comparison.csv` (2 rows)
- `methods.json`
- `test_error_rates.png`

The notebook additionally produces the original Whisper digit-error examples.

---

## Whisper Small fine-tuning

Person 3 fine-tuned OpenAI's multilingual Whisper Small model (~242M parameters)
on the frozen Samrómur v1 train split. The [training notebook](NLP_Project3_Person3_Whisper_Clean.ipynb)
contains saved training output. The [historical training summary](training_metrics.json)
records the configuration and original validation scores from that output.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | openai/whisper-small |
| Training samples | 10,488 |
| Validation samples | 1,314 |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Effective batch size | 16 |
| Precision | FP16 |
| GPU | NVIDIA Tesla T4 |
| Training duration | 5 hours 12 minutes |
| Optimizer steps | 1,968 |

### Historical validation during training

| Checkpoint | WER | CER |
|---|---|---|
| Step 1000 | 34.51% | 9.70% |
| Step 1600 | **30.62%** | 8.41% |
| Step 1968 | 30.70% | **8.40%** |

Steps 1600 and 1968 had the lowest historical dev WER and CER respectively. Only step 1000 was recovered and exported. The table uses the original notebook's scoring, and the historical JSON's evaluation status describes the time of that training snapshot.

### Recovered checkpoint evaluation

Both development and test evaluations below use the recovered step-1000 checkpoint, frozen v1 references, and the shared `prepare.normalize_text` scoring policy.

| Evaluation | Clips | WER | CER | Evidence |
|---|---:|---:|---:|---|
| Dev smoke test (CPU) | 5 | 40.54% | 12.36% | [Run record](results/person3_whisper_smoke/dev_run.json) · [Predictions](results/person3_whisper_smoke/dev_predictions.csv) |
| Full development set (T4, beam 1) | 1,314 | 34.51% | 9.69% | [Run record](results/person4_evaluation/person3_whisper/dev_run.json) · [Predictions](results/person4_evaluation/person3_whisper/dev_predictions.csv) |
| Full held-out test (T4, beam 1) | 1,884 | 36.06% | 10.16% | [Run record](results/person4_evaluation/person3_whisper/test_run.json) · [Predictions](results/person4_evaluation/person3_whisper/test_predictions.csv) |

The complete standardized development and test evaluations were performed using Person 3's recovered Whisper Small checkpoint 1000. Both prediction CSVs and their run metadata are included in [PR #3](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/3), together with confidence intervals and paired statistical comparisons.

The original training-time validation results are preserved separately above because they were produced during checkpoint selection. The standardized development evaluation uses the team's shared scoring policy.

[PR #4](https://github.com/RanimAlyaseen/icelandic-asr-project3/pull/4) documents the separate checkpoint download and verification workflow.

### Recovered model download and verification

**Model:** Fine-tuned OpenAI Whisper Small  
**Available checkpoint:** Step 1000  
**Original validation WER:** 34.51%  
**Original validation CER:** 9.70%  

**Download the exported model:**

https://drive.google.com/drive/folders/1BKWu0PBXdNlp7pqeX7MCn9lhzFsay_Md?usp=sharing

The complete model folder contains the model weights, configuration,
processor, tokenizer, and recovery metadata.

#### SHA-256 Checksums

| File | SHA-256 |
|---|---|
| config.json | 572ff2d955ca237231a371e756e41b4a4c0c63df3c30e64538b90294f468e449 |
| generation_config.json | 2d3e60d85c13324649773e42932b2a9916edd3ff0f24434b17db657c61c57fd0 |
| model.safetensors | d056c430a0b4c1d9006a9ddf68a0b8d92f1cf4828ccb4f5bd96f77d5f911f80b |
| processor_config.json | 8c648fa85aafc59a07435ed751322499bd4238b819ca9200ddd2b5ba4df50e68 |
| recovery_metadata.json | 21e255d551be5020ca1c0304b8b25b6bf88c018928bc6768b0b5cd341bff7f09 |
| tokenizer.json | 783ab7faf82595b8abc49cef57865b055d763753e6383d0d98774abcfe16e57a |
| tokenizer_config.json | 0940f4bfe73310f1fdc9f86242568178700211fde2d9b49b15dd1ad892552641 |

These checksums were calculated from the exported files stored
in Google Drive.

Use the downloaded checkpoint with the [recovered Whisper evaluation script](training/evaluate_whisper_step1000.py) and [Whisper dependencies](training/requirements-person3.txt). The run records above document the evaluated checkpoint and decoding settings.

### Reproduce Person 3's Whisper Small Training

Person 3's standalone training entry point is
[`training/train_whisper.py`](training/train_whisper.py).
It fine-tunes multilingual Whisper Small using the frozen
Samrómur v1 training and validation manifests.

**Reproducibility limitation:** The exact base-model revision
and dataset checkout used for the original training were not
recorded. The revisions below are fixed for future runs;
they are not verified historical revisions.

| Component | Fixed revision or selection |
|---|---|
| Base model | `openai/whisper-small` |
| Base-model revision | `973afd24965f72e36ca33b3055d56a652f456b4d` |
| Dataset selection | `samromur-21.05-20h-v1` |
| Preparation commit | `55f5a6856e92cdf89608c492cc470aea0d019e1b` |
| Audio-enabled commit | `96e5839b27d9871d6b2408b3c3719dc809a730b7` |

The preparation commit predates the Git LFS audio pointers.
The audio-enabled commit contains those pointers, while its
three prepared manifests are byte-identical to the frozen
preparation revision.

**1. Prepare a separate dataset checkout**

Run these commands in a terminal, outside the development
repository. Git LFS must be installed.

~~~bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RanimAlyaseen/icelandic-asr-project3.git icelandic-asr-dataset
cd icelandic-asr-dataset
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 96e5839b27d9871d6b2408b3c3719dc809a730b7
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""
cd ..
~~~

The Git LFS command retrieves the prepared cleaned audio,
not the complete raw subset.

**2. Install dependencies**

Run this command from the development repository:

~~~bash
python -m pip install -r training/requirements-person3.txt
~~~

GPU training requires a compatible CUDA-enabled PyTorch
installation and sufficient GPU memory.

**3. Validate the dataset without training**

~~~bash
python training/train_whisper.py --repo-dir ../icelandic-asr-dataset --check-only
~~~

The preflight checks the frozen manifest SHA-256 hashes
and verifies that manifest-referenced audio files exist
and contain FLAC data rather than unresolved LFS pointers.

| Manifest | Expected SHA-256 |
|---|---|
| `output/train.jsonl` | `d36a1adccdb10682b30a6fd8b2f7e20f8c86babb4b1d20a007af79179c4436ed` |
| `output/dev.jsonl` | `f0e7675cd31497788250728bba966d1d884ad5f929b3488724b2fef2110f9bf4` |
| `output/test.jsonl` | `f5e346f545a5d7f4298f1838bc63b978a4d7ab9d063ddc97897fa065223d0481` |

**4. Run training when GPU resources are available**

~~~bash
python training/train_whisper.py --repo-dir ../icelandic-asr-dataset --output-dir ./whisper_training_output
~~~

The configuration uses three epochs, learning rate `1e-5`,
FP16, gradient checkpointing, and validation WER for
checkpoint selection.

**Validation status:** The standalone script passed Python
syntax and CLI checks. Its missing-audio failure path was
also tested. A complete training run using this script
has not been performed or independently verified.

For evaluation without retraining, use the recovered
step-1000 checkpoint procedure below.

---

### Reproduce Person 3's Five-Sample Evaluation

The recovered Whisper Small checkpoint can be downloaded and
evaluated without retraining or accessing a personal Google Drive.

The original training did not record the exact base-model revision
or dataset checkout commit.

**1. Install dependencies**

```bash
python -m pip install -r training/requirements-person3.txt gdown
```

**2. Download and verify the checkpoint**

```bash
python training/download_whisper_checkpoint.py
```

The downloader verifies SHA-256 checksums for seven checkpoint files.
The model is saved to `models/recovered_model_step1000/`.

**3. Download development audio**

```bash
git lfs install
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/dev/**" --exclude=""
```

**4. Run the five-sample evaluation**

```bash
python training/evaluate_whisper_step1000.py \
  --model models/recovered_model_step1000 \
  --audio-root data/cleaned/samromur-21.05-20h-v1 \
  --split dev \
  --max-samples 5 \
  --output-dir results/person3_whisper_repro_test
```

The verified five-sample CPU results were **40.54% WER** and
**12.36% CER**. These are smoke-test results, not full-split metrics.

No retraining is required.
