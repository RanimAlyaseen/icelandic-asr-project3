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

The preparation environment was verified on Windows with Python 3.14.10. Run from the repository root.

```powershell
python -m venv .venv
$venvPy = '.\.venv\Scripts\python.exe'
& $venvPy -m pip install --only-binary=:all: -r requirements-windows-py314.lock.txt
& $venvPy -m pip check
```

On macOS or Linux with Python 3.14, use `python3.14 -m venv .venv`, `.venv/bin/python`, and [requirements.txt](requirements.txt). See the [Windows dependency lock](requirements-windows-py314.lock.txt).

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
| [requirements-modeling.txt](requirements-modeling.txt) | Pinned modeling environment (PyTorch, Transformers, Accelerate) used for every baseline and HuBERT run |
| [tests/](tests/) | Independent tests with known inputs and expected outputs |
| [output/](output/) | Frozen manifests and references, vocabulary, audit, review queues, and verification records |
| [data/](data/) | Raw subset and original documents under **raw/samromur_21.05/** and cleaned v1 FLACs under **cleaned/samromur-21.05-20h-v1/** |
| [review/](review/) | Manual listening decisions kept outside the generated outputs |
| [contributions/](contributions/) | Individual work records authored and committed by team members |
| [baselines/](baselines/) | Inference code for published pretrained checkpoints (baselines, no training by the group) |
| [training/](training/) | Our own training code: HuBERT + CTC fine-tuning and its diagnostic |
| [notebooks/](notebooks/) | Kaggle/Colab notebooks that run the committed scripts end to end from this repository |
| [results/](results/) | Per-clip predictions (CSV), run records (JSON: checkpoint, revision, settings, GPU) and training logs |

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

## Human listening remains pending

Record decisions for the **53 near-silence exclusions and seven retained possible-clipping cases** in the [60-recording manual review log](review/focused_manual_review.tsv). The [generated measurements](output/focused_review.tsv) remain available as evidence. Preprocessing writes the generated outputs separately. Keep leading zeros when importing IDs.

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

The [541-clip evaluation checklist](output/flagged_dev_test_review.tsv) covers 315 test and 226 dev recordings. Its paths start with **audio/** so they resolve from the handoff root. For the original release root, drop that prefix. The [full listening queue](output/listening_queue.tsv) also covers flagged train clips and unflagged controls.

The [focused review queue](output/focused_review.tsv) documents **53 near-silence exclusions and seven selected possible-clipping flags**. All 53 met the RMS threshold, and 15 also met the quiet-fraction threshold. The seven selected clips have 0.1095% to 0.3238% full-scale samples. Quiet speech may still be intelligible and full-scale samples may be inaudible. Use the release paths to listen and record proposed changes by ID.

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

After cloning and installing dependencies as shown above, run these commands from the repository root.

~~~powershell
& $venvPy -X utf8 -m unittest discover -s tests -v
& $venvPy -X utf8 verify.py --replay-only --root data/raw/samromur_21.05 --output output
~~~

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

## Pretrained baselines and HuBERT + CTC (Muhammet Çelik)

All systems below are scored on the same frozen v1 test clips (1,884 utterances, 3.0 h) against the same
references (`output/evaluation_references.tsv`), with the same text normalization (`prepare.normalize_text`:
lowercase, Unicode NFC, punctuation and symbols removed). Settings were chosen on the dev split; test was decoded once.
Per-clip predictions and run records (checkpoint, revision, decoding settings, GPU, package versions) are in
[`results/pretrained/`](results/pretrained/) and [`results/hubert_ctc/`](results/hubert_ctc/).
Hardware: Kaggle, one NVIDIA Tesla T4 (16 GB), fp16.

Every Hugging Face checkpoint is pinned to a full commit hash in the code (`REVISIONS` in
[`baselines/run_pretrained.py`](baselines/run_pretrained.py), `Config.base_revision` in
[`training/train_hubert_ctc.py`](training/train_hubert_ctc.py)), including the `openai/whisper-small`
generation config used as a fallback for `Valdimarb13/whisper-small-icelandic`. A script refuses to run with an
unpinned checkpoint, and [`tests/test_pretrained.py`](tests/test_pretrained.py) checks the pins.

#### Setup for the modeling code

The modeling code uses its own pinned environment, [`requirements-modeling.txt`](requirements-modeling.txt):
the exact versions recorded for every reported run (Python 3.13.16, PyTorch 2.11.0 with CUDA 13.0,
Transformers 5.23.0, Accelerate 1.15.0). Keep it separate from the Windows data-pipeline environment above.

On Kaggle or Colab (GPU T4, internet on), open a notebook from [`notebooks/`](notebooks/) and run it top to bottom.
Its first cell checks out a fixed data commit of this repository (`DATA_COMMIT`) and a fixed code commit
(`CODE_COMMIT`), pulls only the needed audio through Git LFS, and installs `requirements-modeling.txt`.
No manual uploads are needed.

On your own Linux machine with an NVIDIA GPU:

```bash
python3.13 -m venv .venv-modeling
.venv-modeling/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.venv-modeling/bin/python -m pip install -r requirements-modeling.txt
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""
.venv-modeling/bin/python -m unittest discover -s tests -p "test_pretrained.py"
.venv-modeling/bin/python baselines/run_pretrained.py --model wav2vec2_is --split dev
```

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
Samrómur 21.05, so these models may have heard some of them during training. Their scores are an optimistic
upper bound and should not be compared one-to-one with models trained only on our 15 h train split.

### Course baselines that do not support Icelandic (not run)

As the course instructions require, these were not run with a wrong language setting:

| Model | Model card | Icelandic |
|---|---|---|
| `nvidia/canary-1b-v2` | https://huggingface.co/nvidia/canary-1b-v2 | not in the supported languages (25 European languages) |
| `nvidia/parakeet-tdt-0.6b-v3` | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | not in the supported languages (25 European languages) |
| `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | not in the supported languages |

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
the final-decoding numbers (66.30 % dev, 68.46 % test) because the same script produced the test result and every other
system's predictions in this table. Decoding each clip alone (batch size 1, no padding) would remove the effect; we did not run it.
Dev WER was still falling slowly at epoch 30, so longer training would probably help. CER is far below WER: most errors are
merged or split words and single-letter spelling mistakes that a language model would fix.

### Why WER and CER

We report word error rate as the main metric because it is the standard ASR metric: it counts substitutions,
deletions and insertions at the word level and divides by the number of reference words, so lower is better and
a misspelled word counts as a full error. Icelandic is highly inflected, so one wrong ending turns a word into an
error even when the sound was recognized almost correctly; we therefore also report character error rate, which
uses the same edit operations at the character level and gives partial credit for partly correct words.
Both are computed on normalized text (lowercase, no punctuation), as the Hugging Face audio course recommends,
so that systems are not penalized for casing or punctuation conventions (Whisper outputs them, CTC models do not).
Reference: [Hugging Face audio course, chapter 5, "Evaluation"](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

Known caveat: references spell numbers as words, while Whisper can output digits; this inflates Whisper's WER.
The raw outputs are kept in the `hypothesis_raw` column for the error analysis.

### License of model weights

The project's code is under the repository's MIT license. The fine-tuned HuBERT weights are a derivative of
`utter-project/mHuBERT-147` and are released under its license, CC BY-NC-SA 4.0.

## Remaining work

Team listening and Icelandic transcript-agreement review remain pending. The [confirmed baseline checklist](BASELINES.md) records the open applicability decisions for named checkpoints. The professor's minimum training duration remains unconfirmed. Each teammate documents and commits their own contribution from their own GitHub account.

## Source and attribution

The repository's original code and documentation use [MIT](LICENSE). The Samrómur audio, prompts, metadata, and derived scoring references use CC BY 4.0.

Samrómur Icelandic Speech 21.05 by David Erik Mollberg et al. is distributed under CC BY 4.0. This repository selects IDs and derives normalized scoring references from that release. Cite Mollberg et al., “Samrómur: Crowd-sourcing Data Collection for Icelandic Speech Recognition,” LREC 2020. [CLARIN release](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/189) · [OpenSLR 112](https://openslr.org/112/) · [LREC paper](https://aclanthology.org/2020.lrec-1.425/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## (Saleh Almasaadi) — Evaluation and Statistical Analysis

I evaluated four models on the same 1,884 frozen test clips from 1,026 speaker/session groups. I verified the clip IDs and references and reproduced the reported scores using `prepare.normalize_text`.

| Model | Test WER | Test CER |
| --- | ---: | ---: |
| Icelandic wav2vec 2.0* | 7.76% | 1.86% |
| Published Icelandic Whisper Small* | 21.75% | 5.60% |
| HuBERT + CTC | 68.46% | 23.31% |
| Original Whisper Small | 86.82% | 36.35% |

\* These published Icelandic checkpoints may have seen some Samrómur test clips during external training.

I calculated 95% confidence intervals using 10,000 bootstrap resamples of whole speaker/session groups. For paired tests, I used 10,000 session-level random swaps and applied Holm adjustment across the 12 baseline comparisons. The [results folder](results/person4_evaluation/) contains the intervals, p-values, error counts, examples, and chart. The [evaluation notebook](notebooks/Person4_Icelandic_ASR_Evaluation.ipynb) reproduces the analysis on a CPU.

The main comparison between the team's fine-tuned Whisper and the original Whisper Small is pending the official test predictions.
