Individual Contribution — Person 3

Name: Tasnim Alibraheem
Student ID: 2309015259
GitHub: tasneemzouhair-coder
Project: Icelandic Automatic Speech Recognition
Role: Whisper Small Fine-Tuning

Responsibilities

My responsibility was to fine-tune the multilingual Whisper Small model for Icelandic automatic speech recognition using the team’s cleaned Samrómur 21.05 dataset.

Work Completed

* Implemented the Whisper Small fine-tuning pipeline using Hugging Face Transformers and PyTorch.
* Prepared the model for Icelandic transcription.
* Used the team’s training and validation splits.
* Configured training with FP16 and gradient accumulation to accommodate GPU memory limitations.
* Completed three training epochs on an NVIDIA Tesla T4 GPU.
* Recorded validation WER and CER throughout training.
* Recovered and exported checkpoint 1000 after the best checkpoint became unavailable.
* Documented the training configuration, results, and checkpoint limitations.
* Uploaded the cleaned notebook and training metrics to GitHub.

Training Configuration

Parameter	Value
Model	openai/whisper-small
Training samples	10,488
Validation samples	1,314
Epochs	3
Learning rate	1e-5
Batch size	2
Gradient accumulation	8
Effective batch size	16
GPU	NVIDIA Tesla T4
Training duration	5 hours 12 minutes
Optimizer steps	1,968

Results

The lowest recorded validation WER was 30.62% at step 1600, with CER of 8.41%.

The lowest recorded validation CER was 8.40% at step 1968, with WER of 30.70%.

These results came from different checkpoints.

The recovered checkpoint 1000 originally achieved:

* WER: 34.51%
* CER: 9.70%

These measurements have not yet been recomputed using the team’s shared prepare.normalize_text scoring policy.

Deliverables

* Cleaned Whisper Small fine-tuning notebook
* Recorded training metrics
* Exported checkpoint 1000, stored separately on Google Drive
* Training documentation

Remaining Work

The recovered model requires standardized evaluation on the development set using the team’s shared normalization policy.

Once decoding settings are finalized, another teammate can evaluate it on the held-out test set and record predictions, WER, CER, inference speed, and GPU information.

No final held-out test results are claimed in this contribution.
