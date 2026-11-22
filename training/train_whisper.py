#!/usr/bin/env python3
"""Reproducible Icelandic Whisper Small fine-tuning.

Based on NLP_Project3_Person3_Whisper_Clean.ipynb.

This script does not train when imported.
Use --check-only to validate paths and configuration.
"""

import argparse
import hashlib
import inspect
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-small"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
DATASET_COMMIT = "55f5a6856e92cdf89608c492cc470aea0d019e1b"

DATASET_SELECTION = "samromur-21.05-20h-v1"

EXPECTED_TRAIN = 10488
EXPECTED_VALIDATION = 1314
EXPECTED_TEST = 1884

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper Small on Icelandic Samromur data."
    )

    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/content/icelandic-asr-project3"),
        help="Dataset repository containing output/ and data/.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./whisper_training_output"),
        help="Directory for checkpoints and results.",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset paths without loading models or training.",
    )

    return parser.parse_args()


def get_repository_revision(repo_dir):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def validate_dataset(repo_dir):
    """Verify frozen manifests and prepared audio availability."""
    repo_dir = repo_dir.resolve()

    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset repository not found: {repo_dir}"
        )

    actual_commit = get_repository_revision(repo_dir)

    expected_hashes = {
        "train": "d36a1adccdb10682b30a6fd8b2f7e20f8c86babb4b1d20a007af79179c4436ed",
        "validation": "f0e7675cd31497788250728bba966d1d884ad5f929b3488724b2fef2110f9bf4",
        "test": "f5e346f545a5d7f4298f1838bc63b978a4d7ab9d063ddc97897fa065223d0481",
    }

    output_dir = repo_dir / "output"

    manifests = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "dev.jsonl",
        "test": output_dir / "test.jsonl",
    }

    for split, path in manifests.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {split} manifest: {path}"
            )

        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if actual_hash != expected_hashes[split]:
            raise RuntimeError(
                f"{split} manifest SHA-256 mismatch.\\n"
                f"Expected: {expected_hashes[split]}\\n"
                f"Actual:   {actual_hash}\\n"
                "The dataset differs from frozen v1."
            )

    audio_root = (
        repo_dir / "data" / "cleaned" / DATASET_SELECTION
    )

    if not audio_root.is_dir():
        raise FileNotFoundError(
            f"Prepared audio directory not found: {audio_root}\\n"
            "The frozen manifests are verified, but the audio "
            "is not available. Use the repository revision "
            "containing Git LFS audio pointers and download "
            "the cleaned recordings before training."
        )

    print("Dataset repository:", repo_dir)
    print("Dataset checkout:", actual_commit)
    print("Frozen preparation commit:", DATASET_COMMIT)
    print("Dataset selection:", DATASET_SELECTION)
    print("Manifest SHA-256 verification: PASS")

    # Verify all frozen-manifest audio files.
    missing = []
    pointers = []
    checked = 0

    for split, manifest_path in manifests.items():
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue

                record = json.loads(line)
                relative = Path(record["audio_path"])

                expected_split = (
                    "dev" if split == "validation" else split
                )

                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != expected_split
                ):
                    raise ValueError(
                        f"Invalid audio path at {split} "
                        f"line {line_number}: {relative}"
                    )

                audio_path = audio_root / relative
                checked += 1

                if not audio_path.is_file():
                    missing.append(str(relative))
                    continue

                with audio_path.open("rb") as audio_file:
                    header = audio_file.read(80)

                if header.startswith(
                    b"version https://git-lfs.github.com/spec"
                ):
                    pointers.append(str(relative))
                elif not header.startswith(b"fLaC"):
                    raise RuntimeError(
                        f"Invalid FLAC header: {audio_path}"
                    )

    if missing or pointers:
        raise RuntimeError(
            "Audio hydration incomplete.\n"
            f"Checked: {checked}\n"
            f"Missing: {len(missing)}\n"
            f"Git LFS pointers: {len(pointers)}\n"
            f"First missing: {missing[:3]}\n"
            f"First pointers: {pointers[:3]}\n"
            "Download the cleaned audio through Git LFS."
        )

    print(
        f"Audio availability verification: PASS "
        f"({checked} recordings)"
    )

    return manifests, audio_root





def main():
    args = parse_args()

    manifests, audio_root = validate_dataset(args.repo_dir)

    if args.check_only:
        print("Configuration and dataset paths: PASS")
        print("Training executed: NO")
        return

    import numpy as np
    import torch
    import soundfile as sf
    import evaluate

    from datasets import load_dataset
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this FP16 training configuration."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pinned Whisper Small model and processor...")

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        language="icelandic",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    model.generation_config.language = "icelandic"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False

    processor.tokenizer.clean_up_tokenization_spaces = False

    print("Loading prepared dataset manifests...")

    dataset = load_dataset(
        "json",
        data_files={
            split: str(path)
            for split, path in manifests.items()
        },
    )

    assert len(dataset["train"]) == EXPECTED_TRAIN
    assert len(dataset["validation"]) == EXPECTED_VALIDATION
    assert len(dataset["test"]) == EXPECTED_TEST

    def prepare_whisper_sample(example):
        audio_path = audio_root / example["audio_path"]

        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
        )

        if sample_rate != 16000:
            raise ValueError(
                f"Unexpected sample rate: {sample_rate}"
            )

        if audio.ndim != 1:
            raise ValueError(
                f"Expected mono audio: {audio_path}"
            )

        features = processor.feature_extractor(
            audio,
            sampling_rate=16000,
        ).input_features[0]

        labels = processor.tokenizer(
            example["text"]
        ).input_ids

        return {
            "input_features": features,
            "labels": labels,
        }

    print("Preparing training features...")

    train_dataset = dataset["train"].map(
        prepare_whisper_sample,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Preparing training audio",
    )

    print("Preparing validation features...")

    eval_dataset = dataset["validation"].map(
        prepare_whisper_sample,
        remove_columns=dataset["validation"].column_names,
        num_proc=1,
        desc="Preparing validation audio",
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features):
            input_features = [
                {"input_features": item["input_features"]}
                for item in features
            ]

            batch = self.processor.feature_extractor.pad(
                input_features,
                return_tensors="pt",
            )

            label_features = [
                {"input_ids": item["labels"]}
                for item in features
            ]

            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt",
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch["attention_mask"].ne(1),
                -100,
            )

            decoder_start_token_id = (
                self.processor.tokenizer.convert_tokens_to_ids(
                    "<|startoftranscript|>"
                )
            )

            if (
                labels.shape[1] > 0
                and (
                    labels[:, 0] == decoder_start_token_id
                ).all().item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = pred.label_ids.copy()

        label_ids[label_ids == -100] = (
            processor.tokenizer.pad_token_id
        )

        pred_str = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )

        label_str = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        pred_str = [
            " ".join(text.strip().split())
            for text in pred_str
        ]

        label_str = [
            " ".join(text.strip().split())
            for text in label_str
        ]

        return {
            "wer": wer_metric.compute(
                predictions=pred_str,
                references=label_str,
            ),
            "cer": cer_metric.compute(
                predictions=pred_str,
                references=label_str,
            ),
        }

    # Preserve the final notebook configuration.
    batch_size = 2
    gradient_accumulation = 8
    epochs = 3

    batches_per_epoch = math.ceil(
        len(train_dataset) / batch_size
    )

    steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation
    )

    total_steps = math.ceil(
        steps_per_epoch * epochs
    )

    warmup_steps = math.ceil(total_steps * 0.05)

    supported_args = inspect.signature(
        Seq2SeqTrainingArguments
    ).parameters

    if "eval_strategy" in supported_args:
        eval_key = "eval_strategy"
    elif "evaluation_strategy" in supported_args:
        eval_key = "evaluation_strategy"
    else:
        raise RuntimeError(
            "Unsupported Transformers evaluation strategy API."
        )

    training_kwargs = {
        "output_dir": str(checkpoint_dir),
        "learning_rate": 1e-5,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": gradient_accumulation,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "linear",
        "fp16": True,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_strategy": "steps",
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": 225,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "seed": SEED,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }

    training_kwargs[eval_key] = "steps"

    unsupported = [
        key
        for key in training_kwargs
        if key not in supported_args
    ]

    if unsupported:
        raise RuntimeError(
            f"Unsupported training arguments: {unsupported}"
        )

    training_args = Seq2SeqTrainingArguments(
        **training_kwargs
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    last_checkpoint = get_last_checkpoint(
        str(checkpoint_dir)
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(eval_dataset))
    print("Warmup steps:", warmup_steps)
    print("Checkpoint interval:", 200)
    print("Resume checkpoint:", last_checkpoint)

    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_selection": DATASET_SELECTION,
        "dataset_commit": DATASET_COMMIT,
        "seed": SEED,
        "epochs": epochs,
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "warmup_steps": warmup_steps,
        "save_steps": 200,
        "eval_steps": 200,
    }

    (results_dir / "training_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    start_time = time.time()

    train_result = trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

    duration = time.time() - start_time

    results = {
        **config,
        "gpu": torch.cuda.get_device_name(0),
        "training_samples": len(train_dataset),
        "validation_samples": len(eval_dataset),
        "training_duration_seconds": duration,
        "training_loss": train_result.training_loss,
        "global_step": trainer.state.global_step,
    }

    (results_dir / "training_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("Training completed.")
    print("Results:", results_dir)


if __name__ == "__main__":
    main()
