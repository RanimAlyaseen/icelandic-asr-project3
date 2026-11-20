"""Inference-only evaluation of recovered Whisper Small checkpoint 1000.
Run from the Icelandic ASR repository root. No training is performed.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast
from prepare import normalize_text, read_jsonl
from preprocess import resolve_audio


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, type=Path, help='Downloaded recovered_model_step1000 folder')
    p.add_argument('--audio-root', default='data/cleaned/samromur-21.05-20h-v1', type=Path)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--output-dir', default='results/person3_whisper_step1000', type=Path)
    p.add_argument('--max-samples', type=int, default=None, help='Smoke test only; omit for full scoring')
    args = p.parse_args()
    if not (args.model / 'model.safetensors').is_file():
        p.error('model.safetensors not found; download the complete model folder')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path('output') / f'{args.split}.jsonl'
    records = list(read_jsonl(manifest))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True).to(device).eval()
    tokenizer = WhisperTokenizerFast.from_pretrained(str(args.model), language='icelandic', task='transcribe', local_files_only=True)
    # The recovered export has processor_config.json but no preprocessor_config.json.
    # Whisper Small uses the standard 80-mel feature extractor settings.
    extractor = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000) 
    generation = {
        'language': 'is',
        'task': 'transcribe',
        'max_new_tokens': 225,
        'num_beams': 1,
        'do_sample': False
    }
    rows, refs, hyps = [], [], []
    start = time.perf_counter()
    for i, record in enumerate(records, 1):
        audio_path = resolve_audio(args.audio_root, record)
        waveform, rate = sf.read(audio_path, dtype='float32')
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError(f'Expected mono 16 kHz: {audio_path}')
        features = extractor(waveform, sampling_rate=16000, return_tensors='pt').input_features.to(device)
        with torch.inference_mode():
            tokens = model.generate(features, **generation)
        prediction = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        reference = normalize_text(record['text'])
        normalized_prediction = normalize_text(prediction)
        refs.append(reference)
        hyps.append(normalized_prediction)
        rows.append({'split': args.split, 'id': record['id'], 'reference': reference,
                     'prediction': prediction, 'prediction_normalized': normalized_prediction})
        if i % 100 == 0:
            print(f'{i}/{len(records)} utterances processed', flush=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    predictions = args.output_dir / f'{args.split}_predictions.csv'
    with predictions.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'id', 'reference', 'prediction', 'prediction_normalized'])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'model': 'recovered Whisper Small step 1000', 'split': args.split, 'samples': len(rows),
        'complete_split': args.max_samples is None, 'wer': jiwer.wer(refs, hyps), 'cer': jiwer.cer(refs, hyps),
        'normalization': 'prepare.normalize_text applied to references and predictions',
        'decoding': generation, 'seconds_total': elapsed,
        'seconds_per_utterance': elapsed / len(rows) if rows else None,
        'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'python': platform.python_version(), 'torch': torch.__version__,
        'model_weights_sha256': sha256(args.model / 'model.safetensors'),
        'manifest_sha256': sha256(manifest),
    }
    (args.output_dir / f'{args.split}_run.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
