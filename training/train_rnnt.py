"""Train an RNN-Transducer (RNN-T) for Icelandic from scratch on the frozen v1 train split.

Course baseline family "Transducer (RNN-T)". No published Icelandic transducer exists
(nvidia/parakeet-tdt-0.6b-v3 does not support Icelandic), so we implement and train a small one:

  80-dim log-mel features (25 ms window, 10 ms hop, per-utterance mean/variance normalization)
  -> 2 x Conv2d subsampling (4x, 40 ms frames) -> Transformer encoder (pre-norm, sinusoidal positions)
  -> joiner(encoder frame, prediction network state) -> characters + blank
  prediction network: character embedding + 1-layer LSTM, started from the blank symbol.

Loss: torchaudio.functional.rnnt_loss (CUDA). Vocabulary: output/ctc_vocab.json (train-only characters,
'[PAD]' (id 0) is the blank, '|' is the word boundary). Decoding: batched greedy search.
Dev WER (every `eval_every` epochs) selects the checkpoint; test is decoded once with --evaluate-test.

Outputs (results/rnnt/): predictions/rnnt_{dev,test}.csv, model_info/rnnt_{dev,test}.json,
training_log.json, training_curves.png, training_summary.json. Weights: models/rnnt/best.pt (git-ignored).

Usage (repository root, one GPU):
  python training/train_rnnt.py --overfit 32            # sanity check: must memorize 32 clips
  python training/train_rnnt.py --train --limit 256 --epochs 2   # smoke test
  python training/train_rnnt.py --train
  python training/train_rnnt.py --evaluate-test         # once, after everything is frozen
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.run_pretrained import (  # noqa: E402
    corpus_scores, environment, load_audio, load_manifest, prediction_rows, write_csv,
)

OUT = ROOT / "results" / "rnnt"
MODEL_DIR = ROOT / "models" / "rnnt"
VOCAB = ROOT / "output" / "ctc_vocab.json"
KEY = "rnnt"
SAMPLE_RATE = 16_000


@dataclass
class Config:
    seed: int = 2105
    # features
    n_mels: int = 80
    # model
    d_model: int = 256
    n_heads: int = 4
    ff_dim: int = 1024
    enc_layers: int = 8
    pred_dim: int = 320
    joint_dim: int = 320
    dropout: float = 0.1
    # training
    epochs: int = 60
    max_batch_seconds: float = 240.0
    max_clip_seconds: float = 16.0
    learning_rate: float = 1e-3
    warmup_steps: int = 2000
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    time_masks: int = 2
    time_mask_max: int = 40
    freq_masks: int = 2
    freq_mask_max: int = 15
    eval_every: int = 2
    early_stopping_patience: int = 6           # evaluations without dev WER improvement
    plateau_guard_epoch: int = 16              # stop if dev WER still >= plateau_guard_wer here
    plateau_guard_wer: float = 0.98
    max_symbols_per_frame: int = 5
    fp16: bool = True


# ---------------- text ----------------

class CharVocab:
    def __init__(self, path: Path = VOCAB):
        self.stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        self.itos = {i: s for s, i in self.stoi.items()}
        self.blank = self.stoi["[PAD]"]
        self.unk = self.stoi["[UNK]"]
        self.space = self.stoi["|"]

    def __len__(self):
        return max(self.itos) + 1

    def encode(self, text: str) -> list[int]:
        return [self.space if ch == " " else self.stoi.get(ch, self.unk) for ch in text.strip()]

    def decode(self, ids) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if i in (self.blank, self.unk):
                continue
            chars.append(" " if i == self.space else self.itos[i])
        return " ".join("".join(chars).split())


# ---------------- features ----------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_filterbank(n_mels=80, n_fft=400, sr=SAMPLE_RATE) -> torch.Tensor:
    """HTK-style triangular mel filters, shape (n_fft // 2 + 1, n_mels)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(20.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (fft_freqs - lo) / (mid - lo)
        down = (hi - fft_freqs) / (hi - mid)
        fb[:, m] = np.clip(np.minimum(up, down), 0, None)
    return torch.from_numpy(fb)


class LogMel:
    def __init__(self, n_mels=80):
        self.fb = mel_filterbank(n_mels)
        self.window = torch.hann_window(400)

    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        spec = torch.stft(wav, n_fft=400, hop_length=160, win_length=400, window=self.window,
                          center=True, return_complex=True).abs().pow(2)      # (freq, frames)
        mel = torch.log(torch.clamp(spec.T @ self.fb, min=1e-10))             # (frames, n_mels)
        return (mel - mel.mean(0)) / (mel.std(0) + 1e-5)


def spec_augment(feats: torch.Tensor, cfg: Config) -> torch.Tensor:
    feats = feats.clone()
    frames, mels = feats.shape
    for _ in range(cfg.freq_masks):
        w = random.randint(0, cfg.freq_mask_max)
        f0 = random.randint(0, max(0, mels - w))
        feats[:, f0:f0 + w] = 0
    for _ in range(cfg.time_masks):
        w = random.randint(0, min(cfg.time_mask_max, frames // 5))
        t0 = random.randint(0, max(0, frames - w))
        feats[t0:t0 + w, :] = 0
    return feats


# ---------------- data ----------------

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, records, vocab: CharVocab, cfg: Config, augment: bool):
        self.records, self.vocab, self.cfg, self.augment = records, vocab, cfg, augment
        self.logmel = LogMel(cfg.n_mels)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        feats = self.logmel(load_audio(record))
        if self.augment:
            feats = spec_augment(feats, self.cfg)
        return feats, torch.tensor(self.vocab.encode(record["text"]), dtype=torch.long), index


def collate(batch):
    feats, targets, idx = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    x = torch.zeros(len(feats), int(feat_lens.max()), feats[0].shape[1])
    y = torch.zeros(len(targets), max(1, int(tgt_lens.max())), dtype=torch.long)
    for i, (f, t) in enumerate(zip(feats, targets)):
        x[i, :len(f)] = f
        y[i, :len(t)] = t
    return x, feat_lens, y, tgt_lens, list(idx)


def duration_batches(records, max_seconds: float, shuffle: bool, rng: random.Random):
    """Batches of similar-length clips whose padded duration stays under max_seconds."""
    order = sorted(range(len(records)), key=lambda i: records[i]["duration_sec"])
    batches, current = [], []
    for i in order:
        longest = max([records[j]["duration_sec"] for j in current] + [records[i]["duration_sec"]])
        if current and longest * (len(current) + 1) > max_seconds:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches


# ---------------- model ----------------

def _conv_out(length):
    return (length - 3) // 2 + 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.d_model
        self.subsample = nn.Sequential(nn.Conv2d(1, c, 3, 2), nn.ReLU(), nn.Conv2d(c, c, 3, 2), nn.ReLU())
        freq = _conv_out(_conv_out(cfg.n_mels))
        self.proj = nn.Linear(c * freq, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, feats, lengths):
        x = self.subsample(feats.unsqueeze(1))                  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.proj(x.transpose(1, 2).reshape(b, t, c * f))
        lengths = _conv_out(_conv_out(lengths)).clamp(min=1, max=t)
        pad = torch.arange(t, device=x.device)[None, :] >= lengths[:, None]
        x = self.layers(self.drop(self.pos(x)), src_key_padding_mask=pad)
        return self.norm(x), lengths


class Predictor(nn.Module):
    def __init__(self, vocab_size, cfg: Config, blank: int):
        super().__init__()
        self.blank = blank
        self.embed = nn.Embedding(vocab_size, cfg.pred_dim)
        self.lstm = nn.LSTM(cfg.pred_dim, cfg.pred_dim, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, targets, state=None):
        """targets (B, U) -> outputs (B, U+1, H), prepending the blank start symbol."""
        start = torch.full((targets.shape[0], 1), self.blank, dtype=torch.long, device=targets.device)
        out, _ = self.lstm(self.drop(self.embed(torch.cat([start, targets], 1))), state)
        return out

    def step(self, tokens, state):
        out, state = self.lstm(self.embed(tokens).unsqueeze(1), state)
        return out.squeeze(1), state


class Joiner(nn.Module):
    def __init__(self, vocab_size, cfg: Config):
        super().__init__()
        self.enc = nn.Linear(cfg.d_model, cfg.joint_dim)
        self.pred = nn.Linear(cfg.pred_dim, cfg.joint_dim)
        self.out = nn.Linear(cfg.joint_dim, vocab_size)

    def forward(self, enc, pred):
        """enc (B, T, D), pred (B, U+1, H) -> logits (B, T, U+1, V)."""
        return self.out(torch.tanh(self.enc(enc).unsqueeze(2) + self.pred(pred).unsqueeze(1)))

    def frame(self, enc_t, pred_t):
        return self.out(torch.tanh(self.enc(enc_t) + self.pred(pred_t)))


class RNNT(nn.Module):
    def __init__(self, vocab_size: int, blank: int, cfg: Config):
        super().__init__()
        self.blank = blank
        self.encoder = Encoder(cfg)
        self.predictor = Predictor(vocab_size, cfg, blank)
        self.joiner = Joiner(vocab_size, cfg)

    def forward(self, feats, feat_lens, targets):
        enc, enc_lens = self.encoder(feats, feat_lens)
        pred = self.predictor(targets)
        return self.joiner(enc.float(), pred.float()), enc_lens

    @torch.no_grad()
    def greedy(self, feats, feat_lens, max_symbols=5):
        enc, enc_lens = self.encoder(feats, feat_lens)
        enc = enc.float()
        b, t_max, _ = enc.shape
        device = enc.device
        tokens = torch.full((b,), self.blank, dtype=torch.long, device=device)
        pred, state = self.predictor.step(tokens, None)
        pred = pred.float()
        hyps = [[] for _ in range(b)]
        for t in range(t_max):
            active = t < enc_lens
            for _ in range(max_symbols):
                k = self.joiner.frame(enc[:, t], pred).argmax(-1)
                emit = active & (k != self.blank)
                if not emit.any():
                    break
                new_pred, new_state = self.predictor.step(k, state)
                m = emit.unsqueeze(-1)
                pred = torch.where(m, new_pred.float(), pred)
                state = tuple(torch.where(m.unsqueeze(0), ns, s) for ns, s in zip(new_state, state))
                for i in emit.nonzero().flatten().tolist():
                    hyps[i].append(int(k[i]))
                active = emit
        return hyps


def rnnt_loss(logits, targets, logit_lens, target_lens, blank):
    from torchaudio.functional import rnnt_loss as _loss
    return _loss(logits.float(), targets.int(), logit_lens.int(), target_lens.int(),
                 blank=blank, reduction="mean")


# ---------------- training ----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(records, vocab, cfg, *, augment, shuffle, rng):
    dataset = ClipDataset(records, vocab, cfg, augment)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=duration_batches(records, cfg.max_batch_seconds, shuffle, rng),
        collate_fn=collate, num_workers=2 if torch.cuda.is_available() else 0)


def decode_records(model, records, vocab, cfg, device):
    model.eval()
    hyps = {}
    loader = make_loader(records, vocab, cfg, augment=False, shuffle=False, rng=random.Random(0))
    with torch.autocast(device, enabled=cfg.fp16 and device == "cuda"):
        for x, x_lens, _, _, idx in loader:
            out = model.greedy(x.to(device), x_lens.to(device), cfg.max_symbols_per_frame)
            hyps.update({records[i]["id"]: vocab.decode(h) for i, h in zip(idx, out)})
    return hyps


def lr_at(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build(cfg, vocab, device):
    model = RNNT(len(vocab), vocab.blank, cfg).to(device)
    return model, sum(p.numel() for p in model.parameters())


def train(cfg: Config, limit: int | None):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = CharVocab()
    train_records = [r for r in load_manifest("train") if r["duration_sec"] <= cfg.max_clip_seconds]
    dropped = len(load_manifest("train")) - len(train_records)
    dev_records = load_manifest("dev")
    if limit:
        train_records, dev_records = train_records[:limit], dev_records[:max(16, limit // 4)]

    model, n_params = build(cfg, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device == "cuda")
    rng = random.Random(cfg.seed)
    steps_per_epoch = len(duration_batches(train_records, cfg.max_batch_seconds, False, rng))
    total_steps = steps_per_epoch * cfg.epochs
    print(f"params {n_params:,}  train clips {len(train_records)}  steps/epoch {steps_per_epoch}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log, best_wer, best_epoch, evals_since_best, step = [], float("inf"), 0, 0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stopped_by = "epochs"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loader = make_loader(train_records, vocab, cfg, augment=True, shuffle=True, rng=rng)
        running, seen = 0.0, 0
        for x, x_lens, y, y_lens, _ in loader:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg, total_steps)
            x, x_lens, y, y_lens = x.to(device), x_lens.to(device), y.to(device), y_lens.to(device)
            with torch.autocast(device, enabled=cfg.fp16 and device == "cuda"):
                logits, enc_lens = model(x, x_lens, y)
            loss = rnnt_loss(logits, y, enc_lens, y_lens, vocab.blank)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running, seen, step = running + loss.item(), seen + 1, step + 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / seen:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} ({time.perf_counter() - start:.0f}s)", flush=True)
        entry = {"epoch": epoch, "step": step, "train_loss": running / max(1, seen),
                 "seconds": round(time.perf_counter() - start, 1)}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            rows = prediction_rows(dev_records, decode_records(model, dev_records, vocab, cfg, device))
            scores = corpus_scores(rows)
            entry.update(dev_wer=scores["wer"], dev_cer=scores["cer"])
            print(f"epoch {epoch}: train loss {entry['train_loss']:.3f}  dev WER {scores['wer']:.4f}  "
                  f"CER {scores['cer']:.4f}  sample: {rows[0]['hypothesis']!r}", flush=True)
            if scores["wer"] < best_wer:
                best_wer, best_epoch, evals_since_best = scores["wer"], epoch, 0
                torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), "epoch": epoch},
                           MODEL_DIR / "best.pt")
            else:
                evals_since_best += 1
        log.append(entry)
        (OUT / "training_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        if "dev_wer" in entry and epoch >= cfg.plateau_guard_epoch and best_wer >= cfg.plateau_guard_wer:
            stopped_by = "plateau_guard"
            print(f"PLATEAU GUARD: dev WER still {best_wer:.3f} at epoch {epoch}; stopping.", flush=True)
            break
        if evals_since_best >= cfg.early_stopping_patience:
            stopped_by = "early_stopping"
            break

    train_seconds = time.perf_counter() - start
    plot_curves(log, OUT / "training_curves.png")
    summary = {
        "config": asdict(cfg), "parameters": n_params,
        "train_utterances": len(train_records),
        "train_hours": round(sum(r["duration_sec"] for r in train_records) / 3600, 5),
        "dropped_longer_than_max_clip_seconds": None if limit else dropped,
        "best_epoch": best_epoch, "best_dev_wer_during_training": best_wer,
        "epochs_run": log[-1]["epoch"], "updates": step, "stopped_by": stopped_by,
        "train_seconds": round(train_seconds, 1), "gpus_used": 1 if device == "cuda" else 0,
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if device == "cuda" else None),
        "smoke_test_limit": limit,
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if stopped_by == "plateau_guard":
        raise SystemExit("No usable model: the plateau guard stopped training. Test not decoded.")
    decode_split("dev", summary, limit)
    return summary


def plot_curves(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot([e["epoch"] for e in log], [e["train_loss"] for e in log], marker=".")
    left.set(xlabel="epoch", ylabel="RNN-T loss", title="Train loss")
    dev = [e for e in log if "dev_wer" in e]
    if dev:
        right.plot([e["epoch"] for e in dev], [e["dev_wer"] * 100 for e in dev], marker="o", label="dev WER")
        right.plot([e["epoch"] for e in dev], [e["dev_cer"] * 100 for e in dev], marker="o", label="dev CER")
        right.legend()
    right.set(xlabel="epoch", ylabel="%", title="Dev error rate (greedy)")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_best(device):
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location=device, weights_only=True)
    cfg = Config(**ckpt["config"])
    vocab = CharVocab()
    model, n_params = build(cfg, vocab, device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg, vocab, n_params, ckpt["epoch"]


def decode_split(split, training=None, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, vocab, n_params, epoch = load_best(device)
    records = load_manifest(split)[:limit] if limit else load_manifest(split)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    hyps = decode_records(model, records, vocab, cfg, device)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    rows = prediction_rows(records, hyps)
    tag = f"{KEY}_{split}" + (f"_first{limit}" if limit else "")
    write_csv(OUT / "predictions" / f"{tag}.csv", rows)
    if training is None and (OUT / "training_summary.json").exists():
        training = json.loads((OUT / "training_summary.json").read_text(encoding="utf-8"))
    audio_seconds = sum(r["duration_sec"] for r in records)
    info = {
        "model_key": KEY,
        "checkpoint": "our RNN-T trained from scratch (models/rnnt/best.pt)",
        "selected_epoch": epoch,
        "trained_by_our_group": True,
        "family": "transducer (RNN-T)",
        "decoding": f"batched greedy, max {cfg.max_symbols_per_frame} symbols per frame, no LM",
        "split": split, "utterances": len(rows), "audio_hours": round(audio_seconds / 3600, 5),
        "device": device, "dtype": "float16 autocast" if device == "cuda" and cfg.fp16 else "float32",
        "batching": f"duration-sorted, <= {cfg.max_batch_seconds} s padded audio per batch",
        "parameters": n_params,
        "wall_seconds_incl_audio_loading": round(seconds, 2),
        "rtf_incl_audio_loading": round(seconds / audio_seconds, 4),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if device == "cuda" else None),
        "text_normalization": "prepare.normalize_text (lowercase, NFC, punctuation/symbols -> space)",
        "scores_preliminary": corpus_scores(rows),
        "training": training,
        "environment": environment(),
    }
    path = OUT / "model_info" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tag}: WER {info['scores_preliminary']['wer']:.4f} CER {info['scores_preliminary']['cer']:.4f} "
          f"RTF {info['rtf_incl_audio_loading']}", flush=True)
    return info


# ---------------- sanity check ----------------

def overfit(n_clips: int, steps: int, cfg: Config):
    """Memorize n short train clips without augmentation. A working pipeline gets near-zero loss
    and copies the references; if not, the problem is in features/labels/loss, not hyperparameters."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = CharVocab()
    pool = [r for r in load_manifest("train") if 2.0 <= r["duration_sec"] <= 6.0]
    records = [pool[i] for i in np.random.default_rng(cfg.seed).choice(len(pool), n_clips, replace=False)]
    model, n_params = build(cfg, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    logmel = LogMel(cfg.n_mels)
    data = collate([(logmel(load_audio(r)), torch.tensor(vocab.encode(r["text"])), i)
                    for i, r in enumerate(records)])
    x, x_lens, y, y_lens = (t.to(device) for t in data[:4])
    print(f"overfit: {n_clips} clips, {n_params:,} params, {device}", flush=True)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        logits, enc_lens = model(x, x_lens, y)
        loss = rnnt_loss(logits, y, enc_lens, y_lens, vocab.blank)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step == 1 or step % 50 == 0:
            model.eval()
            hyp = vocab.decode(model.greedy(x[:1], x_lens[:1])[0])
            print(f"  step {step:4d}  loss {loss.item():8.3f}  ref {records[0]['text'][:40]!r}  "
                  f"hyp {hyp[:40]!r}  ({time.perf_counter() - start:.0f}s)", flush=True)
    return loss.item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    mode.add_argument("--overfit", type=int, metavar="N_CLIPS")
    parser.add_argument("--overfit-steps", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
        cfg.eval_every = min(cfg.eval_every, args.epochs)
    if args.overfit:
        overfit(args.overfit, args.overfit_steps, cfg)
    elif args.train:
        train(cfg, args.limit)
    else:
        decode_split("test")


if __name__ == "__main__":
    main()
