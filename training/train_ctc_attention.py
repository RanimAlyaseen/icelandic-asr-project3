"""Train CTC, attention encoder-decoder, and hybrid CTC/attention models from scratch for Icelandic.

Course baseline families "CTC encoder" and "Attention-based encoder-decoder, including hybrid CTC/attention".
All three share the RNN-T pipeline in training/train_rnnt.py, so the four neural families are compared with the
same features, encoder, data, batching, and training budget:

  80-dim log-mel -> 4x conv subsampling -> 8-layer pre-norm Transformer encoder (d=256)    [shared]
  --mode ctc        linear CTC head over the train-only characters, greedy CTC decoding
  --mode attention  4-layer Transformer decoder with cross-attention, label-smoothed cross-entropy,
                    beam search (beam 5, length-normalized score)
  --mode hybrid     both heads, loss 0.3 * CTC + 0.7 * attention (Watanabe et al., 2017);
                    attention beam search, n-best rescored with 0.3 * CTC + 0.7 * attention log-probability

Dev WER with greedy decoding (every `eval_every` epochs) selects the checkpoint. The final dev and test
decodings use each mode's final decoder. Test is decoded once with --evaluate-test.

Outputs: results/<key>/{predictions,model_info}/<key>_{dev,test}.*, training_log.json, training_curves.png,
training_summary.json. Weights: models/<key>/best.pt (published on the Hugging Face Hub; see README).
Keys: ctc_scratch, attention_scratch, hybrid_ctc_attention.

Usage (repository root, one GPU):
  python training/train_ctc_attention.py --mode hybrid --overfit 32
  python training/train_ctc_attention.py --mode hybrid --train
  python training/train_ctc_attention.py --mode hybrid --evaluate-test
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
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.run_pretrained import (  # noqa: E402
    corpus_scores, environment, load_audio, load_manifest, prediction_rows, write_csv,
)
from training.train_rnnt import (  # noqa: E402
    CharVocab, Encoder, LogMel, PositionalEncoding, collate, duration_batches, lr_at, make_loader, set_seed,
)

KEYS = {"ctc": "ctc_scratch", "attention": "attention_scratch", "hybrid": "hybrid_ctc_attention"}
CTC_WEIGHT = {"ctc": 1.0, "attention": 0.0, "hybrid": 0.3}


@dataclass
class Config:
    mode: str = "hybrid"
    seed: int = 2105
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 4
    ff_dim: int = 1024
    enc_layers: int = 8
    dec_layers: int = 4
    dropout: float = 0.1
    ctc_weight: float = 0.3                  # training loss weight; 1.0 = pure CTC, 0.0 = pure attention
    label_smoothing: float = 0.1
    epochs: int = 60
    max_batch_seconds: float = 240.0
    decode_batch_seconds: float = 120.0
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
    early_stopping_patience: int = 6
    plateau_guard_epoch: int = 16
    plateau_guard_wer: float = 0.98
    beam_size: int = 5
    decode_ctc_weight: float = 0.3           # hybrid n-best rescoring weight
    fp16: bool = True

    @classmethod
    def for_mode(cls, mode: str, **kw):
        return cls(mode=mode, ctc_weight=CTC_WEIGHT[mode], **kw)


# ---------------- model ----------------

class Decoder(nn.Module):
    def __init__(self, vocab_size, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerDecoder(layer, cfg.dec_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, vocab_size)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, ys, memory, memory_pad, ys_pad=None):
        length = ys.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ys.device), 1)
        x = self.drop(self.pos(self.embed(ys) * self.scale))
        x = self.layers(x, memory, tgt_mask=causal, tgt_key_padding_mask=ys_pad,
                        memory_key_padding_mask=memory_pad)
        return self.out(self.norm(x))


class EncDec(nn.Module):
    """Shared encoder with an optional CTC head and an optional attention decoder."""

    def __init__(self, vocab: CharVocab, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blank = vocab.blank
        self.n_chars = len(vocab)
        self.sos, self.eos = self.n_chars, self.n_chars + 1    # attention vocabulary = characters + sos + eos
        self.encoder = Encoder(cfg)
        self.ctc_head = nn.Linear(cfg.d_model, self.n_chars) if cfg.ctc_weight > 0 else None
        self.decoder = Decoder(self.n_chars + 2, cfg) if cfg.ctc_weight < 1 else None

    def encode(self, feats, feat_lens):
        enc, enc_lens = self.encoder(feats, feat_lens)
        pad = torch.arange(enc.shape[1], device=enc.device)[None, :] >= enc_lens[:, None]
        return enc, enc_lens, pad

    def ctc_log_probs(self, enc):
        return self.ctc_head(enc).float().log_softmax(-1)

    def loss(self, feats, feat_lens, targets, target_lens):
        """targets: (B, U) character ids, zero-padded. Returns total, ctc, attention losses."""
        enc, enc_lens, pad = self.encode(feats, feat_lens)
        total = enc.new_zeros((), dtype=torch.float32)
        ctc = att = None
        if self.ctc_head is not None:
            ctc = ctc_loss(self.ctc_log_probs(enc), enc_lens, targets, target_lens, self.blank)
            total = total + self.cfg.ctc_weight * ctc
        if self.decoder is not None:
            ys_in, ys_out, ys_pad = teacher_forcing(targets, target_lens, self.sos, self.eos)
            logits = self.decoder(ys_in, enc, pad, ys_pad).float()
            att = F.cross_entropy(logits.transpose(1, 2), ys_out, ignore_index=-100,
                                  label_smoothing=self.cfg.label_smoothing)
            total = total + (1 - self.cfg.ctc_weight) * att
        return total, ctc, att

    @torch.no_grad()
    def greedy(self, feats, feat_lens):
        enc, enc_lens, pad = self.encode(feats, feat_lens)
        if self.decoder is None:
            return ctc_greedy(self.ctc_log_probs(enc), enc_lens, self.blank)
        return attention_greedy(self.decoder, enc, pad, self.sos, self.eos, max_len(enc_lens))

    @torch.no_grad()
    def decode(self, feats, feat_lens):
        """Final decoder per mode: CTC greedy, attention beam, or attention beam + CTC rescoring."""
        if self.decoder is None:
            return self.greedy(feats, feat_lens)
        enc, enc_lens, pad = self.encode(feats, feat_lens)
        hyps, att_scores = beam_search(self.decoder, enc, pad, self.sos, self.eos,
                                       self.cfg.beam_size, max_len(enc_lens))
        if self.ctc_head is None:
            lengths = torch.tensor([[len(h) + 1 for h in row] for row in hyps], device=att_scores.device)
            best = (att_scores / lengths).argmax(1)
        else:
            ctc_scores = ctc_nbest_scores(self.ctc_log_probs(enc), enc_lens, hyps, self.blank)
            w = self.cfg.decode_ctc_weight
            best = ((1 - w) * att_scores + w * ctc_scores).argmax(1)
        return [hyps[b][int(k)] for b, k in enumerate(best)]


def max_len(enc_lens):
    return min(300, int(enc_lens.max()) + 5)


def ctc_loss(log_probs, in_lens, targets, target_lens, blank):
    return F.ctc_loss(log_probs.transpose(0, 1), targets, in_lens, target_lens, blank=blank,
                      reduction="mean", zero_infinity=True)


def teacher_forcing(targets, target_lens, sos, eos):
    b, u = targets.shape
    ys_in = torch.full((b, u + 1), eos, dtype=torch.long, device=targets.device)
    ys_out = torch.full((b, u + 1), -100, dtype=torch.long, device=targets.device)
    ys_in[:, 0] = sos
    for i, n in enumerate(target_lens.tolist()):
        ys_in[i, 1:n + 1] = targets[i, :n]
        ys_out[i, :n] = targets[i, :n]
        ys_out[i, n] = eos
    ys_pad = torch.arange(u + 1, device=targets.device)[None, :] > target_lens[:, None]
    return ys_in, ys_out, ys_pad


def ctc_greedy(log_probs, lengths, blank):
    best = log_probs.argmax(-1).cpu()
    out = []
    for row, n in zip(best, lengths.tolist()):
        ids, prev = [], None
        for k in row[:n].tolist():
            if k != prev and k != blank:
                ids.append(k)
            prev = k
        out.append(ids)
    return out


def _strip(seq, eos):
    seq = seq.tolist()
    return seq[:seq.index(eos)] if eos in seq else seq


def attention_greedy(decoder, enc, pad, sos, eos, limit):
    b = enc.shape[0]
    ys = torch.full((b, 1), sos, dtype=torch.long, device=enc.device)
    done = torch.zeros(b, dtype=torch.bool, device=enc.device)
    for _ in range(limit):
        nxt = decoder(ys, enc, pad)[:, -1].float()
        nxt[:, sos] = -math.inf
        tok = nxt.argmax(-1)
        tok = torch.where(done, torch.full_like(tok, eos), tok)
        ys = torch.cat([ys, tok[:, None]], 1)
        done |= tok == eos
        if done.all():
            break
    return [_strip(row[1:], eos) for row in ys]


def beam_search(decoder, enc, pad, sos, eos, beam, limit):
    """Batched beam search. Returns hyps[b][k] (token lists) and summed log-probs (B, K)."""
    b, device = enc.shape[0], enc.device
    memory, memory_pad = enc.repeat_interleave(beam, 0), pad.repeat_interleave(beam, 0)
    ys = torch.full((b * beam, 1), sos, dtype=torch.long, device=device)
    scores = torch.full((b, beam), -math.inf, device=device)
    scores[:, 0] = 0.0
    done = torch.zeros(b * beam, dtype=torch.bool, device=device)
    for _ in range(limit):
        logp = decoder(ys, memory, memory_pad)[:, -1].float().log_softmax(-1)
        logp[:, sos] = -math.inf
        logp[done] = -math.inf
        logp[done, eos] = 0.0
        vocab = logp.shape[-1]
        cand = (scores.reshape(-1, 1) + logp).reshape(b, beam * vocab)
        scores, idx = cand.topk(beam, dim=1)
        src = (torch.arange(b, device=device)[:, None] * beam + idx // vocab).reshape(-1)
        tok = (idx % vocab).reshape(-1)
        ys = torch.cat([ys[src], tok[:, None]], 1)
        done = done[src] | (tok == eos)
        if done.all():
            break
    hyps = [[_strip(ys[i * beam + k, 1:], eos) for k in range(beam)] for i in range(b)]
    return hyps, scores


def ctc_nbest_scores(log_probs, lengths, hyps, blank):
    """CTC log-likelihood of every n-best hypothesis, shape (B, K)."""
    b, k = len(hyps), len(hyps[0])
    flat = [h for row in hyps for h in row]
    width = max(1, max(len(h) for h in flat))
    targets = torch.zeros(b * k, width, dtype=torch.long, device=log_probs.device)
    for i, h in enumerate(flat):
        if h:
            targets[i, :len(h)] = torch.tensor(h, device=log_probs.device)
    t_lens = torch.tensor([len(h) for h in flat], device=log_probs.device)
    lp = log_probs.repeat_interleave(k, 0).transpose(0, 1)
    nll = F.ctc_loss(lp, targets, lengths.repeat_interleave(k), t_lens, blank=blank,
                     reduction="none", zero_infinity=False)
    return (-torch.nan_to_num(nll, nan=1e9, posinf=1e9)).reshape(b, k)   # impossible alignment -> -1e9


# ---------------- train / decode ----------------

def paths(mode):
    key = KEYS[mode]
    return key, ROOT / "results" / key, ROOT / "models" / key


def decode_records(model, records, vocab, cfg, device, final: bool):
    model.eval()
    hyps = {}
    decode_cfg = Config(**{**asdict(cfg), "max_batch_seconds": cfg.decode_batch_seconds})
    loader = make_loader(records, vocab, decode_cfg, augment=False, shuffle=False, rng=random.Random(0))
    with torch.autocast(device, enabled=cfg.fp16 and device == "cuda"):
        for x, x_lens, _, _, idx in loader:
            x, x_lens = x.to(device), x_lens.to(device)
            out = model.decode(x, x_lens) if final else model.greedy(x, x_lens)
            hyps.update({records[i]["id"]: vocab.decode(h) for i, h in zip(idx, out)})
    return hyps


def build(cfg, vocab, device):
    model = EncDec(vocab, cfg).to(device)
    return model, sum(p.numel() for p in model.parameters())


def train(cfg: Config, limit: int | None):
    key, out_dir, model_dir = paths(cfg.mode)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = CharVocab()
    all_train = load_manifest("train")
    train_records = [r for r in all_train if r["duration_sec"] <= cfg.max_clip_seconds]
    dev_records = load_manifest("dev")
    if limit:
        train_records, dev_records = train_records[:limit], dev_records[:max(16, limit // 4)]

    model, n_params = build(cfg, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device == "cuda")
    rng = random.Random(cfg.seed)
    total_steps = len(duration_batches(train_records, cfg.max_batch_seconds, False, rng)) * cfg.epochs
    print(f"{key}: params {n_params:,}  train clips {len(train_records)}  steps {total_steps}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log, best_wer, best_epoch, evals_since_best, step = [], float("inf"), 0, 0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start, stopped_by = time.perf_counter(), "epochs"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loader = make_loader(train_records, vocab, cfg, augment=True, shuffle=True, rng=rng)
        sums, seen = {"loss": 0.0, "ctc": 0.0, "att": 0.0}, 0
        for x, x_lens, y, y_lens, _ in loader:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg, total_steps)
            x, x_lens, y, y_lens = x.to(device), x_lens.to(device), y.to(device), y_lens.to(device)
            with torch.autocast(device, enabled=cfg.fp16 and device == "cuda"):
                loss, ctc, att = model.loss(x, x_lens, y, y_lens)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            seen, step = seen + 1, step + 1
            sums["loss"] += loss.item()
            sums["ctc"] += ctc.item() if ctc is not None else 0.0
            sums["att"] += att.item() if att is not None else 0.0
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {sums['loss'] / seen:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} ({time.perf_counter() - start:.0f}s)", flush=True)
        entry = {"epoch": epoch, "step": step, "train_loss": sums["loss"] / max(1, seen),
                 "seconds": round(time.perf_counter() - start, 1)}
        if model.ctc_head is not None:
            entry["train_ctc_loss"] = sums["ctc"] / max(1, seen)
        if model.decoder is not None:
            entry["train_attention_loss"] = sums["att"] / max(1, seen)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            rows = prediction_rows(dev_records, decode_records(model, dev_records, vocab, cfg, device, final=False))
            scores = corpus_scores(rows)
            entry.update(dev_wer_greedy=scores["wer"], dev_cer_greedy=scores["cer"])
            print(f"epoch {epoch}: train loss {entry['train_loss']:.3f}  dev WER {scores['wer']:.4f}  "
                  f"CER {scores['cer']:.4f}  sample: {rows[0]['hypothesis']!r}", flush=True)
            if scores["wer"] < best_wer:
                best_wer, best_epoch, evals_since_best = scores["wer"], epoch, 0
                torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), "epoch": epoch},
                           model_dir / "best.pt")
            else:
                evals_since_best += 1
        log.append(entry)
        (out_dir / "training_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        if "dev_wer_greedy" in entry and epoch >= cfg.plateau_guard_epoch and best_wer >= cfg.plateau_guard_wer:
            stopped_by = "plateau_guard"
            print(f"PLATEAU GUARD: dev WER still {best_wer:.3f} at epoch {epoch}; stopping.", flush=True)
            break
        if evals_since_best >= cfg.early_stopping_patience:
            stopped_by = "early_stopping"
            break

    train_seconds = time.perf_counter() - start
    plot_curves(log, out_dir / "training_curves.png", key)
    summary = {
        "config": asdict(cfg), "parameters": n_params,
        "train_utterances": len(train_records),
        "train_hours": round(sum(r["duration_sec"] for r in train_records) / 3600, 5),
        "dropped_longer_than_max_clip_seconds": None if limit else len(all_train) - len(train_records),
        "best_epoch": best_epoch, "best_dev_wer_greedy_during_training": best_wer,
        "epochs_run": log[-1]["epoch"], "updates": step, "stopped_by": stopped_by,
        "train_seconds": round(train_seconds, 1), "gpus_used": 1 if device == "cuda" else 0,
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if device == "cuda" else None),
        "smoke_test_limit": limit,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if stopped_by == "plateau_guard":
        raise SystemExit("No usable model: the plateau guard stopped training. Test not decoded.")
    decode_split(cfg.mode, "dev", summary, limit)
    return summary


def plot_curves(log, path, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [e["epoch"] for e in log]
    left.plot(epochs, [e["train_loss"] for e in log], marker=".", label="total")
    for name in ("train_ctc_loss", "train_attention_loss"):
        if name in log[0]:
            left.plot(epochs, [e[name] for e in log], marker=".", label=name.replace("train_", ""))
    left.set(xlabel="epoch", ylabel="loss", title=f"Train loss ({key})")
    left.legend()
    dev = [e for e in log if "dev_wer_greedy" in e]
    if dev:
        right.plot([e["epoch"] for e in dev], [e["dev_wer_greedy"] * 100 for e in dev], marker="o", label="dev WER")
        right.plot([e["epoch"] for e in dev], [e["dev_cer_greedy"] * 100 for e in dev], marker="o", label="dev CER")
        right.legend()
    right.set(xlabel="epoch", ylabel="%", title="Dev error rate (greedy)")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_best(mode, device):
    _, _, model_dir = paths(mode)
    ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    cfg = Config(**ckpt["config"])
    vocab = CharVocab()
    model, n_params = build(cfg, vocab, device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg, vocab, n_params, ckpt["epoch"]


def decode_split(mode, split, training=None, limit=None):
    key, out_dir, _ = paths(mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, vocab, n_params, epoch = load_best(mode, device)
    records = load_manifest(split)[:limit] if limit else load_manifest(split)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    hyps = decode_records(model, records, vocab, cfg, device, final=True)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    rows = prediction_rows(records, hyps)
    tag = f"{key}_{split}" + (f"_first{limit}" if limit else "")
    write_csv(out_dir / "predictions" / f"{tag}.csv", rows)
    if training is None and (out_dir / "training_summary.json").exists():
        training = json.loads((out_dir / "training_summary.json").read_text(encoding="utf-8"))
    decoding = {
        "ctc": "greedy CTC, no LM",
        "attention": f"attention beam search, beam {cfg.beam_size}, length-normalized log-probability, no LM",
        "hybrid": (f"attention beam search, beam {cfg.beam_size}; n-best rescored with "
                   f"{cfg.decode_ctc_weight} x CTC + {1 - cfg.decode_ctc_weight:.1f} x attention log-probability, no LM"),
    }[mode]
    audio_seconds = sum(r["duration_sec"] for r in records)
    info = {
        "model_key": key,
        "checkpoint": f"our {mode} model trained from scratch (models/{key}/best.pt)",
        "selected_epoch": epoch,
        "trained_by_our_group": True,
        "family": {"ctc": "CTC encoder", "attention": "attention encoder-decoder",
                   "hybrid": "hybrid CTC/attention encoder-decoder"}[mode],
        "training_loss": f"{cfg.ctc_weight} x CTC + {1 - cfg.ctc_weight:.1f} x attention cross-entropy",
        "decoding": decoding,
        "split": split, "utterances": len(rows), "audio_hours": round(audio_seconds / 3600, 5),
        "device": device, "dtype": "float16 autocast" if device == "cuda" and cfg.fp16 else "float32",
        "batching": f"duration-sorted, <= {cfg.decode_batch_seconds} s padded audio per batch",
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
    path = out_dir / "model_info" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tag}: WER {info['scores_preliminary']['wer']:.4f} CER {info['scores_preliminary']['cer']:.4f} "
          f"RTF {info['rtf_incl_audio_loading']}", flush=True)
    return info


def overfit(mode, n_clips, steps, cfg: Config):
    """Memorize n short train clips without augmentation; greedy output must copy the references."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = CharVocab()
    pool = [r for r in load_manifest("train") if 2.0 <= r["duration_sec"] <= 6.0]
    records = [pool[i] for i in np.random.default_rng(cfg.seed).choice(len(pool), n_clips, replace=False)]
    cfg = Config(**{**asdict(cfg), "dropout": 0.0})
    model, n_params = build(cfg, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    logmel = LogMel(cfg.n_mels)
    data = collate([(logmel(load_audio(r)), torch.tensor(vocab.encode(r["text"])), i)
                    for i, r in enumerate(records)])
    x, x_lens, y, y_lens = (t.to(device) for t in data[:4])
    print(f"overfit {mode}: {n_clips} clips, {n_params:,} params, {device}", flush=True)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        loss, _, _ = model.loss(x, x_lens, y, y_lens)
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
    parser.add_argument("--mode", required=True, choices=sorted(KEYS))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--evaluate-test", action="store_true")
    action.add_argument("--overfit", type=int, metavar="N_CLIPS")
    parser.add_argument("--overfit-steps", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    cfg = Config.for_mode(args.mode)
    if args.epochs:
        cfg.epochs = args.epochs
        cfg.eval_every = min(cfg.eval_every, args.epochs)
    if args.overfit:
        overfit(args.mode, args.overfit, args.overfit_steps, cfg)
    elif args.train:
        train(cfg, args.limit)
    else:
        decode_split(args.mode, "test")


if __name__ == "__main__":
    main()
