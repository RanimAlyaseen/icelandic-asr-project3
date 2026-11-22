import itertools
import math
import random
import unittest

import numpy as np
import torch

from training.train_rnnt import (
    RNNT, CharVocab, Config, LogMel, collate, duration_batches, mel_filterbank,
)


def reference_rnnt_loss(logits, targets, logit_lens, target_lens, blank):
    """Plain forward-algorithm RNN-T loss (Graves 2012), used to test shapes and the model wiring."""
    log_probs = logits.float().log_softmax(-1)
    losses = []
    for b in range(logits.shape[0]):
        t_len, u_len = int(logit_lens[b]), int(target_lens[b])
        alpha = torch.full((t_len, u_len + 1), -math.inf)
        alpha[0, 0] = 0.0
        for t in range(t_len):
            for u in range(u_len + 1):
                if t == 0 and u == 0:
                    continue
                terms = []
                if t > 0:
                    terms.append(alpha[t - 1, u] + log_probs[b, t - 1, u, blank])
                if u > 0:
                    terms.append(alpha[t, u - 1] + log_probs[b, t, u - 1, targets[b, u - 1]])
                alpha[t, u] = torch.logsumexp(torch.stack(terms), 0)
        losses.append(-(alpha[t_len - 1, u_len] + log_probs[b, t_len - 1, u_len, blank]))
    return torch.stack(losses).mean()


def tiny_config():
    return Config(d_model=32, n_heads=2, ff_dim=64, enc_layers=1, pred_dim=24, joint_dim=24,
                  dropout=0.0, fp16=False)


class RNNTTest(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocab()

    def test_vocab_round_trip_icelandic(self):
        text = "þú ert ágæt"
        ids = self.vocab.encode(text)
        self.assertNotIn(self.vocab.unk, ids)
        self.assertNotIn(self.vocab.blank, ids)
        self.assertEqual(self.vocab.decode(ids), text)

    def test_reference_loss_matches_path_enumeration(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 2, 4)
        lp = logits.log_softmax(-1)[0]
        target, blank = 2, 0
        # every path: one emission of `target` placed after some blanks, ending with a final blank
        total = []
        for t_emit in range(3):
            score = sum(lp[t, 0, blank] for t in range(t_emit)) + lp[t_emit, 0, target]
            score = score + sum(lp[t, 1, blank] for t in range(t_emit, 3))
            total.append(score)
        expected = -torch.logsumexp(torch.stack(total), 0)
        got = reference_rnnt_loss(logits, torch.tensor([[target]]), torch.tensor([3]), torch.tensor([1]), blank)
        self.assertAlmostEqual(got.item(), expected.item(), places=5)

    def test_torchaudio_loss_matches_reference(self):
        """Runs where torchaudio is installed (Kaggle/Colab); skipped elsewhere."""
        try:
            from training.train_rnnt import rnnt_loss
            import torchaudio  # noqa: F401
        except (ImportError, OSError) as err:
            self.skipTest(f"torchaudio unavailable: {err}")
        torch.manual_seed(1)
        logits = torch.randn(2, 6, 4, 9)
        targets = torch.tensor([[3, 5, 2], [4, 1, 0]])
        t_lens, u_lens = torch.tensor([6, 4]), torch.tensor([3, 2])
        expected = reference_rnnt_loss(logits, targets, t_lens, u_lens, 0)
        got = rnnt_loss(logits, targets, t_lens, u_lens, 0)
        self.assertAlmostEqual(got.item(), expected.item(), places=4)

    def test_log_mel_shape_and_normalization(self):
        feats = LogMel()(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)
        self.assertEqual(feats.shape, (101, 80))
        self.assertLess(feats.mean(0).abs().max().item(), 1e-4)
        self.assertEqual(mel_filterbank().shape, (201, 80))

    def test_duration_batches_cover_all_and_respect_budget(self):
        records = [{"duration_sec": d} for d in np.random.default_rng(1).uniform(1, 12, 200)]
        batches = duration_batches(records, 60.0, True, random.Random(0))
        self.assertEqual(sorted(itertools.chain(*batches)), list(range(200)))
        for b in batches:
            self.assertTrue(len(b) == 1 or max(records[i]["duration_sec"] for i in b) * len(b) <= 60.0)

    def test_forward_loss_and_greedy_shapes(self):
        torch.manual_seed(0)
        cfg = tiny_config()
        model = RNNT(len(self.vocab), self.vocab.blank, cfg).eval()
        logmel = LogMel()
        batch = [(logmel(np.random.default_rng(i).standard_normal(8000 + 4000 * i).astype(np.float32) * 0.1),
                  torch.tensor(self.vocab.encode(text)), i)
                 for i, text in enumerate(["já", "góðan dag"])]
        x, x_lens, y, y_lens, _ = collate(batch)
        logits, enc_lens = model(x, x_lens, y)
        self.assertEqual(logits.shape[:3], (2, int(enc_lens.max()), int(y_lens.max()) + 1))
        loss = reference_rnnt_loss(logits, y, enc_lens, y_lens, self.vocab.blank)
        self.assertTrue(torch.isfinite(loss))
        hyps = model.greedy(x, x_lens)
        self.assertEqual(len(hyps), 2)
        self.assertTrue(all(self.vocab.blank not in h for h in hyps))

    def test_tiny_model_learns_one_clip(self):
        """Loss must fall far below its start; greedy output is tested separately because a tiny
        model spreads emissions over frames long before it becomes peaky."""
        torch.manual_seed(0)
        cfg = tiny_config()
        model = RNNT(len(self.vocab), self.vocab.blank, cfg)
        feats = LogMel()(np.random.default_rng(3).standard_normal(16000).astype(np.float32) * 0.1)
        x, x_lens, y, y_lens, _ = collate([(feats, torch.tensor(self.vocab.encode("já")), 0)])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = None
        for _ in range(300):
            logits, enc_lens = model(x, x_lens, y)
            loss = reference_rnnt_loss(logits, y, enc_lens, y_lens, self.vocab.blank)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 1.0)
        self.assertLess(loss.item(), first / 20)

    def test_greedy_follows_joiner_decisions(self):
        """Joiner scripted to emit 'j' at frame 0 and 'á' at frame 2: greedy must output 'já',
        stop each frame on blank, and ignore frames past each clip's length."""
        cfg = tiny_config()
        model = RNNT(len(self.vocab), self.vocab.blank, cfg).eval()
        v, script, calls = self.vocab, {0: "j", 2: "á"}, {}

        def fake_encoder(feats, lengths):
            t = feats.shape[1]
            enc = torch.zeros(feats.shape[0], t, cfg.d_model)
            enc[:, :, 0] = torch.arange(t, dtype=torch.float)
            return enc, lengths

        def fake_frame(enc_t, pred_t):
            t = int(enc_t[0, 0])
            n = calls.get(t, 0)
            calls[t] = n + 1
            logits = torch.zeros(enc_t.shape[0], len(v))
            token = v.stoi[script[t]] if t in script and n == 0 else v.blank
            logits[:, token] = 5.0
            return logits

        model.encoder.forward = fake_encoder
        model.joiner.frame = fake_frame
        hyps = model.greedy(torch.zeros(2, 4, cfg.n_mels), torch.tensor([4, 1]))
        self.assertEqual(v.decode(hyps[0]), "já")
        self.assertEqual(v.decode(hyps[1]), "j")

if __name__ == "__main__":
    unittest.main()
