import math
import unittest

import numpy as np
import torch

from training.train_ctc_attention import (
    Config, EncDec, beam_search, ctc_greedy, ctc_nbest_scores, teacher_forcing,
)
from training.train_rnnt import CharVocab, LogMel, collate


def tiny(mode):
    return Config.for_mode(mode, d_model=32, n_heads=2, ff_dim=64, enc_layers=1, dec_layers=1,
                           dropout=0.0, fp16=False, beam_size=3)


def batch(vocab, texts):
    logmel = LogMel()
    items = [(logmel(np.random.default_rng(i).standard_normal(8000 + 4000 * i).astype(np.float32) * 0.1),
              torch.tensor(vocab.encode(t)), i) for i, t in enumerate(texts)]
    return collate(items)


class CTCAttentionTest(unittest.TestCase):
    def setUp(self):
        self.vocab = CharVocab()

    def test_heads_per_mode(self):
        for mode, ctc, dec in [("ctc", True, False), ("attention", False, True), ("hybrid", True, True)]:
            model = EncDec(self.vocab, tiny(mode))
            self.assertEqual(model.ctc_head is not None, ctc, mode)
            self.assertEqual(model.decoder is not None, dec, mode)

    def test_teacher_forcing_shifts_and_pads(self):
        targets = torch.tensor([[5, 6, 7], [8, 0, 0]])
        ys_in, ys_out, pad = teacher_forcing(targets, torch.tensor([3, 1]), sos=38, eos=39)
        self.assertEqual(ys_in.tolist(), [[38, 5, 6, 7], [38, 8, 39, 39]])
        self.assertEqual(ys_out.tolist(), [[5, 6, 7, 39], [8, 39, -100, -100]])
        self.assertEqual(pad.tolist(), [[False] * 4, [False, False, True, True]])

    def test_ctc_greedy_collapses_repeats_and_blanks(self):
        lp = torch.full((1, 7, 5), -10.0)
        for t, k in enumerate([0, 3, 3, 0, 3, 4, 4]):
            lp[0, t, k] = 0.0
        self.assertEqual(ctc_greedy(lp, torch.tensor([7]), blank=0), [[3, 3, 4]])
        self.assertEqual(ctc_greedy(lp, torch.tensor([3]), blank=0), [[3]])

    def test_losses_finite_for_every_mode(self):
        x, x_lens, y, y_lens, _ = batch(self.vocab, ["já", "góðan dag"])
        for mode in ("ctc", "attention", "hybrid"):
            total, ctc, att = EncDec(self.vocab, tiny(mode)).loss(x, x_lens, y, y_lens)
            self.assertTrue(torch.isfinite(total), mode)
            self.assertEqual(ctc is not None, mode != "attention")
            self.assertEqual(att is not None, mode != "ctc")

    def test_beam_search_finds_best_path_of_scripted_decoder(self):
        # scripted decoder: after sos prefer token 5 (p=.6) over 6 (.4); after 5 -> eos; after 6 -> 7 (p=1) then eos.
        sos, eos, vocab = 8, 9, 10

        class Scripted(torch.nn.Module):
            def forward(self, ys, memory, memory_pad):
                out = torch.full((ys.shape[0], ys.shape[1], vocab), -1e4)
                last = ys[:, -1]
                for i, tok in enumerate(last.tolist()):
                    if tok == sos:
                        out[i, -1, 5], out[i, -1, 6] = math.log(.6), math.log(.4)
                    elif tok == 6:
                        out[i, -1, 7] = 0.0
                    else:
                        out[i, -1, eos] = 0.0
                return out

        enc = torch.zeros(1, 4, 2)
        hyps, scores = beam_search(Scripted(), enc, torch.zeros(1, 4, dtype=torch.bool), sos, eos, 2, 6)
        self.assertEqual(hyps[0][0], [5])
        self.assertEqual(hyps[0][1], [6, 7])
        self.assertAlmostEqual(scores[0, 0].item(), math.log(.6), places=4)

    def test_ctc_rescoring_prefers_hypothesis_the_ctc_head_supports(self):
        lp = torch.full((1, 6, 5), -10.0).log_softmax(-1)
        for t, k in enumerate([0, 3, 0, 4, 0, 0]):
            lp[0, t, k] = 0.0
        lp = lp.log_softmax(-1)
        scores = ctc_nbest_scores(lp, torch.tensor([6]), [[[3, 4], [3], [4, 4, 4, 4, 4, 4, 4]]], blank=0)
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertLess(scores[0, 2].item(), -1e8)       # longer than the input: impossible

    def test_tiny_models_learn_one_clip(self):
        x, x_lens, y, y_lens, _ = batch(self.vocab, ["já"])
        for mode in ("ctc", "attention", "hybrid"):
            torch.manual_seed(0)
            model = EncDec(self.vocab, tiny(mode))
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            first = None
            for _ in range(200):
                loss, _, _ = model.loss(x, x_lens, y, y_lens)
                first = first if first is not None else loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertLess(loss.item(), first / 5, mode)
            model.eval()
            if mode != "ctc":
                self.assertEqual(self.vocab.decode(model.greedy(x, x_lens)[0]), "já", mode)
                self.assertEqual(self.vocab.decode(model.decode(x, x_lens)[0]), "já", mode)


if __name__ == "__main__":
    unittest.main()
