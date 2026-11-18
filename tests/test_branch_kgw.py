"""Offline CPU tests with random tiny causal models; no checkpoint downloads."""
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils.transformers_config import TransformersConfig
from watermark.branch_kgw import BranchKGW, BranchOptions
from watermark.branch_kgw.branch_kgw import select_candidate


class TinyTokenizer:
    all_special_ids = [0, 1]
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 16

    def get_vocab(self):
        return {str(i): i for i in range(16)}

    def encode(self, text, add_special_tokens=True):
        return [2, 3, 4]

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(i) for i in ids if i not in self.all_special_ids)


class BranchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = GPT2Config(vocab_size=16, n_positions=32, n_embd=8,
                            n_layer=1, n_head=1, bos_token_id=2, eos_token_id=1, pad_token_id=0)
        self.target = GPT2LMHeadModel(config).eval()
        self.draft = GPT2LMHeadModel(config).eval()
        self.tokenizer = TinyTokenizer()
        self.config = TransformersConfig(self.target, self.tokenizer, device='cpu')
        self.wm = BranchKGW(str(Path(__file__).resolve().parents[1] / 'config/KGW.json'),
                            self.config, self.draft, self.tokenizer,
                            BranchOptions(candidates=3, block_size=2, max_new_tokens=5))

    def test_quality_gate_and_control(self):
        self.assertEqual(select_candidate([-1., -1.1, -2.], [0., 1., 9.], .2), (1, [0, 1]))
        self.assertEqual(select_candidate([-1., -1.1, -2.], [0., 1., 9.], .2, False)[0], 0)
        self.assertEqual(select_candidate([-1., -1.1], [0., 1.], 0)[0], 0)

    def test_quality_alignment(self):
        prefix, candidate = [2, 3], [4, 5]
        ids = torch.tensor([prefix + candidate])
        with torch.no_grad():
            logits = self.target(ids).logits.log_softmax(-1)
        expected = (logits[0, 1, 4] + logits[0, 2, 5]).item() / 2
        self.assertAlmostEqual(self.wm._quality(prefix, candidate), expected, places=6)

    def test_generation_and_scoring(self):
        result = self.wm.generate('prompt')
        tokens = result['generated_token_ids']
        self.assertTrue(0 < len(tokens) <= 5)
        self.assertEqual(result['detection_with_prompt']['num_tokens_scored'], len(tokens))
        self.assertEqual(result['detection_without_prompt']['num_tokens_scored'], len(tokens) - 1)
        reconstructed = []
        for block in result['trace']:
            i = block['selected_index']
            self.assertIn(i, block['eligible_indices'])
            self.assertGreaterEqual(block['quality_scores'][i], max(block['quality_scores']) - .2)
            reconstructed.extend(block['candidate_token_ids'][i])
        self.assertEqual(tokens, reconstructed)
        if 1 in tokens:
            self.assertEqual(tokens.index(1), len(tokens) - 1)
        self.assertIsNone(self.wm.detect_token_ids([])['score'])

    def test_eos_trimming(self):
        self.draft.generate = lambda **kwargs: torch.tensor([[2, 3, 4, 1, 0]] * 3)
        result = self.wm.generate('prompt')
        self.assertEqual(result['generated_token_ids'], [1])
        self.assertEqual(len(result['trace']), 1)

    def test_budget_and_context(self):
        # Prevent EOS and verify a partial final block is generated.
        self.draft.generate = lambda **kw: torch.cat([
            kw['input_ids'].repeat(3, 1), torch.full((3, kw['max_new_tokens']), 5)], dim=1)
        result = self.wm.generate('prompt', watermarked=False)
        self.assertEqual(len(result['generated_token_ids']), 5)
        self.assertEqual([len(b['candidate_token_ids'][0]) for b in result['trace']], [2, 2, 1])
        self.wm.options = BranchOptions(max_new_tokens=32)
        with self.assertRaises(ValueError):
            self.wm.generate('prompt')

    def test_invalid_options(self):
        for kwargs in ({'epsilon': -1}, {'epsilon': float('nan')}, {'block_size': 0},
                       {'temperature': 0}, {'top_p': 2}, {'candidates': 1.5}):
            with self.assertRaises(ValueError):
                BranchOptions(**kwargs)
        different = TinyTokenizer()
        different.get_vocab = lambda: {'different': 0}
        with self.assertRaises(ValueError):
            BranchKGW(self.wm.config, self.config, self.draft, different)


if __name__ == '__main__':
    unittest.main()
