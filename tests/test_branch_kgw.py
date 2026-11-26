"""Offline CPU tests with random tiny causal models; no checkpoint downloads."""
import unittest
from unittest.mock import Mock, patch
import contextlib
import io
import json
import tempfile
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from utils.transformers_config import TransformersConfig
from watermark.branch_kgw import BranchKGW, BranchOptions
from watermark.branch_kgw.branch_kgw import select_candidate


class TinyTokenizer:
    chat_template = 'test-template'
    all_special_ids = [0, 1]
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 16

    def get_vocab(self):
        return {str(i): i for i in range(16)}

    def encode(self, text, add_special_tokens=True):
        return [2, 3, 4]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [2, 8, 4]

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

    def test_direct_prompt_ids_and_fixed_detection(self):
        self.draft.generate = Mock(side_effect=lambda **kw: torch.cat([
            kw['input_ids'].repeat(3, 1), torch.full((3, kw['max_new_tokens']), 5)], dim=1))
        result = self.wm.generate(prompt_token_ids=[2, 8, 4])
        self.assertEqual(result['prompt_token_ids'], [2, 8, 4])
        self.assertEqual(self.draft.generate.call_args_list[0].kwargs['input_ids'].tolist(), [[2, 8, 4]])
        scores = self.wm.detect_fixed_lengths(result['generated_token_ids'], [3, 8])
        self.assertEqual(scores['3']['num_tokens_scored'], 2)
        self.assertFalse(scores['8']['available'])
        self.assertIsNone(scores['8']['score'])
        self.assertEqual(result['diagnostics']['selection_changed_block_rate'], 0)
        self.assertAlmostEqual(result['diagnostics']['duplicate_candidate_rate'], 2 / 3)
        with self.assertRaises(ValueError):
            self.wm.generate('prompt', prompt_token_ids=[2])

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

    def test_adaptive_gate_recomputed_and_mode_independent(self):
        self.wm.options = BranchOptions(candidates=2, max_candidates=5,
                                       block_size=1, max_new_tokens=1)
        # A new best candidate invalidates the initial candidate. Final batch
        # contains only one sample because the budget is not a batch multiple.
        for watermarked in (True, False):
            self.draft.generate = Mock(side_effect=[
                torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]]),
                torch.tensor([[2, 3, 4, 6], [2, 3, 4, 6]]),
                torch.tensor([[2, 3, 4, 7]])])
            self.wm._quality = Mock(side_effect=lambda ctx, c: {5: -1., 6: -.5, 7: -.6}[c[0]])
            self.wm._green_flags = lambda ctx, c: [int(t == 7) for t in c]
            result = self.wm.generate('prompt', watermarked=watermarked)
            block = result['trace'][0]
            self.assertEqual(block['eligible_indices'], [1, 2])
            self.assertEqual(block['sampled_candidate_count'], 5)
            self.assertEqual(block['duplicate_candidate_count'], 2)
            self.assertEqual(self.wm._quality.call_count, 3)
            self.assertEqual([c.kwargs['num_return_sequences']
                              for c in self.draft.generate.call_args_list], [2, 2, 1])
            self.assertEqual(result['generated_token_ids'], [7] if watermarked else [6])
            self.assertEqual(block['selection_changed'], watermarked)
            self.assertAlmostEqual(block['quality_loss'], .1 if watermarked else 0)
            self.assertEqual(block['watermark_gain'], 1 if watermarked else 0)

    def test_adaptive_stops_early_and_duplicate_budget(self):
        self.wm.options = BranchOptions(candidates=2, max_candidates=6,
                                       block_size=1, max_new_tokens=1)
        self.wm._quality = Mock(return_value=-1.)
        self.draft.generate = Mock(return_value=torch.tensor([[2, 3, 4, 5], [2, 3, 4, 6]]))
        result = self.wm.generate('prompt')
        self.assertEqual(self.draft.generate.call_count, 1)
        self.assertEqual(result['trace'][0]['sampling_stop_reason'], 'eligible_target_met')
        self.draft.generate = Mock(return_value=torch.tensor([[2, 3, 4, 1], [2, 3, 4, 1]]))
        self.wm._quality.reset_mock()
        result = self.wm.generate('prompt')
        self.assertEqual(self.draft.generate.call_count, 3)
        self.assertEqual(self.wm._quality.call_count, 1)
        self.assertEqual(result['trace'][0]['sampling_stop_reason'], 'candidate_budget')
        self.assertEqual(result['stop_reason'], 'eos')
        self.assertEqual(result['diagnostics']['duplicate_candidates'], 5)

    def test_comparison_cli_offline(self):
        from examples.branch_kgw import main
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'compare.json')
            argv = ['branch_kgw', '--compare', '--seeds', '42', '43',
                    '--target-device', 'cpu', '--draft-device', 'cpu',
                    '--dtype', 'float32', '--candidates', '2', '--max-candidates', '3',
                    '--gamma', '0.25', '--detection-lengths', '1', '4',
                    '--max-new-tokens', '1', '--output', output]
            with patch('sys.argv', argv), patch(
                    'examples.branch_kgw.AutoTokenizer.from_pretrained', return_value=self.tokenizer), patch(
                    'examples.branch_kgw.AutoModelForCausalLM.from_pretrained',
                    side_effect=[self.target, self.draft]) as loader, contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(loader.call_count, 2)
            summary = json.loads((Path(directory) / 'compare.summary.json').read_text())
            self.assertEqual([(r['seed'], r['mode']) for r in summary],
                             [(42, 'watermarked'), (42, 'quality-only'),
                              (43, 'watermarked'), (43, 'quality-only')])
            for row in summary:
                saved = json.loads(Path(row['output']).read_text())
                self.assertEqual(saved['run']['mode'], row['mode'])
                self.assertEqual(saved['run']['seed'], row['seed'])
                self.assertIn('diagnostics', saved)
                self.assertEqual(saved['prompt_token_ids'], [2, 8, 4])
                self.assertEqual(saved['run']['kgw']['gamma'], .25)
                scored = saved['detection_with_prompt']
                self.assertAlmostEqual(scored['excess_green_tokens'], scored['green_tokens'] - .25)
                self.assertEqual(saved['detection_fixed_lengths']['with_prompt']['1']['num_tokens_scored'], 1)
                self.assertFalse(saved['detection_fixed_lengths']['with_prompt']['4']['available'])
                self.assertFalse(saved['detection_without_prompt']['threshold_is_calibrated'])

    def test_invalid_options(self):
        for kwargs in ({'epsilon': -1}, {'epsilon': float('nan')}, {'block_size': 0},
                       {'temperature': 0}, {'top_p': 2}, {'candidates': 1.5},
                       {'max_candidates': 3}, {'max_candidates': True},
                       {'min_unique_eligible': 0}, {'min_unique_eligible': 1.5}):
            with self.assertRaises(ValueError):
                BranchOptions(**kwargs)
        different = TinyTokenizer()
        different.get_vocab = lambda: {'different': 0}
        with self.assertRaises(ValueError):
            BranchKGW(self.wm.config, self.config, self.draft, different)


if __name__ == '__main__':
    unittest.main()
