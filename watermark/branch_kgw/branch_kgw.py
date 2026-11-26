"""Experimental candidate-path KGW selection; not exact speculative sampling."""
from dataclasses import dataclass, asdict
import math

import torch

from watermark.kgw.kgw import KGW


@dataclass(frozen=True)
class BranchOptions:
    candidates: int = 4
    block_size: int = 4
    epsilon: float = 0.2
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    max_candidates: int | None = None
    min_unique_eligible: int = 2

    def __post_init__(self):
        for name in ('candidates', 'block_size', 'max_new_tokens'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        budget = self.max_candidates if self.max_candidates is not None else self.candidates
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < self.candidates:
            raise ValueError('max_candidates must be an integer >= candidates')
        if (not isinstance(self.min_unique_eligible, int)
                or isinstance(self.min_unique_eligible, bool) or self.min_unique_eligible < 1):
            raise ValueError('min_unique_eligible must be a positive integer')
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('epsilon must be finite and nonnegative')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not 0 < self.top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')


def select_candidate(qualities, watermark_scores, epsilon, watermarked=True):
    """Relative quality gate, then WM score; ties prefer higher quality."""
    if not qualities or len(qualities) != len(watermark_scores):
        raise ValueError('Expected equally sized nonempty score lists')
    if not all(math.isfinite(x) for x in qualities + watermark_scores):
        raise ValueError('Nonfinite candidate score')
    best = max(qualities)
    valid = [i for i, q in enumerate(qualities) if q >= best - epsilon]
    selected = max(valid, key=lambda i: (watermark_scores[i], qualities[i])) if watermarked else max(valid, key=lambda i: qualities[i])
    return selected, valid


class BranchKGW(KGW):
    """Single-prompt, same-tokenizer, decoder-only research implementation.

    Target scoring is sequential across candidates to bound peak memory.
    No cache is retained across blocks. gen_kwargs are used only by inherited
    target-only generation; branch sampling is controlled by BranchOptions.
    """

    def __init__(self, algorithm_config, transformers_config, draft_model,
                 draft_tokenizer, options=None):
        super().__init__(algorithm_config, transformers_config)
        self.options = options or BranchOptions()
        target_tokenizer = self.config.generation_tokenizer
        if (target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
                or target_tokenizer.all_special_ids != draft_tokenizer.all_special_ids):
            raise ValueError('Draft and target must share vocabulary IDs and special tokens')
        if self.config.window_scheme != 'left':
            raise ValueError('Initial BranchKGW implementation supports left window_scheme only')
        if not 0 < self.config.gamma < 1:
            raise ValueError('gamma must be in (0, 1)')
        if self.config.prefix_length < 1:
            raise ValueError('prefix_length must be positive')
        for model in (draft_model, self.config.generation_model):
            if model.config.is_encoder_decoder:
                raise ValueError('Only decoder-only models are supported')
            if model.get_output_embeddings().weight.shape[0] < self.config.vocab_size:
                raise ValueError('Model output vocabulary is smaller than tokenizer vocabulary')
            model.eval()
        self.draft_model = draft_model
        self.last_result = None

    @staticmethod
    def _device(model):
        return model.get_input_embeddings().weight.device

    def _green_flags(self, prefix, continuation):
        context = list(prefix)
        flags = []
        for token in continuation:
            if len(context) < self.config.prefix_length:
                flags.append(-1)
            else:
                ids = torch.tensor(context, dtype=torch.long, device=self.config.device)
                flags.append(int(token in self.utils.get_greenlist_ids(ids)))
            context.append(token)
        return flags

    def detect_token_ids(self, continuation, prefix=()):
        """Score only continuation; prefix supplies context and is never counted.

        With no prefix the first prefix_length tokens are skipped. The threshold
        is inherited from KGW and is provisional until empirically calibrated.
        """
        flags = self._green_flags(prefix, continuation)
        count = sum(f >= 0 for f in flags)
        green = sum(f == 1 for f in flags)
        score = self.utils._compute_z_score(green, count) if count else None
        return {'score': score, 'is_watermarked': score is not None and score > self.config.z_threshold,
                'num_tokens_scored': count, 'green_tokens': green,
                'green_fraction': green / count if count else None,
                'excess_green_tokens': green - self.config.gamma * count,
                'threshold_is_calibrated': False}

    def detect_fixed_lengths(self, continuation, lengths, prefix=()):
        """Score exact-length prefixes; never substitute a shorter EOS output."""
        results = {}
        for length in lengths:
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise ValueError('Detection lengths must be positive integers')
            results[str(length)] = (
                {'available': True, **self.detect_token_ids(continuation[:length], prefix)}
                if len(continuation) >= length else
                {'available': False, 'reason': 'output_too_short', 'score': None})
        return results

    def _quality(self, prefix, candidate):
        model = self.config.generation_model
        ids = torch.tensor([prefix + candidate], device=self._device(model))
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        # A token at position t is predicted by logits at t - 1.
        logits = logits[0, len(prefix) - 1:len(prefix) + len(candidate) - 1].float()
        labels = ids[0, len(prefix):].to(logits.device)
        return logits.log_softmax(-1).gather(-1, labels[:, None]).mean().item()

    @torch.inference_mode()
    def generate(self, prompt=None, watermarked=True, *, prompt_token_ids=None):
        tokenizer = self.config.generation_tokenizer
        if (prompt is None) == (prompt_token_ids is None):
            raise ValueError('Provide exactly one of prompt or prompt_token_ids')
        prefix = (tokenizer.encode(prompt, add_special_tokens=True)
                  if prompt_token_ids is None else list(prompt_token_ids))
        if any(not isinstance(t, int) or isinstance(t, bool) or
               not 0 <= t < self.config.vocab_size for t in prefix):
            raise ValueError('Prompt token IDs must be integers within the vocabulary')
        if not prefix:
            raise ValueError('Prompt must encode to at least one token')
        o = self.options
        for model in (self.draft_model, self.config.generation_model):
            limit = getattr(model.config, 'max_position_embeddings', None)
            if limit and len(prefix) + o.max_new_tokens > limit:
                raise ValueError(f'Prompt plus output budget exceeds model context limit {limit}')
        # Tokenizer controls termination consistently for both models.
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        if pad is None:
            raise ValueError('Tokenizer must define pad_token_id or eos_token_id')
        context = list(prefix)
        continuation, trace = [], []
        while len(continuation) < o.max_new_tokens:
            length = min(o.block_size, o.max_new_tokens - len(continuation))
            ids = torch.tensor([context], device=self._device(self.draft_model))
            budget = o.max_candidates if o.max_candidates is not None else o.candidates
            candidates, qualities, scores = [], [], []
            seen = set()
            sampled = rounds = 0
            while sampled < budget:
                proposals = self.draft_model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=length, min_new_tokens=0, do_sample=True,
                    temperature=o.temperature, top_p=o.top_p, top_k=0,
                    repetition_penalty=1.0, num_beams=1, num_return_sequences=min(o.candidates, budget - sampled),
                    eos_token_id=eos, pad_token_id=pad, use_cache=True,
                    return_dict_in_generate=False, output_scores=False,
                    bad_words_ids=[[v] for v in range(self.config.vocab_size,
                        self.draft_model.get_output_embeddings().weight.shape[0])] or None,
                )
                rounds += 1
                for row in proposals[:, len(context):].tolist():
                    sampled += 1
                    if eos is not None and eos in row:
                        row = row[:row.index(eos) + 1]
                    if not row:
                        raise RuntimeError('Draft produced an empty candidate')
                    key = tuple(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(row)
                    qualities.append(self._quality(context, row))
                    scores.append(sum(f - self.config.gamma
                                      for f in self._green_flags(context, row) if f >= 0))
                # Recompute the gate against the best of ALL candidates. Stopping
                # depends only on Q/diversity/budget, never W or generation mode.
                chosen, valid = select_candidate(qualities, scores, o.epsilon, watermarked)
                if len(valid) >= o.min_unique_eligible:
                    break
            best = max(range(len(qualities)), key=lambda i: qualities[i])
            gain = scores[chosen] - scores[best]
            reason = ('quality_only' if not watermarked else
                      'single_eligible' if len(valid) == 1 else
                      'watermark_gain' if chosen != best else
                      'watermark_tie' if len({scores[i] for i in valid}) == 1 else
                      'quality_best_also_watermark_best')
            trace.append({'offset': len(continuation), 'candidate_token_ids': candidates,
                          'quality_scores': qualities, 'watermark_scores': scores,
                          'eligible_indices': valid, 'selected_index': chosen,
                          'sampled_candidate_count': sampled, 'sampling_rounds': rounds,
                          'unique_candidate_count': len(candidates),
                          'duplicate_candidate_count': sampled - len(candidates),
                          'unique_eligible_count': len(valid), 'quality_best_index': best,
                          'eligible_watermark_tie': len({scores[i] for i in valid}) == 1,
                          'watermark_gain_available': max(scores[i] for i in valid) - scores[best],
                          'selection_changed': chosen != best,
                          'quality_loss': qualities[best] - qualities[chosen],
                          'watermark_gain': gain, 'selection_reason': reason,
                          'sampling_stop_reason': 'eligible_target_met' if
                          len(valid) >= o.min_unique_eligible else 'candidate_budget'})
            selected = candidates[chosen]
            continuation.extend(selected)
            context.extend(selected)
            if eos is not None and selected[-1] == eos:
                break
        result = {'text': tokenizer.decode(context, skip_special_tokens=True),
                  'continuation': tokenizer.decode(continuation, skip_special_tokens=True),
                  'prompt_token_ids': prefix, 'generated_token_ids': continuation,
                  'watermarked': watermarked, 'options': asdict(o), 'trace': trace,
                  'detection_with_prompt': self.detect_token_ids(continuation, prefix),
                  'detection_without_prompt': self.detect_token_ids(continuation)}
        result['stop_reason'] = 'eos' if continuation[-1] == eos else 'max_new_tokens'
        result['diagnostics'] = {
            'total_blocks': len(trace),
            'single_unique_eligible_blocks': sum(b['unique_eligible_count'] == 1 for b in trace),
            'selection_changed_blocks': sum(b['selection_changed'] for b in trace),
            'sampled_candidates': sum(b['sampled_candidate_count'] for b in trace),
            'unique_candidates': sum(b['unique_candidate_count'] for b in trace),
            'duplicate_candidates': sum(b['duplicate_candidate_count'] for b in trace),
            'local_watermark_gain_sum': sum(b['watermark_gain'] for b in trace),
            'mean_quality_loss': sum(b['quality_loss'] for b in trace) / len(trace),
            'selection_changed_block_rate': sum(b['selection_changed'] for b in trace) / len(trace),
            'duplicate_candidate_rate': sum(b['duplicate_candidate_count'] for b in trace) /
                                        sum(b['sampled_candidate_count'] for b in trace),
            'mean_unique_eligible': sum(b['unique_eligible_count'] for b in trace) / len(trace),
            'multiple_eligible_watermark_tie_blocks': sum(
                b['unique_eligible_count'] > 1 and b['eligible_watermark_tie'] for b in trace),
            'quality_best_also_watermark_best_blocks': sum(
                b['unique_eligible_count'] > 1 and not b['eligible_watermark_tie']
                and b['watermark_gain_available'] == 0 for b in trace),
            'positive_gain_available_blocks': sum(b['watermark_gain_available'] > 0 for b in trace),
        }
        self.last_result = result
        return result

    def generate_watermarked_text(self, prompt, *args, **kwargs):
        return self.generate(prompt, watermarked=True)['text']

    def generate_unwatermarked_text(self, prompt, *args, **kwargs):
        """Quality-only branch control, not ordinary target-only generation."""
        return self.generate(prompt, watermarked=False)['text']
