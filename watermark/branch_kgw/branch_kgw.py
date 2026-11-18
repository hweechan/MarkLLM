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

    def __post_init__(self):
        for name in ('candidates', 'block_size', 'max_new_tokens'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
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
                'threshold_is_calibrated': False}

    def _quality(self, prefix, candidate):
        model = self.config.generation_model
        ids = torch.tensor([prefix + candidate], device=self._device(model))
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        # A token at position t is predicted by logits at t - 1.
        logits = logits[0, len(prefix) - 1:len(prefix) + len(candidate) - 1].float()
        labels = ids[0, len(prefix):].to(logits.device)
        return logits.log_softmax(-1).gather(-1, labels[:, None]).mean().item()

    @torch.inference_mode()
    def generate(self, prompt, watermarked=True):
        tokenizer = self.config.generation_tokenizer
        prefix = tokenizer.encode(prompt, add_special_tokens=True)
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
            # Explicit settings avoid checkpoint-specific sampling defaults.
            proposals = self.draft_model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=length, min_new_tokens=0, do_sample=True,
                temperature=o.temperature, top_p=o.top_p, top_k=0,
                repetition_penalty=1.0, num_beams=1, num_return_sequences=o.candidates,
                eos_token_id=eos, pad_token_id=pad, use_cache=True,
                return_dict_in_generate=False, output_scores=False,
                bad_words_ids=[[v] for v in range(self.config.vocab_size,
                    self.draft_model.get_output_embeddings().weight.shape[0])] or None,
            )
            candidates = []
            for row in proposals[:, len(context):].tolist():
                if eos is not None and eos in row:
                    row = row[:row.index(eos) + 1]
                if not row:
                    raise RuntimeError('Draft produced an empty candidate')
                candidates.append(row)
            qualities = [self._quality(context, c) for c in candidates]
            scores = [sum(f - self.config.gamma for f in self._green_flags(context, c) if f >= 0)
                      for c in candidates]
            chosen, valid = select_candidate(qualities, scores, o.epsilon, watermarked)
            trace.append({'offset': len(continuation), 'candidate_token_ids': candidates,
                          'quality_scores': qualities, 'watermark_scores': scores,
                          'eligible_indices': valid, 'selected_index': chosen})
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
        self.last_result = result
        return result

    def generate_watermarked_text(self, prompt, *args, **kwargs):
        return self.generate(prompt, watermarked=True)['text']

    def generate_unwatermarked_text(self, prompt, *args, **kwargs):
        """Quality-only branch control, not ordinary target-only generation."""
        return self.generate(prompt, watermarked=False)['text']
