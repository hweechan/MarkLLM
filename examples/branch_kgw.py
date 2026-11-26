"""Run from repository root: python -m examples.branch_kgw --help."""
import argparse
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from utils.transformers_config import TransformersConfig
from watermark.branch_kgw import BranchKGW, BranchOptions
from watermark.kgw.kgw import KGWConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--draft', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--target-device', default='cuda:0')
    parser.add_argument('--draft-device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--prompt', default='Explain why the sky appears blue in clear, simple terms.')
    parser.add_argument('--prompt-format', choices=['chat', 'raw'], default='chat')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--detection-lengths', type=int, nargs='+', default=[64, 128],
                        help='Exact continuation lengths for detection; short outputs are marked unavailable')
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=None,
                        help='Per-block sampling budget; default: candidates (no retries)')
    parser.add_argument('--min-unique-eligible', type=int, default=2)
    parser.add_argument('--compare', action='store_true',
                        help='Run both modes with the same options and seeds; load models once')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--block-size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hash-key', type=int, default=15485863)
    parser.add_argument('--mode', choices=['watermarked', 'quality-only'], default='watermarked')
    parser.add_argument('--output', type=Path, default=Path('outputs/branch_kgw.json'))
    args = parser.parse_args()
    if not 0 < args.gamma < 1:
        parser.error('--gamma must be in (0, 1)')
    if any(length < 1 for length in args.detection_lengths):
        parser.error('--detection-lengths must be positive')
    options = BranchOptions(args.candidates, args.block_size, args.epsilon,
                            args.max_new_tokens, args.temperature, args.top_p,
                            args.max_candidates, args.min_unique_eligible)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft)
    if args.prompt_format == 'chat':
        if not getattr(tokenizer, 'chat_template', None):
            parser.error('Tokenizer has no chat template; use --prompt-format raw for a base model')
        prompt_ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': args.prompt}], tokenize=True,
            add_generation_prompt=True)
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    dtype = getattr(torch, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(args.target_device)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(args.draft_device)
    config = TransformersConfig(target, tokenizer, device=args.target_device)
    kgw_config = KGWConfig(str(Path(__file__).resolve().parents[1] / 'config/KGW.json'),
                           config, hash_key=args.hash_key, gamma=args.gamma)
    wm = BranchKGW(kgw_config, config, draft, draft_tokenizer, options)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    modes = ['watermarked', 'quality-only'] if args.compare else [args.mode]
    summaries = []
    for seed in seeds:
        for mode in modes:
            set_seed(seed)
            output = args.output
            if args.compare or args.seeds is not None:
                output = output.with_name(f'{output.stem}.{mode}.seed{seed}{output.suffix}')
            devices = {torch.device(args.target_device), torch.device(args.draft_device)}
            for device in devices:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = wm.generate(prompt_token_ids=prompt_ids, watermarked=mode == 'watermarked')
            result['detection_fixed_lengths'] = {
                'with_prompt': wm.detect_fixed_lengths(result['generated_token_ids'],
                                                       args.detection_lengths, prompt_ids),
                'without_prompt': wm.detect_fixed_lengths(result['generated_token_ids'],
                                                          args.detection_lengths)}
            for device in devices:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            result['run'] = {**vars(args), 'seed': seed, 'mode': mode, 'output': str(output),
                             'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                             'target_commit': getattr(target.config, '_commit_hash', None),
                             'draft_commit': getattr(draft.config, '_commit_hash', None),
                             'kgw': kgw_config.config_dict,
                             'elapsed_seconds': elapsed,
                             'tokens_per_second': len(result['generated_token_ids']) / elapsed,
                             'peak_cuda_allocated_bytes': {str(d): torch.cuda.max_memory_allocated(d)
                                                           for d in devices if d.type == 'cuda'}}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(result['continuation'])
            print(f"Mode: {mode}; seed: {seed}; stop: {result['stop_reason']}; "
                  f"tokens: {len(result['generated_token_ids'])}")
            print('Selection diagnostics (gains are local, not a separate-run comparison):')
            print(json.dumps(result['diagnostics'], indent=2))
            detection = result['detection_without_prompt']
            print('Detection without prompt:')
            print(json.dumps(detection, indent=2))
            print(f"Threshold: {kgw_config.z_threshold} (UNCALIBRATED); "
                  f"verdict: {'above threshold' if detection['is_watermarked'] else 'below threshold'}")
            print(f'Saved {output}; {elapsed:.2f}s')

            summaries.append({'seed': seed, 'mode': mode, 'output': str(output),
                              'generated_tokens': len(result['generated_token_ids']),
                              'detection': detection, 'diagnostics': result['diagnostics'],
                              'detection_fixed_lengths': result['detection_fixed_lengths'],
                              'elapsed_seconds': elapsed,
                              'tokens_per_second': result['run']['tokens_per_second']})
    if args.compare or args.seeds is not None:
        summary_path = args.output.with_name(f'{args.output.stem}.summary.json')
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Saved comparison summary: {summary_path}')


if __name__ == '__main__':
    main()
