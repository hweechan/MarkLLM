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
    parser.add_argument('--target', default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--draft', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--target-device', default='cuda:0')
    parser.add_argument('--draft-device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--prompt', default='Explain why the sky appears blue in clear, simple terms.')
    parser.add_argument('--candidates', type=int, default=4)
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
    options = BranchOptions(args.candidates, args.block_size, args.epsilon,
                            args.max_new_tokens, args.temperature, args.top_p)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft)
    dtype = getattr(torch, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(args.target_device)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(args.draft_device)
    config = TransformersConfig(target, tokenizer, device=args.target_device)
    kgw_config = KGWConfig(str(Path(__file__).resolve().parents[1] / 'config/KGW.json'),
                           config, hash_key=args.hash_key)
    wm = BranchKGW(kgw_config, config, draft, draft_tokenizer, options)
    devices = {torch.device(args.target_device), torch.device(args.draft_device)}
    for device in devices:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = wm.generate(args.prompt, watermarked=args.mode == 'watermarked')
    for device in devices:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result['run'] = {**vars(args), 'output': str(args.output),
                     'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                     'target_commit': getattr(target.config, '_commit_hash', None),
                     'draft_commit': getattr(draft.config, '_commit_hash', None),
                     'kgw': kgw_config.config_dict,
                     'elapsed_seconds': elapsed,
                     'tokens_per_second': len(result['generated_token_ids']) / elapsed,
                     'peak_cuda_allocated_bytes': {str(d): torch.cuda.max_memory_allocated(d)
                                                   for d in devices if d.type == 'cuda'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result['continuation'])
    print(json.dumps(result['detection_without_prompt'], indent=2))
    print(f'Saved {args.output}; {elapsed:.2f}s')


if __name__ == '__main__':
    main()
