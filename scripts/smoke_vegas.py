#!/usr/bin/env python3
"""Run a small end-to-end Vegas generation on one GPU."""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speculative_config = {
        "method": "sparse_attn",
        "num_speculative_tokens": 2,
        "sparse_attn_algorithm": "vegas",
        "sparse_attn_ratio": 0.07,
        "sparse_attn_min_tokens": 64,
    }
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        block_size=16,
        gpu_memory_utilization=0.60,
        enforce_eager=True,
        speculative_config=speculative_config,
        seed=42,
    )
    prompt = (
        "Explain in one sentence why overlapping KV-cache restoration with "
        "draft computation can improve long-context inference throughput."
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    generated = outputs[0].outputs[0].text
    print(f"prompt: {prompt}")
    print(f"output: {generated!r}")
    print("Vegas end-to-end generation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
