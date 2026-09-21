#!/usr/bin/env python3
"""Run a small end-to-end sparse-self-spec + LMCache smoke test."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-4B")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure LMCache before importing vLLM's connector integration.
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")
    os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
    os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "2.0")

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    speculative_config = {
        "method": "sparse_attn",
        "num_speculative_tokens": args.num_speculative_tokens,
        "sparse_attn_algorithm": "streamingllm",
        "sparse_attn_ratio": 0.05,
        "sparse_attn_min_tokens": 256,
    }
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
    )

    shared_prefix = (
        "Self-speculative decoding uses a sparse draft pass and a full "
        "verification pass. LMCache stores reusable KV states. " * 80
    )
    prompts = [
        shared_prefix + "\nSummarize the two components in one sentence.",
        shared_prefix + "\nState one reason to benchmark them together.",
    ]

    try:
        params = SamplingParams(temperature=0, max_tokens=8)
        for index, prompt in enumerate(prompts, start=1):
            output = llm.generate([prompt], params)[0]
            print(f"request={index} output={output.outputs[0].text!r}")
    finally:
        del llm

    print("Sparse self-speculation and LMCache smoke test passed.")


if __name__ == "__main__":
    main()
