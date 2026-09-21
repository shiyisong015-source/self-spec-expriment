#!/usr/bin/env python3
"""Offline end-to-end SparseSpec smoke test using a local checkpoint."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from serve.request.request import Request
from serve.run import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--result-dir", default="result/sparsespec-a100-smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No local Hugging Face checkpoint at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    messages = [{"role": "user", "content": "用一句话解释什么是推测解码。"}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    request = Request(token_ids=token_ids, desired_output_length=args.max_new_tokens)

    outputs = run_inference(
        reqs=[request],
        model_name=str(model_path),
        max_batch_size=1,
        tp_ranks=1,
        mem_fraction=0.80,
        cuda_graph=False,
        async_cpu=False,
        dtype="bfloat16",
        kv_cache="pillar",
        admit_policy="offloading",
        batch_policy="greedy",
        chunk_size=256,
        clip_length=len(token_ids) + args.max_new_tokens,
        spec_stride=4,
        budget_ratio=0.05,
        num_min_budget=16,
        temperature=0.0,
        result_dir=args.result_dir,
        dataset_name="local-smoke",
    )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
