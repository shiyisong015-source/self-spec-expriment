#!/usr/bin/env python3
"""Run the unmodified vLLM self_specs baseline on prepared JSONL prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# These must be set before importing vLLM.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--sink-size", type=int, default=32)
    parser.add_argument("--recent-ratio", type=float, default=0.05)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) == limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def metric_snapshot(llm: LLM) -> list[dict[str, Any]]:
    return [
        {
            "name": metric.name,
            "value": metric.value,
            "labels": getattr(metric, "labels", {}),
        }
        for metric in llm.get_metrics()
        if "self_spec" in metric.name
    ]


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input), args.limit)
    prompts = [record["prompt"] for record in records]

    started = time.perf_counter()
    llm = LLM(
        model=str(Path(args.model).resolve()),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=40_960,
        max_num_seqs=16,
        max_num_batched_tokens=2_048,
        block_size=1,
        sink_size=args.sink_size,
        recent_ratio=args.recent_ratio,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        cuda_graph_sizes=[1, 2, 4],
        disable_log_stats=False,
        speculative_config={
            "method": "self_specs",
            "model": None,
            "num_speculative_tokens": args.gamma,
        },
    )
    initialized = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
            seed=args.seed,
        ),
        use_tqdm=True,
    )
    finished = time.perf_counter()

    generations = []
    for record, output in zip(records, outputs):
        actual_prompt_tokens = len(output.prompt_token_ids)
        if actual_prompt_tokens != record["prompt_tokens"]:
            raise RuntimeError(
                f"Prompt-token mismatch for {record['id']}: prepared="
                f"{record['prompt_tokens']} runtime={actual_prompt_tokens}"
            )
        generations.append(
            {
                "id": record["id"],
                "source_dataset": record["source_dataset"],
                "source_task": record["source_task"],
                "prompt_tokens": actual_prompt_tokens,
                "output_tokens": len(output.outputs[0].token_ids),
                "generated_text": output.outputs[0].text,
                "reference_answer": record["reference_answer"],
            }
        )

    result = {
        "command": " ".join(sys.argv),
        "model": str(Path(args.model).resolve()),
        "tensor_parallel_size": 1,
        "gpu_type": "NVIDIA A100-PCIE-40GB",
        "batch_size": len(records),
        "gamma": args.gamma,
        "sink_size_blocks": args.sink_size,
        "recent_ratio": args.recent_ratio,
        "block_size": 1,
        "temperature": 0.0,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "engine_init_seconds": initialized - started,
        "generation_seconds": finished - initialized,
        "generations": generations,
        "self_spec_metrics": metric_snapshot(llm),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"BASELINE_RESULT={json.dumps(result, ensure_ascii=False)}")
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
