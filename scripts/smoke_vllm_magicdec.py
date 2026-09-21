#!/usr/bin/env python3
"""Offline smoke test for vLLM's MagicDec-style self-speculation."""

import argparse
import json
import os
import time
from pathlib import Path

# The SparseSpec vLLM fork requires the V1 engine and FlashInfer for self_specs.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--output", default="result/vllm-magicdec-a100-smoke.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model).resolve())
    if not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError(f"No local Hugging Face checkpoint at {model_path}")

    started = time.perf_counter()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        # self_specs warmup needs at least num_speculative_tokens + 1 slots.
        max_num_seqs=16,
        max_num_batched_tokens=512,
        block_size=1,
        sink_size=32,
        recent_ratio=0.05,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        cuda_graph_sizes=[1, 2, 4],
        disable_log_stats=False,
        speculative_config={
            "method": "self_specs",
            "model": None,
            "num_speculative_tokens": 8,
        },
    )
    initialized = time.perf_counter()

    prompt = "请用两句话解释推测解码为什么能够加速大语言模型推理。"
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    finished = time.perf_counter()

    self_spec_metrics = []
    for metric in llm.get_metrics():
        if "self_spec" in metric.name:
            self_spec_metrics.append(
                {
                    "name": metric.name,
                    "value": metric.value,
                    "labels": getattr(metric, "labels", {}),
                }
            )

    result = {
        "model": model_path,
        "method": "self_specs",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "engine_init_seconds": initialized - started,
        "generation_seconds": finished - initialized,
        "prompt_tokens": len(outputs[0].prompt_token_ids),
        "output_tokens": len(outputs[0].outputs[0].token_ids),
        "generated_text": outputs[0].outputs[0].text,
        "self_spec_metrics": self_spec_metrics,
    }
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_json + "\n", encoding="utf-8")
    print("MAGICDEC_SMOKE_RESULT=" + json.dumps(result, ensure_ascii=False))
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
