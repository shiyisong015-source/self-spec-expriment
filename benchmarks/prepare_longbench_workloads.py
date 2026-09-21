#!/usr/bin/env python3
"""Prepare deterministic, naturally sized LongBench workloads for Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer


QASPER_TEMPLATE = """You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.

Article: {context}

Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.

Question: {input}

Answer:"""

LBV2_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: \"The correct answer is (insert answer here)\"."""

BUCKETS = (
    {
        "name": "LB_16K_18K",
        "output": "lb_16k_18k.jsonl",
        "minimum": 16_384,
        "maximum": 18_432,
        "repo": "THUDM/LongBench",
        "config": "qasper",
        "split": "test",
    },
    {
        "name": "LBV2_30K_40K",
        "output": "lbv2_30k_40k.jsonl",
        "minimum": 30_720,
        "maximum": 40_960,
        "repo": "THUDM/LongBench-v2",
        "config": None,
        "split": "train",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--output-dir", default="data/workloads")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--reserve-output-tokens", type=int, default=96)
    return parser.parse_args()


def build_user_prompt(source_task: str, row: dict[str, Any]) -> str:
    if source_task == "qasper":
        return QASPER_TEMPLATE.format(
            context=row["context"], input=row["input"]
        )
    if source_task == "longbench-v2":
        return LBV2_TEMPLATE.format(**row)
    raise ValueError(f"Unsupported source task: {source_task}")


def build_final_prompt(tokenizer: Any, user_prompt: str) -> str:
    """Render exactly what the offline vLLM baseline passes to generate()."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def stable_rank(seed: int, dataset: str, example_id: str) -> str:
    key = f"{seed}:{dataset}:{example_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(name: str, records: list[dict[str, Any]]) -> None:
    lengths = sorted(record["prompt_tokens"] for record in records)
    if not lengths:
        print(f"{name}: count=0 (no naturally matching examples)")
        return
    print(
        f"{name}: count={len(lengths)} min={lengths[0]} "
        f"mean={statistics.fmean(lengths):.1f} "
        f"p50={percentile(lengths, 0.50):.1f} "
        f"p90={percentile(lengths, 0.90):.1f} max={lengths[-1]}"
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_limit = int(config.max_position_embeddings)
    safe_prompt_limit = model_limit - args.reserve_output_tokens
    output_dir = Path(args.output_dir)

    for bucket in BUCKETS:
        dataset = load_dataset(
            bucket["repo"],
            bucket["config"],
            split=bucket["split"],
            trust_remote_code=bucket["repo"] == "THUDM/LongBench",
        )
        source_task = "qasper" if bucket["config"] else "longbench-v2"
        matches = []

        for position, row in enumerate(dataset):
            example_id = str(row.get("_id", position))
            user_prompt = build_user_prompt(source_task, row)
            final_prompt = build_final_prompt(tokenizer, user_prompt)
            prompt_tokens = len(
                tokenizer.encode(final_prompt, add_special_tokens=False)
            )
            effective_maximum = min(bucket["maximum"], safe_prompt_limit)
            if not bucket["minimum"] <= prompt_tokens <= effective_maximum:
                continue

            reference_answer = (
                row["answers"] if source_task == "qasper" else row["answer"]
            )
            matches.append(
                {
                    "id": example_id,
                    "source_dataset": bucket["repo"],
                    "source_task": source_task,
                    "prompt_tokens": prompt_tokens,
                    "prompt": final_prompt,
                    "reference_answer": reference_answer,
                    "selection_seed": args.seed,
                    "original_position": position,
                }
            )

        matches.sort(
            key=lambda record: stable_rank(
                args.seed, record["source_dataset"], record["id"]
            )
        )
        selected = matches[: args.target_count]
        write_jsonl(output_dir / bucket["output"], selected)
        summarize(bucket["name"], selected)
        print(
            f"  natural_matches={len(matches)} requested={args.target_count} "
            f"effective_range=[{bucket['minimum']}, "
            f"{min(bucket['maximum'], safe_prompt_limit)}] "
            f"output={output_dir / bucket['output']}"
        )


if __name__ == "__main__":
    main()
