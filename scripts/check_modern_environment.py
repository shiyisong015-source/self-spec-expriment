#!/usr/bin/env python3
"""Validate the pinned modern vLLM sparse-self-spec/LMCache environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VLLM_SOURCE = (PROJECT_ROOT / "external/vegas-vllm-pr1/vllm").resolve()
EXPECTED_VERSIONS = {
    "torch": "2.11.0+cu130",
    "flashinfer-python": "0.6.12",
    "lmcache": "0.5.5",
    "cupy-cuda13x": "14.0.1",
}


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL':4}] {name:<12} {detail}")
    return ok


def main() -> int:
    passed = []
    passed.append(report(
        "python", sys.version_info[:2] == (3, 12),
        f"{sys.version.split()[0]} ({sys.executable})"))

    distributions = {
        "torch": "torch",
        "vllm": "vllm",
        "flashinfer": "flashinfer-python",
        "lmcache": "lmcache",
    }
    for module_name, distribution_name in distributions.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution_name)
            expected = EXPECTED_VERSIONS.get(distribution_name)
            version_ok = (
                "g79085a31a" in version
                if distribution_name == "vllm"
                else expected is None or version == expected
            )
            passed.append(report(module_name, version_ok, str(version)))
        except Exception as exc:
            passed.append(report(module_name, False, repr(exc)))

    try:
        cupy_version = importlib.metadata.version("cupy-cuda13x")
        passed.append(report(
            "cupy", cupy_version == EXPECTED_VERSIONS["cupy-cuda13x"],
            cupy_version))
    except importlib.metadata.PackageNotFoundError as exc:
        passed.append(report("cupy", False, repr(exc)))

    try:
        import torch
        cuda_detail = (f"available={torch.cuda.is_available()} "
                       f"torch_cuda={torch.version.cuda}")
        passed.append(report("cuda", torch.cuda.is_available(), cuda_detail))
    except Exception as exc:
        passed.append(report("cuda", False, repr(exc)))

    try:
        import vllm
        source = Path(vllm.__file__).resolve().parent
        passed.append(report(
            "editable", source == EXPECTED_VLLM_SOURCE,
            f"{source} (expected {EXPECTED_VLLM_SOURCE})"))
        from vllm.config.speculative import SpeculativeConfig
        field_names = {field.name for field in fields(SpeculativeConfig)}
        sparse_fields = {
            "sparse_attn_algorithm",
            "sparse_attn_ratio",
            "sparse_attn_min_tokens",
        }
        passed.append(report(
            "sparse_attn", sparse_fields <= field_names,
            f"fields={sorted(sparse_fields & field_names)}"))
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )
        passed.append(report(
            "connector", LMCacheConnectorV1 is not None,
            "LMCacheConnectorV1 importable"))
    except Exception as exc:
        passed.append(report("vllm source", False, repr(exc)))

    print()
    if all(passed):
        print("Modern sparse-self-spec and LMCache environment is ready.")
        return 0
    print(f"Environment check failed: {passed.count(False)} item(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
