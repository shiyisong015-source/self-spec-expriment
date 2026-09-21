#!/usr/bin/env python3
"""Smoke-test the locally built Vegas/vLLM CUDA stack."""

from __future__ import annotations

import inspect

import torch
import vllm
import vllm._C  # noqa: F401
import vllm._moe_C  # noqa: F401
import vllm.vllm_flash_attn._vllm_fa2_C  # noqa: F401
from vllm.v1.spec_decode.sparse_attn import SparseAttnProposer
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def main() -> int:
    print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"vLLM/Vegas: {vllm.__version__} ({vllm.__file__})")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU")

    device = torch.cuda.current_device()
    print(
        "GPU:",
        torch.cuda.get_device_name(device),
        f"SM {torch.cuda.get_device_capability(device)[0]}."
        f"{torch.cuda.get_device_capability(device)[1]}",
    )

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        raise RuntimeError("CUDA smoke-test produced non-finite values")

    signature = inspect.signature(flash_attn_varlen_func)
    if "scores" not in signature.parameters:
        raise RuntimeError("Companion FlashAttention build lacks the Vegas scores API")

    print(f"Sparse proposer: {SparseAttnProposer.__name__}")
    print("CUDA matmul: OK")
    print("vLLM CUDA extensions: OK")
    print("Vegas FlashAttention scores API: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
