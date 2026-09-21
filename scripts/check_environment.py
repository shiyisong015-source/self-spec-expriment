#!/usr/bin/env python3
"""Check the base environment needed by the self-spec/HBM experiments."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def command_version(command: str, *args: str) -> Check:
    path = shutil.which(command)
    if path is None:
        return Check(command, False, "not found on PATH")
    result = subprocess.run(
        [path, *args], capture_output=True, text=True, check=False
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0] if output else path
    return Check(command, result.returncode == 0, detail)


def main() -> int:
    checks = [
        Check(
            "python",
            sys.version_info[:2] == (3, 12),
            f"{sys.version.split()[0]} ({sys.executable})",
        ),
        command_version("cmake", "--version"),
        command_version("ninja", "--version"),
        command_version("git", "--version"),
        command_version("nvcc", "--version"),
        command_version("nsys", "--version"),
    ]

    for module in (
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "yaml",
        "psutil",
        "datasets",
        "pynvml",
    ):
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "available")
            checks.append(Check(module, True, str(version)))
        except Exception as exc:  # report broken binary imports as well
            checks.append(Check(module, False, str(exc)))

    gpu = command_version(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    )
    # A driver is required for running experiments, but not for creating the
    # CPU-side development environment.
    gpu.required = False
    checks.append(gpu)

    width = max(len(check.name) for check in checks)
    for check in checks:
        status = "OK" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"[{status:4}] {check.name:<{width}}  {check.detail}")

    failures = [check for check in checks if check.required and not check.ok]
    if failures:
        print(f"\nBase environment check failed: {len(failures)} required item(s).")
        return 1
    if not gpu.ok:
        print("\nBase environment is ready; GPU experiments require a working NVIDIA driver.")
    else:
        print("\nBase environment and NVIDIA driver are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
