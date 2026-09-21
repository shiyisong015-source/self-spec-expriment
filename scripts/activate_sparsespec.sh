#!/usr/bin/env bash
# Source this file before running SparseSpec on this A100 host:
#   source scripts/activate_sparsespec.sh

_self_spec_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_self_spec_conda_base="$(conda info --base)"

# shellcheck disable=SC1091
source "${_self_spec_conda_base}/etc/profile.d/conda.sh"
conda activate "${_self_spec_root}/.conda/envs/sparsespec-a100"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_WORKSPACE_BASE="${_self_spec_root}/.cache/flashinfer-sparsespec-a100"
export HF_HOME="${_self_spec_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${_self_spec_root}/.cache"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${_self_spec_root}/external/SparseSpec${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${_self_spec_root}/.conda/envs/sparsespec-a100/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${HF_DATASETS_CACHE}"

unset _self_spec_conda_base
unset _self_spec_root
