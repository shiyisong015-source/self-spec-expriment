#!/usr/bin/env bash
# Official PyPI vLLM + LMCache environment (no source checkouts or patches).
# Usage: source scripts/activate_clean_lmcache.sh

_self_spec_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_self_spec_conda_base="$(conda info --base)"
_clean_env="${_self_spec_root}/.conda/envs/vllm-lmcache-official"

if [[ ! -d "${_clean_env}" ]]; then
  echo "Missing environment: ${_clean_env}" >&2
  return 1
fi

# shellcheck disable=SC1091
source "${_self_spec_conda_base}/etc/profile.d/conda.sh"
conda activate "${_clean_env}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export HF_HOME="${_self_spec_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export VLLM_CACHE_ROOT="${_self_spec_root}/.cache/vllm-lmcache-official"
export FLASHINFER_WORKSPACE_BASE="${_self_spec_root}/.cache/flashinfer-official"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${_clean_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${HF_DATASETS_CACHE}" "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}"

unset _clean_env
unset _self_spec_conda_base
unset _self_spec_root
