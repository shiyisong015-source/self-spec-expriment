#!/usr/bin/env bash
# Modern vLLM sparse-attention self-spec development environment.
# Usage: source scripts/activate_modern_selfspec.sh

_self_spec_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_self_spec_conda_base="$(conda info --base)"
_modern_env="${_self_spec_root}/.conda/envs/modern-selfspec-lmcache"

if [[ ! -d "${_modern_env}" ]]; then
  echo "Missing environment: ${_modern_env}" >&2
  return 1
fi

# shellcheck disable=SC1091
source "${_self_spec_conda_base}/etc/profile.d/conda.sh"
conda activate "${_modern_env}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export HF_HOME="${_self_spec_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export VLLM_CACHE_ROOT="${_self_spec_root}/.cache/vllm-modern"
export FLASHINFER_WORKSPACE_BASE="${_self_spec_root}/.cache/flashinfer-modern"
# LMCache probes CUDA during import. Spawn avoids reinitializing CUDA in a
# forked vLLM engine process when using the offline Python API.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${_modern_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${HF_DATASETS_CACHE}" "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}"

unset _modern_env
unset _self_spec_conda_base
unset _self_spec_root
