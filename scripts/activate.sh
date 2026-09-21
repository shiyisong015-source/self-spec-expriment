#!/usr/bin/env bash
# Source this file from the repository root or any other directory:
#   source scripts/activate.sh

_self_spec_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_self_spec_conda_base="$(conda info --base)"

# shellcheck disable=SC1091
source "${_self_spec_conda_base}/etc/profile.d/conda.sh"
conda activate "${_self_spec_root}/.conda/envs/self-spec-hbm"

export HF_HOME="${_self_spec_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="${_self_spec_root}/.cache/matplotlib"
export XDG_CACHE_HOME="${_self_spec_root}/.cache"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"

unset _self_spec_conda_base
unset _self_spec_root
