# Modern vLLM + LMCache migration stack

This branch keeps experiment assets in the outer `self-spec` repository while
scheduler, KV-residency, and model-runner changes live in a separately versioned
vLLM fork.

## Pinned source baseline

- Outer repository branch: `research/modern-vllm-lmcache-selfspec`
- vLLM source: `external/vegas-vllm-pr1`
- Upstream: `https://github.com/npz7yyk/vegas.git`
- Upstream branch: `vllm-pr1`
- Pinned feature commit: `79085a31ab9c2eb850ef354424552b9ec797f001`
- Parent vLLM commit: `ee0da84ab`
- Local vLLM development branch: `research/lmcache-residency`

The feature commit adds StreamingLLM sparse-attention self-speculation using
Python-only changes. The editable installation therefore reuses the native
extensions built for the parent vLLM commit. Scheduler/model-runner Python edits
only require restarting the engine; Triton changes JIT on first use. Changes to
`csrc/`, C++/CUDA extensions, or FlashAttention require a native rebuild.

## Repository layout

```text
self_spec/                         # experiment/control repository
├── benchmarks/                    # workloads and benchmark launchers
├── docs/                          # audits and reproduction notes
├── scripts/                       # environment helpers
└── external/vegas-vllm-pr1/       # independently versioned vLLM fork
```

The `external/` checkout is intentionally ignored by the outer repository. Its
commits must be pushed to a dedicated remote fork; committing the outer branch
does not save vLLM source changes.

## Environment

```bash
conda create -y -p "$PWD/.conda/envs/modern-selfspec-lmcache" \
  python=3.12 pip

# Install build dependencies in the target environment first. This avoids an
# additional multi-gigabyte isolated Torch build environment.
UV_LINK_MODE=copy uv pip install \
  --python "$PWD/.conda/envs/modern-selfspec-lmcache/bin/python" \
  --torch-backend=cu130 \
  'torch==2.11.0' 'setuptools>=77.0.3,<81.0.0' \
  'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' \
  'cmake>=3.26.1' ninja packaging wheel

# The sparse-attention feature commit changes Python only, so use native
# extensions from its exact upstream parent commit.
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
UV_LINK_MODE=copy uv pip install \
  --python "$PWD/.conda/envs/modern-selfspec-lmcache/bin/python" \
  --editable "$PWD/external/vegas-vllm-pr1" \
  --torch-backend=cu130 \
  --no-build-isolation

UV_LINK_MODE=copy uv pip install \
  --python "$PWD/.conda/envs/modern-selfspec-lmcache/bin/python" \
  'lmcache==0.5.5' 'cupy-cuda13x==14.0.1'

source scripts/activate_modern_selfspec.sh
python scripts/check_modern_environment.py
```

The validated package set is Torch 2.11.0+cu130, FlashInfer 0.6.12, LMCache
0.5.5, and CuPy 14.0.1. CuPy is intentionally below 14.1, matching vLLM's
`requirements/kv_connectors.txt` compatibility constraint.

The activation helper also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. LMCache
probes CUDA during import, and the offline vLLM Python API otherwise defaults
to `fork`, which cannot safely reinitialize CUDA in the engine subprocess. It
also fixes `PYTHONHASHSEED=0` so LMCache's builtin chunk hashes are stable
across the scheduler and worker processes.

For the first local CPU-offload experiment, use the in-process connector:

```bash
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0

vllm serve /data/models/Qwen3-8B \
  --speculative-config \
    '{"method":"sparse_attn","num_speculative_tokens":4,"sparse_attn_algorithm":"streamingllm","sparse_attn_ratio":0.05,"sparse_attn_min_tokens":256}' \
  --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

This deliberately uses vLLM's normal block size; unlike the older MagicDec
runtime, the modern sparse-attention path does not require `--block-size 1`.

Run the end-to-end local-model smoke test with:

```bash
source scripts/activate_modern_selfspec.sh
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_modern_stack.py \
  --model /data/models/Qwen3-4B
```

The validated smoke test stored 1,536 shared-prefix tokens (about 0.211 GiB)
on the first request and retrieved all 1,536 tokens from LMCache on the second
request. The sparse draft path also JIT-compiled `copy_blocks_kernel` and the
speculative rejection-sampling kernels, confirming both integrations executed.

## Known integration issue

LMCache 0.5.5 currently emits a `Double unpin occurred somewhere` warning for
each retrieved chunk with this development vLLM baseline. LMCache clamps the
pin count back to zero and the smoke test completes, so this is not an initial
bring-up blocker. It must be resolved or pinned to a known-good LMCache commit
before long-running stability tests or performance conclusions. The optional
`lmcache.cuda_ops` extension is also absent from the published environment, so
LMCache currently uses its functional Torch baseline transfer implementation.

The old `sparsespec-a100` environment remains independent and is still the
reference environment for the audited baseline.

## vLLM fork remote

Create an empty GitHub repository named `self-spec-vllm`, then run:

```bash
cd external/vegas-vllm-pr1
git remote add origin git@github.com:shiyisong015-source/self-spec-vllm.git
git push -u origin research/lmcache-residency
```

Keep `upstream` pointing to `npz7yyk/vegas`; use `origin` only for the research
fork. Do not push research commits to `upstream`.
