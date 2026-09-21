# Self-Speculation / Future-HBM research environment

This workspace follows the first-stage plan in
`self_spec_future_hbm_research_plan.docx`: single-GPU profiling of draft,
verify, and KV-transfer phases, followed by FutureCapacityTracker and adaptive
draft-length work. The current integration route starts from a clean official
**vLLM + LMCache MP** baseline and adds self-speculation only after that
control passes. The previous SparseSpec and MagicDec experiments remain below
as references.

## Active route: official vLLM + LMCache MP baseline

The isolated environment contains only published wheels: vLLM 0.28.0,
LMCache 0.5.5, and their resolved PyTorch 2.13.0+cu130 runtime. It does not use
an editable vLLM checkout or a patched LMCache tree.

```bash
source scripts/activate_clean_lmcache.sh
python scripts/check_clean_lmcache.py
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clean_lmcache_smoke.sh
```

The smoke wrapper starts an independently managed LMCache MP server, runs two
Qwen3-4B requests with a shared prefix, proves that 1,792 tokens are stored and
then retrieved, checks logs for cache-lifecycle failures, and shuts the server
down. See [docs/official_vllm_lmcache_baseline.md](docs/official_vllm_lmcache_baseline.md)
for installation and design notes.

## Previous route: SparseSpec on A100

The isolated project-local environment is already installed at
`.conda/envs/sparsespec-a100`. Activate it with the helper so that the SM80
architecture, RAFT/RMM libraries, SparseSpec source tree, and local JIT caches
are all configured together:

```bash
source scripts/activate_sparsespec.sh
```

Because this is a prefix environment rather than an environment created in
Conda's default named-env directory, `conda activate sparsespec-a100` is not
portable here. The equivalent direct command is:

```bash
conda activate /home/songshiyi/self_spec/.conda/envs/sparsespec-a100
```

The current tested stack is PyTorch 2.8.0+cu128, Transformers 4.55.2,
SparseSpec's customized FlashInfer 0.2.14.post1, and its customized vLLM
0.11.1 development fork. SparseSpec is pinned at commit
`5c9616c9af109ecec0fdf193221d3f5be6f05922`.

Run the offline end-to-end smoke test without downloading a model or dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_sparsespec.py \
  --model /data/models/Qwen3-8B
```

This exercises Qwen3-8B weight loading, the Pillar sparse KV cache, sparse
draft attention, verification, rejection sampling, and text decoding. On the
current A100-PCIE-40GB host, the cached smoke run completed its inference phase
in 0.9142 seconds, generated 20 tokens, and accepted 15/15 speculative tokens.
The first run took 74.9 seconds in the inference phase because the custom SM80
kernels were being JIT-compiled. These small-run figures are compatibility
evidence, not benchmark results. Sparse verification can finish a whole stride,
so the generated length may exceed the requested target by up to one stride.

The upstream kernel checks also pass on SM80:

```bash
python external/SparseSpec/serve/tests/test-chunked-prefill-attn.py
python external/SparseSpec/serve/tests/test-batch-topk-sort.py
python external/SparseSpec/serve/tests/test-batch-topk-attention.py
```

### vLLM MagicDec-style self-spec

The customized vLLM fork's existing V1 `self_specs` path also runs on one
A100; no separate MagicDec serving scheduler is used or maintained here. Run
the offline compatibility test with:

```bash
source scripts/activate_sparsespec.sh
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_vllm_magicdec.py
```

The test uses the fork's native FlashInfer backend, block-size-1 streaming KV
cache, 32 sink blocks, a 5% recent window, an 8-token self-spec stride, and
PIECEWISE CUDA Graph capture. The initial A100 run generated 48 tokens in
1.54 seconds after engine initialization; the cached rerun took 0.849 seconds.
Its runtime metrics reported 6 verification batches, 48 drafted tokens, and
48 accepted tokens. This is a compatibility smoke test rather than a
throughput comparison.

## Base Conda environment

Create a named environment:

```bash
conda env create -f environment.yml
conda activate self-spec-hbm
python scripts/check_environment.py
```

For this workspace, the environment has also been created at a project-local
prefix. The helper below activates it and redirects Hugging Face, datasets,
Matplotlib, and XDG caches into the project-local ignored `.cache` directory:

```bash
source scripts/activate.sh
python scripts/check_environment.py
```

The environment contains Python 3.12, build tools, dataset support, plotting,
testing, and NVML bindings. It intentionally does not install PyTorch or stock
vLLM: Vegas is a vLLM source fork with a companion FlashAttention fork, and its
editable install supplies the compatible PyTorch/CUDA build dependencies.

## Previous candidate: Vegas

Vegas remains available as a previous candidate, but its Ampere gather fallback
is not the experimental performance path. Its setup notes are retained below
for reference.

Run these steps on a host where `nvidia-smi` succeeds and a CUDA 12.x toolkit is
available:

```bash
git clone https://github.com/platformxlab/vegas.git external/vegas
git -C external/vegas apply ../../patches/vegas-fa2-ampere.patch
python -m pip install -v -e external/vegas
```

The source build compiles CUDA and FlashAttention-3 kernels and can take a long
time. This workspace currently pins the Vegas source checkout to commit
`7bf9d1edb8223307cff2089f855ca69cd094bc69`. Verify the completed installation
and its custom FlashAttention `scores` API before experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_vegas.py
```

An end-to-end smoke test uses the shared Qwen3-8B checkpoint without copying it:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_vegas.py \
  --model /data/models/Qwen3-8B
```

The upstream sparse draft path uses FlashAttention-3 page-size-1 KV access.
This host has Ampere A100 GPUs and therefore uses FlashAttention-2, whose paged
cache requires a block size divisible by 16. The included patch adds an
Ampere-only functional fallback that gathers Vegas-selected KV slots into a
non-paged variable-length buffer. It is suitable for correctness checks and
instrumentation development, but its extra gather/synchronization overhead
must not be included in Vegas throughput claims. Reproduce published kernel
performance on a Hopper/FA3 machine without this patch.

## Nsight Systems baseline

The research plan's baseline trace can be collected as follows after adding
NVTX ranges to the benchmark/server path:

```bash
mkdir -p profiles
TAG=baseline
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-devices=cuda-visible \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "profiles/selfspec_${TAG}" \
  python external/vegas/benchmarks/benchmark_vegas.py
```

GPU metrics may require performance-counter permissions. CUDA/NVTX and memory
operation traces can still be collected without those counters by dropping the
`--gpu-metrics-devices` option.
