# MagicDec baseline profiling

The profiling hooks are an opt-in patch against the pinned SparseSpec vLLM
checkout (`9cdebe5afed2c754c64fead58064ac6ac7a43c6d`). They do not implement
VeriCache, CPU KV offload, or a new serving scheduler. With
`VLLM_SELFSPEC_PROFILE` unset, the helper performs only a cached boolean check
and emits neither logs nor NVTX ranges.

## Apply the patch

From the repository root:

```bash
git -C external/SparseSpec/3rdparty/vllm apply \
  ../../../../patches/sparsespec-vllm-selfspec-profile.patch
source scripts/activate_sparsespec.sh
```

The current editable vLLM installation resolves directly to this checkout, so
no reinstall is required. To confirm that the patch matches before applying it:

```bash
git -C external/SparseSpec/3rdparty/vllm apply --check \
  ../../../../patches/sparsespec-vllm-selfspec-profile.patch
```

## Structured step log

```bash
mkdir -p data/profiles data/results
source scripts/activate_sparsespec.sh
VLLM_SELFSPEC_PROFILE=1 \
VLLM_SELFSPEC_PROFILE_LOG="$PWD/data/profiles/selfspec.jsonl" \
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/run_magicdec_longbench.py \
  --input data/workloads/lb_16k_18k.jsonl \
  --output data/results/magicdec_profile.json \
  --limit 1
```

Each JSONL record includes timestamp, scheduler iteration, request ID, mode,
context length, current draft length, gamma, sparse/full page counts, accepted
tokens, batch size, and forward-token count. `accepted_tokens` is null on the
scheduling record and populated by the corresponding `verify_result` record.
Page counts are host-side calculations and introduce no CUDA synchronization.

## Nsight Systems smoke

```bash
source scripts/activate_sparsespec.sh
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --trace-fork-before-exec=true \
  --output=data/profiles/magicdec_nvtx_smoke \
  env VLLM_SELFSPEC_PROFILE=1 \
      VLLM_SELFSPEC_PROFILE_LOG="$PWD/data/profiles/nsys_selfspec.jsonl" \
      CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/run_magicdec_longbench.py \
    --input data/workloads/lb_16k_18k.jsonl \
    --output data/results/magicdec_nsys_smoke.json \
    --limit 1 --max-new-tokens 16

nsys stats --force-export=true --report nvtx_sum \
  data/profiles/magicdec_nvtx_smoke.nsys-rep
```

The validated A100 smoke contained all requested ranges: 9 `prefill`, 48
`draft_step`, 6 `verify_step`, 48 `sparse_attention`, 15 `full_attention`, and
64 `scheduler_step` instances. The structured log reported sparse working sets
of 930--938 pages versus about 17.9K full pages. Six verification steps accepted
9 tokens in total, matching the native vLLM self-spec metric.

All reports, SQLite exports, JSONL logs, workloads, and result JSON files live
under ignored `data/` paths (and `*.nsys-rep`/`*.sqlite` are ignored globally).
Do not commit them.
