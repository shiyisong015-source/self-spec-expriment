# Official vLLM + LMCache baseline

## Scope

This is the clean control stack for the self-speculation integration. Neither
package is editable and no source patch is applied:

- Python 3.12.14
- PyTorch 2.13.0+cu130
- vLLM 0.28.0 (official wheel)
- LMCache 0.5.5 (official wheel)
- CuPy 14.2.0 for CUDA 13
- NVIDIA A100-PCIE-40GB (SM80)

The root requirements are pinned in
`requirements-vllm-lmcache-official.txt`. Install both roots in a single
resolver run so vLLM, LMCache, PyTorch, and CUDA packages are selected as one
stack:

```bash
CONDA_PKGS_DIRS="$PWD/.conda/pkgs" conda create -y \
  -p "$PWD/.conda/envs/vllm-lmcache-official" \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.12 pip

UV_LINK_MODE=hardlink uv pip install \
  --python "$PWD/.conda/envs/vllm-lmcache-official/bin/python" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-vllm-lmcache-official.txt
```

Activate and validate the environment:

```bash
source scripts/activate_clean_lmcache.sh
python scripts/check_clean_lmcache.py
```

The checker rejects an editable vLLM install, checks the exact package roots,
loads LMCache's external MP connector and CUDA extension, and requires a
working CUDA device.

## Validated topology

Use LMCache's multiprocess architecture: an independent `lmcache server`
stores the KV objects, and vLLM loads `LMCacheMPConnector` from
`lmcache.integration.vllm.lmcache_mp_connector`. This keeps cache ownership
outside the engine and is the current LMCache-recommended deployment mode.

Run the local offline control with the existing model checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_clean_lmcache_smoke.sh
```

The wrapper uses ports 15555 and 18080, allocates a 2 GiB LMCache L1 tier,
loads `/data/models/Qwen3-4B`, and always terminates the standalone cache
server. The validated run stored 1,792 shared-prefix tokens and retrieved all
1,792 on the second request. It emitted no `Double unpin`, negative pin count,
or traceback. During teardown of the short-lived offline `LLM` process,
PyTorch prints one CUDA-IPC producer-lifetime warning after LMCache unregisters
its KV caches; both services still exit cleanly. Use the normal long-running
server topology for benchmarks rather than including this teardown phase.

## Rejected control path

The same official package pair was also tested with the in-process dynamic
connector. Cache lookup and retrieval worked, but each retrieved chunk caused
a `Double unpin` warning (seven warnings for seven 256-token chunks). That is
an LMCache lifecycle failure in an otherwise unmodified control, so the
in-process connector is not the base for self-spec work. The MP control must
remain green before any scheduler or KV-residency modification is evaluated.
