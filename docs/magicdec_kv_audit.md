# vLLM MagicDec-style self-spec and KV ownership audit

## Audit scope and pinned baseline

- Workspace baseline commit: `63433a89e5550c110baeae6c1594921d2a0aecb8`
- Workspace branch: `research/magicdec-longbench-profile`
- Workspace remote: `git@github.com:shiyisong015-source/self-spec.git`
- SparseSpec vLLM fork commit: `9cdebe5afed2c754c64fead58064ac6ac7a43c6d`
- Runtime method: vLLM V1 `SpeculativeConfig(method="self_specs")`

This document describes the current code without proposing or implementing CPU
KV offload.

## Executive conclusions

1. **Sparse KV is not an independent physical allocation.** There is one set
   of GPU KV tensors. Sparse attention builds a shorter list of physical page
   IDs from the request's full block table and passes those IDs to FlashInfer.
2. Self-spec state is per request. Different requests in one scheduler output
   can be `ACCUMULATING` (draft) and `VERIFYING` (verify) at the same time, and
   they are handled by one target-model forward. The implementation is
   therefore not a global `Draft batch -> ... -> Verify batch` pipeline.
3. A single request still follows an ordered state machine:
   `NORMAL -> ACCUMULATING x gamma -> VERIFYING -> NORMAL`.
4. `gamma` is `SpeculativeConfig.num_speculative_tokens`; the scheduler copies
   it into `self_spec_threshold`. The tested baseline uses `gamma = 8`.

## Request and scheduler call path

The engine core repeatedly calls:

```text
Scheduler.schedule()
  -> admit waiting requests / schedule running requests
  -> KVCacheManager.allocate_slots()
  -> SchedulerOutput
GpuModelRunner.execute_model()
  -> _update_states()
  -> _prepare_inputs()
  -> model forward
  -> sampling / rejection sampling
Scheduler.update_from_output()
  -> commit, buffer, or accept/reject output tokens
```

Relevant code:

- Request state and token buffers:
  `vllm/v1/request.py::SelfSpecState`, `Request`,
  `Request.add_pending_token()`, and
  `Request.start_self_spec_verification()`.
- Request admission and running-request scheduling:
  `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()`.
- State transitions and result processing:
  `Scheduler.update_from_output()` and
  `Scheduler._update_request_with_output()`.
- Worker-side execution:
  `vllm/v1/worker/gpu_model_runner.py::GpuModelRunner.execute_model()`.

Paths above are relative to
`external/SparseSpec/3rdparty/vllm/`.

## One request from prefill through a speculative round

1. **Admission and prefill (`NORMAL`)**

   `Scheduler.schedule()` removes an admissible request from `waiting`, asks
   `KVCacheManager.allocate_slots()` for its required blocks, appends the
   request to `running`, and schedules prompt tokens. The model forward writes
   prompt K/V through the normal `slot_mapping` into the physical GPU KV
   tensors. Prefill and full attention use the complete block table.

2. **First committed decode and transition**

   A token generated in `NORMAL` is immediately appended to the request's
   verified output by `Scheduler._update_request_with_output()`. After exposing
   that token, `Scheduler.update_from_output()` changes the request to
   `ACCUMULATING` and records `full_kv_start_block_offset` at the current
   computed position.

3. **Draft steps (`ACCUMULATING`)**

   Each subsequent target-model decode produces a token, but
   `Request.add_pending_token()` stores it in `_pending_output_tokens`; it is
   not yet returned to the client. `Scheduler.schedule()` uses
   `delay_cache_blocks=True` for this state. The physical slots are still from
   the same `KVCacheManager` and the same per-request block table.

   In `GpuModelRunner._prepare_inputs()`, self-spec enables FlashInfer's
   `build_with_streaming()`. The builder constructs attention page IDs from:

   - the first `sink_size` blocks;
   - the most recent `recent_ratio * sequence_length` blocks before the saved
     full-KV offset; and
   - every block from `full_kv_start_block_offset` through the current tail.

   The page IDs select existing entries in the full block table. No K/V tensor
   is copied into a separate sparse cache.

4. **Verification (`VERIFYING`)**

   `Scheduler.should_start_self_spec_verification()` fires when the pending
   token count reaches `self_spec_threshold` (or early near the model-length
   limit). `Request.start_self_spec_verification()` moves pending IDs to
   `spec_token_ids`. The scheduler rolls the computed position back by one
   token, sets `full_kv_start_block_offset = 0`, and schedules the pending IDs
   through vLLM's normal speculative-decode metadata.

   Offset zero makes `build_with_streaming()` select all pages for that request,
   so verification uses full attention. `GpuModelRunner._calc_spec_decode_metadata()`
   forms target/bonus logits and `GpuModelRunner._sample()` invokes the
   `RejectionSampler`.

5. **Acceptance and next round**

   `Scheduler.update_from_output()` computes
   `num_accepted = len(generated_token_ids) - 1`, rolls back rejected computed
   tokens, clears `spec_token_ids`, and restores `NORMAL`. Accepted tokens and
   the bonus token are processed by `_update_request_with_output()`. If the
   request remains active, its next committed `NORMAL` token starts another
   `ACCUMULATING` round.

## Draft and verify batching structure

The outer scheduling loop processes each running request independently. It can
transition one request to `VERIFYING` while another remains `ACCUMULATING`.
`scheduled_spec_decode_tokens` only contains verifying requests, while the
same `SchedulerOutput.num_scheduled_tokens` includes the other scheduled
requests. `GpuModelRunner._prepare_inputs()` builds one packed input and one
model forward for that mixed set.

The FlashInfer streaming metadata is also per request:
`sink_sizes`, `recent_sizes`, and `full_kv_start_offset`. An accumulating row
has a nonzero offset and receives selected pages; a normal or verifying row has
offset zero and receives all pages. Consequently, draft and verify requests
**can coexist in one target-model forward**.

## Physical KV and page-table ownership

```text
Request (scheduler state)
  |
  +-- KVCacheManager / coordinator
  |     |
  |     +-- request_id -> KVCacheBlock objects
  |             |
  |             +-- physical block_id values
  |
  +-- worker MultiGroupBlockTable row (full block_id sequence)
          |
          +-- slot_mapping ------------------------------+
          |                                               |
          +-- FlashInfer build_with_streaming()           |
                 |                                        |
                 +-- derived paged_kv_indices             |
                     (sink + recent + post-offset pages)   |
                                                          v
                     one physical set of per-layer GPU KV tensors
```

Physical storage is created only in
`GpuModelRunner._allocate_kv_cache_tensors()`, reshaped by
`_reshape_kv_cache_tensors()`, and bound to attention layers by
`initialize_kv_cache_tensors()` / `bind_kv_cache()`. There is no second
allocation for sparse K/V.

The authoritative scheduler-side ownership is held by
`KVCacheManager.coordinator`; the worker's `MultiGroupBlockTable` mirrors the
allocated block IDs for kernel input. FlashInfer's `paged_kv_indices` is a
temporary selection/index buffer, not an owner of K/V pages.

## KV allocation and release

- Allocation entry point:
  `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots()`.
- Physical block objects and free queue:
  `vllm/v1/core/block_pool.py::BlockPool` and `BlockPool.get_new_blocks()`.
- Scheduler-to-worker block propagation:
  `Scheduler._make_cached_request_data()`,
  `GpuModelRunner._update_states()`, and
  `vllm/v1/worker/block_table.py::MultiGroupBlockTable`.
- Request completion:
  `Scheduler._free_request()` -> `Scheduler._free_blocks()` ->
  `KVCacheManager.free()` -> coordinator `free(request_id)`.

`delay_cache_blocks=True` during accumulation delays prefix-cache commitment;
it does not create a different physical allocator or CPU backing store.

## Sparse page selection and attention invocation

- Dispatch decision:
  `GpuModelRunner._prepare_inputs()` sets `use_selective_kv` for
  `speculative_config.use_self_specs()` and invokes
  `FlashInferMetadataBuilder.build_with_streaming()`.
- Page count and page list:
  `vllm/v1/attention/backends/flashinfer.py::_compute_num_selective_blocks()`
  and `_copy_selective_page_indices_kernel()`.
- Sparse draft attention:
  nonzero `full_kv_start_offset` selects sink, recent, and post-offset pages.
- Full prefill/verify attention:
  offset zero causes the same builder to copy every page from the full block
  table.
- Physical K/V read/write:
  the normal model forward in `GpuModelRunner.execute_model()` uses the single
  bound KV tensor set and the metadata/page IDs described above.

## Configuration map

- Gamma: `SpeculativeConfig.num_speculative_tokens` ->
  `Scheduler.num_spec_tokens` -> `Scheduler.self_spec_threshold`.
- Sink budget: `SchedulerConfig.sink_size`, measured in blocks.
- Recent budget: `SchedulerConfig.recent_ratio`, converted by
  `GpuModelRunner._prepare_inputs()` to a per-request page count.
- Page size: `CacheConfig.block_size`; the current MagicDec-style baseline uses
  page size 1.
- Acceptance: `GpuModelRunner._sample()` / `RejectionSampler.parse_output()`,
  followed by `Scheduler.update_from_output()`.

## Implication for the next phase

The present design reduces draft attention bandwidth but retains the complete
KV allocation on GPU. A future VeriCache-style design must introduce a real
ownership split: CPU-owned full-KV backing plus a separately managed set of
physically resident GPU sparse pages. That change is intentionally out of
scope for this audit.
