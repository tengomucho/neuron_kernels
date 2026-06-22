# Bug Report: `nki_qkv` prefill path (`qkv_cte`) produces wrong / non-deterministic output

A self-contained reproducer is provided in **`repro_qkv_cte_bug.py`** (single file,
no project dependencies beyond `torch` + `nkilib`).

## Summary

`nkilib.core.qkv.qkv` (`nki_qkv`), on its **context-encoding / prefill path
`qkv_cte`** (selected when `seqlen > SEQLEN_THRESHOLD_FOR_QKV_CTE == 96`), returns an
**all-zero** output instead of the QKV projection. The root cause has two parts (see
"Root cause" below for the instrumented detail):

- **Bug A:** `_multi_buffering_degree_for_seqlen()` computes a **negative**
  `s_multi_buffer_degree` (the lookahead SBUF-space estimate exceeds available SBUF for
  H=1024/I=4096), which collapses the block count to 0 and **skips the entire projection
  loop**, so the output buffer is never written.
- **Bug B:** even after clamping the degree to ≥ 1 so the loop runs, the CTE output is
  still never delivered to the caller — a forced write to the output buffer does not
  appear in the returned tensor, while the structurally-identical `qkv_tkg` path works.
  This points to an output-binding problem in `qkv_cte`'s codegen.

The all-zero (or, with real weights, uninitialised-HBM) output propagates downstream and
overflows to `inf`/`NaN`. The `qkv_tkg` path (`seqlen <= 96`) is correct, as are
`attention_cte` and `output_projection_cte`.

## Environment

- Platform: AWS Trainium2 (`trn2.48xlarge`, LNC=2)
- `nki`: 0.4.0b4+25404641739.g5cabf9dc
- `nki_library` (nkilib): 1.0.10859.0a0+f2f9f1d3
- `torch_neuronx`: 0.1.0+d0471391
- `neuronx_cc`: 2.0.253977.0a0+2ba785af
- Model context: Qwen3-0.6B (H=1024, heads=16, kv_heads=8, head_dim=128), bf16

## Minimal reproduction

Run the self-contained script: `python repro_qkv_cte_bug.py`. It does a projection
only (no fused norm, no RoPE), so the output must equal `hidden @ W.T` to bf16
rounding. Observed on Trn2 (`trn2.48xlarge`):

```
seqlen=64    path=qkv_tkg (decode)    maxdiff=0.0231   kernel_range=[-4.69,4.72]   -> OK
seqlen=128   path=qkv_cte (prefill)   maxdiff=4.7790   kernel_range=[ 0.00,0.00]   -> WRONG
seqlen=256   path=qkv_cte (prefill)   maxdiff=4.7790   kernel_range=[ 0.00,0.00]   -> WRONG
```

Expected: `maxdiff` ≈ bf16 rounding (~0.03) at all sequence lengths. On the CTE path
the kernel output is all zeros (with these random weights) — i.e. the projection
result never reaches the output buffer.

With real (Qwen3-0.6B) weights and inputs the CTE output is instead small non-zero
**garbage that also varies between identical consecutive calls** (e.g. S=128:
maxdiff 12.158 then 12.172), i.e. an uninitialized-buffer read rather than a clean
zero. Either way the projection is not correctly produced, and downstream it overflows
to `inf`/`NaN`.

## Key diagnostic signals

- Output magnitude is ~0 and independent of the input → output buffer not written.
- QK-norm config (`qk_norm_pre_rope`) has **no effect** on the output → the projection
  itself is never correctly computed on the CTE path.
- The TKG path (`seqlen <= 96`) is correct.
- `NEURON_RT_NUM_CORES=1` does **not** change the result (this does not necessarily alter
  the kernel's internal SPMD grid, but together with the root-cause analysis below an
  LNC2 cross-core race is **not** the cause — the block loop is simply skipped).
- With real Qwen3 weights the CTE output is small **non-deterministic** garbage rather
  than a clean zero (uninitialised-HBM read); with random weights it is exactly zero.

## Root cause (instrumented on trn2.48xlarge)

Debugging was done with a local editable copy of `qkv_cte.py`
(`qwen3-neuron-kernels/qkv_cte_local.py`, driven by `repro_qkv_cte_bug.py --local`).
NOTE: the NKI on-disk compile cache (`nki/compiler/disk_cache.py`) must be disabled
(`NKI_DISABLE_COMPILE_CACHE=1`) or source edits silently have no effect.

**Bug A — negative multi-buffering degree skips ALL compute (primary cause).**
`_multi_buffering_degree_for_seqlen()` computes a **negative** `s_multi_buffer_degree`.
Printed values for H=1024, I=4096 (per partition):

```
total_available_sbuf_space_to_this_kernel = 212984 B  (~208 KB)
sbuf_tile_space_non_buffered              = 262148 B   <-- exceeds available
  of which weights_space_per_partition    = 262144 B   (256 KB, dominant term)
sbuf_tile_space_pre_buffering             = 10240 B
max_s_buffer_without_exceeding_sbuf = (212984 - 262148) // 10240 = -5
s_multi_buffer_degree = min(initial_degree, -5) = -5      # no clamp to >= 1
```

Then `S_BLOCK_SIZE = degree * min(S_shard,128) < 0` and
`num_blocks_per_S_shard = ceil(S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0 = 0`,
so `for i_block_S in nl.affine_range(0)` runs **zero iterations** — the QKV projection
is never computed and `output_hbm` is returned as freshly-allocated zeros (→ NaN
downstream). The lookahead `weights_space_per_partition` estimate (256 KB) alone
exceeds the available SBUF (208 KB), which is the source of the negative value.
Two fixes are needed: (1) clamp `s_multi_buffer_degree = max(1, ...)`; (2) correct the
over-large non-prefetched weights-space estimate so it does not exceed available SBUF.

**Bug B — CTE output is not bound to the kernel result (still unresolved).**
After clamping the degree to 1 (loop now runs: `degree=1`, `num_blocks=1/2`), the CTE
output is *still* all zeros. A forced sentinel write of a constant to `output_hbm`
(tested inside the block loop, in `qkv_cte`'s outer scope, via both `.ap(pattern=...)`
and plain slicing, with `dge_mode` `swdge` and `none`) **never appears in the returned
tensor**, while the structurally-identical `qkv_tkg` path writes its internally-
allocated `nl.shared_hbm` output correctly through the same dispatcher/return. This
indicates the CTE kernel's internally-allocated output is not wired as the kernel
result — consistent with the `qkv()` entry's
`experimental_flags="skip-non-top-level-shared-hbm-check"` masking a real output-binding
problem in codegen. This part needs AWS/compiler-level investigation (e.g. dumping the
generated IR, or having `qkv_cte` accept/return the output at the jit top level).

Ruled out as causes of Bug B: cross-module monkeypatch, DMA addressing style
(`.ap` vs slice), `dge_mode` (`swdge` vs `none`), placement inside vs outside
`nl.affine_range`, the `use_BxS_input_reshape` output reshape, and allocating the
output at the jit top level and passing it in via `output_hbm=` (still zero, while the
internally-allocating `qkv_tkg` works — so the binding failure is specific to the
`qkv_cte` codegen, not where/how the output buffer is allocated).

## Workaround (in our model wrapper)

Bypass `nki_qkv`: compute QKV projection + QK-norm + RoPE in PyTorch and feed the
result into `attention_cte` (+ `output_projection_cte`, both verified correct). This
yields a stable, NaN-free model whose predictions match the eager baseline exactly.
