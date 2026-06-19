# Bug Report: `nki_qkv` prefill path (`qkv_cte`) produces wrong / non-deterministic output

A self-contained reproducer is provided in **`repro_qkv_cte_bug.py`** (single file,
no project dependencies beyond `torch` + `nkilib`).

## Summary

`nkilib.core.qkv.qkv` (`nki_qkv`), on its **context-encoding / prefill path
`qkv_cte`** (selected when `seqlen > SEQLEN_THRESHOLD_FOR_QKV_CTE == 96`), does not
correctly produce its output. It returns **near-zero, input-independent, and
non-deterministic** values — consistent with an unwritten / partially-written
`shared_hbm` output buffer, likely compounded by an LNC2 cross-core race. The garbage
propagates downstream and overflows to `inf`/`NaN` after subsequent operations.

The `qkv_tkg` path (`seqlen <= 96`) is correct. `attention_cte` and
`output_projection_cte` are correct and stable.

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
- Output **changes between identical consecutive calls** → reads uninitialized HBM;
  suggests a missing barrier / sharding race on the LNC2 write path.
- QK-norm config (`qk_norm_pre_rope`) has **no effect** on the output → the projection
  itself is never correctly computed on the CTE path.
- The TKG path (single-core, `seqlen <= 96`) is correct.

## Suspected area

`nkilib/core/qkv/qkv_cte.py`, `_qkv_cte_impl` BSD output write
(`nisa.dma_copy(dst=output_hbm.ap(...), src=output_sb[...])` around the
`cfg.output_layout == QKVOutputLayout.BSD` branch), the matmul→`output_sb` chain
(`nc_matmul` into `qkv_MM_output_psum`, evicted to `output_sb`), and the LNC2 sharding
(`dims.S_shard_offset` / `S_shard`) + end barrier
(`get_verified_program_sharding_info("qkv_cte_barrier", ...)` /
`nisa.core_barrier(output_hbm, (0, 1))`).

Isolation attempts that did **not** change the (bit-identical broken) output:
- `NEURON_RT_NUM_CORES=1` — likely does not alter the kernel's *internal* SPMD grid on
  an LNC=2 part, so it does not cleanly rule out the sharding race.
- `load_input_with_DMA_transpose=False` — appears ignored internally on this path.

Within one process the output varies between consecutive identical calls (S=128:
12.158 then 12.172); across processes it is reproducible for a fixed compiled NEFF —
i.e. an uninitialized-buffer read with a deterministic allocation pattern. Confirming
the exact cause requires dumping intermediate `input_sb` / `weights_sb` / PSUM tiles
inside `_qkv_cte_impl`.

## Workaround (in our model wrapper)

Bypass `nki_qkv`: compute QKV projection + QK-norm + RoPE in PyTorch and feed the
result into `attention_cte` (+ `output_projection_cte`, both verified correct). This
yields a stable, NaN-free model whose predictions match the eager baseline exactly.
