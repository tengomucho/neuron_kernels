#!/usr/bin/env python
"""
Standalone reproducer: nkilib `nki_qkv` returns wrong / non-deterministic output on
its context-encoding (prefill) path `qkv_cte`, selected when seqlen > 96
(SEQLEN_THRESHOLD_FOR_QKV_CTE). The decode path `qkv_tkg` (seqlen <= 96) is correct.

We test the simplest possible configuration: a plain QKV projection with NO fused
norm and NO fused RoPE. The kernel output must then equal `hidden @ weights.T` to
bf16 rounding. Instead, on the CTE path the output is ~0, independent of the input,
and varies between identical consecutive calls.

No model / Hugging Face download required -- only torch + nkilib on a Trainium host.

Usage:
    python repro_qkv_cte_bug.py
"""
import torch

import nki
import nki.language as nl  # noqa: F401  (import kept to surface nki load errors early)
from nkilib.core.qkv.qkv import qkv as nki_qkv
from nkilib.core.qkv.qkv import SEQLEN_THRESHOLD_FOR_QKV_CTE
from nkilib.core.utils.common_types import QKVOutputLayout, NormType

# Qwen3-0.6B attention dimensions (the bug is not specific to these values).
H = 1024          # hidden size
D = 128           # head dim
N_Q = 16          # query heads
N_KV = 8          # key/value heads
I = (N_Q + 2 * N_KV) * D   # fused qkv output dim = 4096
DEVICE = "neuron"
DTYPE = torch.bfloat16
TOL = 0.1         # generous bf16 tolerance; correct path lands ~0.03


def versions():
    def v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            try:
                import importlib.metadata as m
                return m.version(mod)
            except Exception:
                return "?"
    print("=" * 70)
    print("Environment")
    print(f"  nki          : {v('nki')}")
    print(f"  nki_library  : {v('nki_library')}")
    print(f"  torch_neuronx: {v('torch_neuronx')}")
    print(f"  neuronx_cc   : {v('neuronx-cc')}")
    print(f"  torch        : {torch.__version__}")
    print(f"  SEQLEN_THRESHOLD_FOR_QKV_CTE = {SEQLEN_THRESHOLD_FOR_QKV_CTE}")
    print("=" * 70)


def run_case(seqlen, weights_cpu):
    """Run projection-only nki_qkv twice and compare to hidden @ W.T (computed on CPU)."""
    B = 1
    torch.manual_seed(0)
    hidden_cpu = torch.randn(B, seqlen, H, dtype=torch.float32)
    ref = hidden_cpu @ weights_cpu.T                       # [B, seqlen, I]
    hidden = hidden_cpu.to(DEVICE, DTYPE)
    weights = weights_cpu.to(DEVICE, DTYPE)

    path = "qkv_tkg (decode)" if seqlen <= SEQLEN_THRESHOLD_FOR_QKV_CTE else "qkv_cte (prefill)"
    print(f"\nseqlen={seqlen:<5}  path={path}")
    diffs = []
    outs = []
    for r in range(2):
        out = nki_qkv(
            hidden,
            weights.T,                                     # [H, I]
            output_layout=QKVOutputLayout.BSD,
            fused_norm_type=NormType.NO_NORM,
            fused_rope=False,
            d_head=D,
            num_q_heads=N_Q,
            num_kv_heads=N_KV,
        )
        out_cpu = out.detach().to("cpu", torch.float32)
        d = (out_cpu - ref).abs().max().item()
        diffs.append(d)
        outs.append(out_cpu)
        print(f"  run{r+1}: maxdiff_vs_(hidden@W.T)={d:8.4f}   "
              f"ref_range=[{ref.min():.2f},{ref.max():.2f}]   "
              f"kernel_range=[{out_cpu.min():.2f},{out_cpu.max():.2f}]")

    nondet = (outs[0] - outs[1]).abs().max().item()
    ok = max(diffs) < TOL
    print(f"  -> {'OK   ' if ok else 'WRONG'}  (max maxdiff={max(diffs):.4f}, "
          f"run-to-run delta={nondet:.4f})")
    return ok


def main():
    versions()
    torch.manual_seed(1234)
    weights_cpu = torch.randn(I, H, dtype=torch.float32) * (H ** -0.5)

    results = {}
    for seqlen in (64, 128, 256):
        results[seqlen] = run_case(seqlen, weights_cpu)

    print("\n" + "=" * 70)
    print("Summary (expected: all correct to bf16 rounding ~0.03)")
    for seqlen, ok in results.items():
        path = "TKG" if seqlen <= SEQLEN_THRESHOLD_FOR_QKV_CTE else "CTE"
        print(f"  seqlen={seqlen:<5} [{path}] : {'OK' if ok else 'WRONG  <-- BUG'}")
    print("=" * 70)
    if not all(results.values()):
        print("REPRODUCED: nki_qkv qkv_cte path (seqlen > "
              f"{SEQLEN_THRESHOLD_FOR_QKV_CTE}) produces incorrect output.")


if __name__ == "__main__":
    main()
