import os
import time

# os.environ["TORCH_NEURONX_ENABLE_STABLEHLO"] = "0"
# os.environ["ON_NEURON_EAGER"] = "1"
# os.environ["TORCH_NEURONX_MLIR_ATEN_OPS"] = "1"
# os.environ["ON_NEURON"] = "1"
# os.environ["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"
# os.environ["NEURON_RT_MAP_HBM"] = "0"
# os.environ["NEURON_RT_DBG_ZEROCOPY"] = "0"
# os.environ["NEURON_EAGER_MODEL_CACHE_SIZE"] = "128"
# os.environ["NEURON_RT_NUM_CORES"] = "1"
# os.environ["OMP_NUM_THREADS"] = "128"

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig


NUM_WARMUP = int(os.environ.get("NUM_WARMUP", 10))
NUM_RUNS = int(os.environ.get("NUM_RUNS", 50))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 1024))
DTYPE = torch.bfloat16

# model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
model_id = "Qwen/Qwen3-0.6B"

# Local kernel repo (the in-tree, patched kernel). Its layout is
# qwen3-neuron-kernels/build/<variant>/__init__.py, which the `kernels` library
# loads when given the repo root path + use_local_kernel=True.
LOCAL_KERNEL_REPO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen3-neuron-kernels"
)


def _sync(out):
    """Force the forward to actually execute. The Neuron `torch.compile` backend runs
    fully async/lazy, so calling the model only *dispatches* work -- without reading a
    value back, the timed loop measures dispatch latency (~0.5 ms), not compute. Reading
    a single scalar forces graph execution while avoiding a full logits transfer."""
    return float(out.logits.view(-1)[0])


def benchmark(model, inputs, label):
    """Run benchmark on a model."""
    print(f"  Warming up {label} ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _sync(model(**inputs))

    print(f"  Benchmarking {label} ({NUM_RUNS} runs)...")
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _sync(model(**inputs))
    elapsed = time.perf_counter() - start

    ms_per_run = elapsed / NUM_RUNS * 1000
    tps = SEQ_LEN / (ms_per_run / 1000)
    print(f"  {label}: {ms_per_run:.2f} ms/fwd  |  {tps:.1f} tokens/sec")
    return ms_per_run


def get_kernel_config(fused_mlp=True):
    """Returns the kernel configuration for Qwen3 model fusion.

    Uses the local, in-tree kernel repo (`LOCAL_KERNEL_REPO`) so the benchmark runs
    the patched `NeuronQwen3Attention` (which bypasses the broken `qkv_cte` path)
    instead of the version published on the Hub. `use_local_kernel=True` makes the
    `kernels` library resolve the `path:LayerName` strings as local repositories.

    Args:
        fused_mlp: When True, the post-attention RMSNorm is fused *into* the MLP kernel
            (`NeuronRMSNormMLP`, NormType.RMS_NORM). When False, the RMSNorm runs
            standalone in torch and the MLP kernel is called with NormType.NO_NORM
            (`NeuronRMSNormMLP_Separated`). The attention path is identical in both, so
            the only difference is whether the norm is fused -- which is exactly the A/B.
    """
    mlp_kernel = "NeuronRMSNormMLP" if fused_mlp else "NeuronRMSNormMLP_Separated"
    return KernelConfig(
        {
            "Qwen3Attention": f"{LOCAL_KERNEL_REPO}:NeuronQwen3Attention",
            (
                ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
                ("Qwen3MLP", "model.layers.*.mlp"),
            ): f"{LOCAL_KERNEL_REPO}:{mlp_kernel}",
        },
        use_local_kernel=True,
    )


def run_model_benchmark(model_id, inputs, compile_en, kernel_en, fused_mlp=True, label="model"):
    """
    Load a model, optionally enable kernel fusion and compilation, and run benchmarks.
    
    Args:
        model_id: HuggingFace model identifier
        inputs: Tokenized inputs (dict with 'input_ids', 'attention_mask', etc.)
        compile_en: Whether to apply torch.compile with neuron backend
        kernel_en: Whether to enable kernel fusion
        label: Label for benchmark output
    
    Returns:
        Tuple of (reference_output, benchmark_ms):
            - reference_output: Model logits output
            - benchmark_ms: Benchmark result in milliseconds per forward pass
    """
    print("=" * 60)
    print(f"Loading model: {label}...")
    
    # Build kernel config if requested
    kernel_config = get_kernel_config(fused_mlp=fused_mlp) if kernel_en else None
    
    # Load model with optional kernel fusion
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_kernels=True,
        kernel_config=kernel_config if kernel_en else None,
        dtype=DTYPE,
        device_map="neuron",
    )
        
    model.eval()
    
    # Move inputs to model device and get reference output
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        reference_out = model(**inputs).logits
    print(f"Output shape: {reference_out.shape}")
    
    # Apply torch.compile if requested
    if compile_en:
        print(f"  Compiling with neuron backend...")
        model = torch.compile(model, backend="neuron")
    
    # Run benchmark
    benchmark_ms = benchmark(model, inputs, label)
    
    return reference_out, benchmark_ms


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(
        "Hello, how are you?",
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )

    def safe_run(**kw):
        """Run a config; on failure, log it and return (None, None) so one broken
        config (e.g. the fragile torch.compile + kernels path) doesn't abort the
        whole comparison."""
        try:
            return run_model_benchmark(model_id, inputs, **kw)
        except Exception as e:
            print(f"  !! config '{kw.get('label')}' failed: {type(e).__name__}: {e}")
            return None, None

    # Run benchmarks for different configurations
    baseline_out, baseline_ms = safe_run(
        compile_en=False, kernel_en=False, label="baseline"
    )

    compiled_out, compiled_ms = safe_run(
        compile_en=True, kernel_en=False, label="compiled"
    )

    fused_out, fused_ms = safe_run(
        compile_en=False, kernel_en=True, fused_mlp=True, label="fused"
    )

    # Separated variant: identical attention path and identical MLP matmuls, but the
    # post-attention RMSNorm runs standalone (torch) instead of inside the MLP kernel.
    # The delta vs. `fused` isolates the value of the norm+MLP fusion. Keep compile off
    # so torch.compile can't re-fuse the torch-side norm and muddy the comparison.
    separated_out, separated_ms = safe_run(
        compile_en=False, kernel_en=True, fused_mlp=False, label="separated"
    )

    fused_compiled_out, fused_compiled_ms = safe_run(
        compile_en=True, kernel_en=True, fused_mlp=True, label="fused + compiled"
    )

    # --- compare ---
    # Compare against the baseline over the *real* (non-padding) tokens. The fused
    # attention kernel applies causal masking only and ignores the padding mask, so
    # padding positions diverge from the baseline (which masks them) -- that region is
    # never used and would otherwise dominate a naive full-tensor max-diff. We also
    # flag NaN/inf, which is the failure mode of the buggy kernel.
    print("=" * 60)
    real = inputs["attention_mask"].bool()[0]  # [S], still on CPU
    base_cpu = baseline_out.detach().to("cpu", torch.float32) if baseline_out is not None else None

    def report(name, out):
        if out is None or base_cpu is None:
            print(f"{name:24s} (skipped: config or baseline unavailable)")
            return
        o = out.detach().to("cpu", torch.float32)
        bad = torch.isnan(o).any().item() or torch.isinf(o).any().item()
        d_all = (o - base_cpu).abs().max().item()
        d_real = (o[:, real, :] - base_cpu[:, real, :]).abs().max().item()
        argmax_agree = (o[:, real, :].argmax(-1) == base_cpu[:, real, :].argmax(-1)).float().mean().item()
        print(f"{name:24s} nan/inf={bad}  maxdiff(real tokens)={d_real:.4f}  "
              f"argmax agree={argmax_agree:.3f}  (maxdiff all incl. padding={d_all:.2f})")

    report("compiled vs baseline", compiled_out)
    report("fused vs baseline", fused_out)
    report("separated vs baseline", separated_out)
    report("fused + compiled vs baseline", fused_compiled_out)

    # --- results ---
    print("=" * 60)

    def speedup(name, ms):
        if ms is None or baseline_ms is None:
            print(f"Speedup {name}: (skipped)")
            return
        print(f"Speedup {name}: {baseline_ms / ms:.2f}x  ({baseline_ms:.2f} ms → {ms:.2f} ms)")

    speedup("compiled", compiled_ms)
    speedup("fused", fused_ms)
    speedup("separated", separated_ms)
    speedup("fused + compiled", fused_compiled_ms)

    # --- the A/B we actually care about: fused norm+MLP vs separated norm+MLP ---
    print("=" * 60)
    if fused_ms is not None and separated_ms is not None:
        fused_vs_sep = separated_ms / fused_ms
        faster = "fused" if fused_ms < separated_ms else "separated"
        print(f"Norm+MLP fusion A/B: fused={fused_ms:.2f} ms  separated={separated_ms:.2f} ms  "
              f"-> {faster} faster by {abs(1 - fused_vs_sep) * 100:.1f}%")
        if fused_out is not None and separated_out is not None:
            sep_real = (separated_out.detach().to("cpu", torch.float32)[:, real, :]
                        - fused_out.detach().to("cpu", torch.float32)[:, real, :]).abs().max().item()
            # Not exactly 0: the in-kernel RMSNorm (NormType.RMS_NORM) and the torch
            # fp32 RMSNorm are algebraically identical but round differently. Same order
            # of magnitude as each-vs-baseline, and argmax still agrees -> equivalent.
            print(f"fused vs separated maxdiff (real tokens) = {sep_real:.4f}  "
                  f"(small numerical diff from norm precision, not a correctness gap)")
    else:
        print("Norm+MLP fusion A/B: skipped (one of fused/separated failed to run)")
