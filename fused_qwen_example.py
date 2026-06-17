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


NUM_WARMUP = 10
NUM_RUNS = 50
SEQ_LEN = 1024
DTYPE = torch.bfloat16

# model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
model_id = "Qwen/Qwen3-0.6B"


def benchmark(model, inputs, label):
    """Run benchmark on a model."""
    print(f"  Warming up {label} ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            model(**inputs)

    print(f"  Benchmarking {label} ({NUM_RUNS} runs)...")
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            model(**inputs)
    elapsed = time.perf_counter() - start

    ms_per_run = elapsed / NUM_RUNS * 1000
    tps = SEQ_LEN / (ms_per_run / 1000)
    print(f"  {label}: {ms_per_run:.2f} ms/fwd  |  {tps:.1f} tokens/sec")
    return ms_per_run


def get_kernel_config():
    """Returns the kernel configuration for Qwen3 model fusion."""
    return KernelConfig(
        {
            "Qwen3Attention": "michaelbenayoun/qwen3-neuron-kernels:NeuronQwen3Attention",
            (
                ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
                ("Qwen3MLP", "model.layers.*.mlp"),
            ): "michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP",
        },
    )


def run_model_benchmark(model_id, inputs, compile_en, kernel_en, label="model"):
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
    kernel_config = get_kernel_config() if kernel_en else None
    
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

    # Run benchmarks for different configurations
    baseline_out, baseline_ms = run_model_benchmark(
        model_id, inputs, compile_en=False, kernel_en=False, label="baseline"
    )

    compiled_out, compiled_ms = run_model_benchmark(
        model_id, inputs, compile_en=True, kernel_en=False, label="compiled"
    )

    fused_out, fused_ms = run_model_benchmark(
        model_id, inputs, compile_en=False, kernel_en=True, label="fused"
    )

    fused_compiled_out, fused_compiled_ms = run_model_benchmark(
        model_id, inputs, compile_en=True, kernel_en=True, label="fused + compiled"
    )

    # --- compare ---
    print("=" * 60)
    print("Max diff compiled vs baseline:", (compiled_out - baseline_out).abs().max().item())
    print("Max diff fused vs baseline:", (fused_out - baseline_out).abs().max().item())
    print("Max diff fused + compiled vs baseline:", (fused_compiled_out - baseline_out).abs().max().item())

    # --- results ---
    print("=" * 60)
    print(f"Speedup compiled: {baseline_ms / compiled_ms:.2f}x  ({baseline_ms:.2f} ms → {compiled_ms:.2f} ms)")
    print(f"Speedup fused: {baseline_ms / fused_ms:.2f}x  ({baseline_ms:.2f} ms → {fused_ms:.2f} ms)")
    print(f"Speedup fused + compiled: {baseline_ms / fused_compiled_ms:.2f}x  ({baseline_ms:.2f} ms → {fused_compiled_ms:.2f} ms)")
