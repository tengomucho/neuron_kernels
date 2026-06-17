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

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(
    "Hello, how are you?",
    return_tensors="pt",
    padding="max_length",
    max_length=SEQ_LEN,
    truncation=True,
)

# --- baseline: plain model, no fusion ---
print("=" * 60)
print("Loading baseline model (no fusion)...")
baseline = AutoModelForCausalLM.from_pretrained(
    model_id, use_kernels=True, torch_dtype=DTYPE
)
baseline = baseline.to("neuron")
baseline.eval()
inputs = {k: v.to(baseline.device) for k, v in inputs.items()}

with torch.no_grad():
    baseline_out = baseline(**inputs).logits
print("Baseline output shape:", baseline_out.shape)

# --- fused model ---
print("=" * 60)
print("Loading fused model...")

# kernel_repo_id = "michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP"
# kernel_config = KernelConfig(
#     {
#         (
#             ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
#             ("Qwen3MLP",     "model.layers.*.mlp"),
#         ): kernel_repo_id,
#     },
# )

kernel_config = KernelConfig(
    {
        "Qwen3Attention": "michaelbenayoun/qwen3-neuron-kernels:NeuronQwen3Attention",
        (
            ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
            ("Qwen3MLP", "model.layers.*.mlp"),
        ): "michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP",
    },
)

fused_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_kernels=True,
    kernel_config=kernel_config,
    torch_dtype=DTYPE,
    device_map="neuron",
)
fused_model.eval()
print(fused_model)

with torch.no_grad():
    fused_out = fused_model(**inputs).logits
print("Fused output shape:", fused_out.shape)

# --- compiled model ---

compiled_model = torch.compile(baseline, backend="neuron")
with torch.no_grad():
    compiled_out = compiled_model(**inputs).logits
print("Compiled output shape:", compiled_out.shape)


# --- compare ---
print("=" * 60)
print("Max diff fused vs baseline:   ", (fused_out - baseline_out).abs().max().item())
print("Max diff compiled vs baseline:", (compiled_out - baseline_out).abs().max().item())


# --- benchmark ---
def benchmark(model, inputs, label):
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


print("=" * 60)
print("Benchmarking...")
baseline_ms = benchmark(baseline, inputs, "baseline")
fused_ms = benchmark(fused_model, inputs, "fused")
compiled_ms = benchmark(compiled_model, inputs, "compiled")

print("=" * 60)
print(
    f"Speedup fused: {baseline_ms / fused_ms:.2f}x  ({baseline_ms:.2f} ms → {fused_ms:.2f} ms)"
)
print(
    f"Speedup compiled: {baseline_ms / compiled_ms:.2f}x  ({baseline_ms:.2f} ms → {compiled_ms:.2f} ms)"
)
