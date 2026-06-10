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

os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig


NUM_WARMUP = 10
NUM_RUNS = 50
SEQ_LEN = 1024
DTYPE = torch.bfloat16  # switch to torch.float32 to check numerical precision

# model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("Hello, how are you?", return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)

# --- baseline: plain model, no fusion ---
print("=" * 60)
print("Loading baseline model (no fusion)...")
baseline = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=False, torch_dtype=DTYPE)
baseline = baseline.to("neuron")
baseline.eval()
inputs = {k: v.to(baseline.device) for k, v in inputs.items()}

with torch.no_grad():
    baseline_out = baseline(**inputs).logits
print("Baseline output shape:", baseline_out.shape)

# --- fused model ---
print("=" * 60)
print("Loading fused model...")

kernel_repo_id = "michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP"
kernel_config = KernelConfig(
    {
        (
            ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
            ("Qwen3MLP",     "model.layers.*.mlp"),
        ): kernel_repo_id,
    },
)

fused_model = AutoModelForCausalLM.from_pretrained(
    model_id, use_kernels=True, kernel_config=kernel_config, torch_dtype=DTYPE, device_map="neuron"
)
fused_model = fused_model.to("neuron")
fused_model.eval()
print(fused_model)

with torch.no_grad():
    fused_out = fused_model(**inputs).logits
print("Fused output shape:", fused_out.shape)

# --- weight check ---
print("=" * 60)
print("Checking weights match between baseline and fused model...")
for i, (bl_layer, fused_layer) in enumerate(zip(baseline.model.layers, fused_model.model.layers)):
    bl_norm   = bl_layer.post_attention_layernorm
    fused_mod = fused_layer.post_attention_layernorm

    diffs = {
        "norm_weight": (bl_norm.weight.cpu()                - fused_mod.norm_weight.cpu()).abs().max().item(),
        "gate_proj":   (bl_layer.mlp.gate_proj.weight.cpu() - fused_mod.gate_proj.weight.cpu()).abs().max().item(),
        "up_proj":     (bl_layer.mlp.up_proj.weight.cpu()   - fused_mod.up_proj.weight.cpu()).abs().max().item(),
        "down_proj":   (bl_layer.mlp.down_proj.weight.cpu() - fused_mod.down_proj.weight.cpu()).abs().max().item(),
    }
    any_mismatch = any(v > 0 for v in diffs.values())
    if any_mismatch:
        print(f"  Layer {i} MISMATCH: { {k: v for k, v in diffs.items() if v > 0} }")
print("Weight check done.")

# --- compare ---
print("=" * 60)
print("Max diff fused vs baseline:", (fused_out - baseline_out).abs().max().item())

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

print("=" * 60)
print(f"Speedup: {baseline_ms / fused_ms:.2f}x  ({baseline_ms:.2f} ms → {fused_ms:.2f} ms)")
