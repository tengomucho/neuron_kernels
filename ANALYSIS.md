> ⚠️ **SUPERSEDED / PARTLY INCORRECT.** The conclusion below ("state corruption in
> `nki_attention_cte`") is wrong. The verified root cause is that **`nki_qkv`'s prefill
> path `qkv_cte` (seqlen > 96) produces wrong/non-deterministic output**;
> `nki_attention_cte` is correct. See **`ISSUE.md`** for the verified analysis. This
> file is retained as a record of the investigation.
>
> ✅ **FIXED.** The kernel has been updated: `NeuronQwen3Attention` in
> `qwen3-neuron-kernels/build/torch-neuron/__init__.py` now computes the QKV
> projection + QK-norm + RoPE in PyTorch (bypassing the broken `nki_qkv`) and feeds
> the result to the working `nki_attention_cte` and `nki_output_projection_cte`
> kernels. Verified on Trn2: no NaN at seqlen 128/256/512 across repeated runs, with
> predictions matching the eager baseline exactly (argmax agreement = 1.000 on real
> tokens). `fused_qwen_example.py` is wired to load this updated kernel locally via
> `use_local_kernel=True`. The recommendations below (e.g. "DO NOT USE attention
> fusion", "use MLP fusion only") are therefore obsolete.

# Analysis: Fused Qwen3 Kernel Output Differences

## Executive Summary

The fused kernel implementation for Qwen3-0.6B using `torch_neuronx` produces NaN values due to a **state corruption bug in the NKI attention kernel** (`nki_qkv` and/or `nki_attention_cte`). The issue is NOT related to QK normalization specifically.

**Key Finding:** The first inference works correctly, but subsequent inferences produce NaN. This bug affects the attention fusion kernel and persists across fresh model loads, indicating corruption in the NKI runtime state.

## Test Results

### Sequence Length Dependency

The bug is triggered based on sequence length and consecutive inference count:

| Sequence Length | First Run | Second Run | Status |
|-----------------|-----------|------------|--------|
| 128 | ✅ OK | ✅ OK | Stable |
| 192 | ✅ OK | ❌ NaN | Fails on 2nd run |
| 256 | ❌ NaN | ❌ NaN | Fails immediately |
| 512 | ❌ NaN | ❌ NaN | Fails immediately |
| 1024 | ❌ NaN | ❌ NaN | Fails immediately |

### Configuration Comparison

| Configuration | Seq 128 | Seq 256 | Seq 1024 | Status |
|--------------|---------|---------|----------|--------|
| Baseline (no fusion) | ✅ | ✅ | ✅ | Stable |
| Compiled only | ✅ | ✅ | ✅ | Stable |
| MLP fusion only | ✅ | ✅ | ✅ | Stable |
| Attention fusion only | ✅* | ❌ | ❌ | **Broken** |
| Both fusions | ✅ | ❌ | ❌ | **Broken** |

*Seq 128 works but attention output values vary (see below)

### Attention Output Variation (Seq 128, Attention Only)

Even when seq 128 works, attention output values vary between runs:
```
Run 1: attn0 min=-0.1270, max=0.1177 (normal range)
Run 2: attn0 min=-115.0000, max=93.0000 (larger values)
Run 3: attn0 min=0.0000, max=0.0000 (zeros)
Run 4: attn0 min=-143.0000, max=129.0000 (larger values)
Run 5: attn0 min=0.0000, max=0.0000 (zeros)
```

Despite the variation, final logits remain consistent for seq 128. Longer sequences produce NaN.

## Root Cause Analysis

### Confirmed Root Cause

The issue is a **memory-related state corruption bug in the NKI runtime** when used with the model forward path.

**Key Discovery:** The NKI functions (`nki_qkv`, `nki_attention_cte`) work correctly when called directly, but fail when used through the model forward path on the second inference.

**Evidence:**

1. **NKI functions work in isolation**: Calling NKI functions directly with fused model weights works correctly for all runs

2. **Model forward path fails**: Using the full model produces NaN on run 2 with seq > 128

3. **MLP kernel is stable**: `nki_mlp` works correctly with all sequence lengths

4. **Layer 0 produces NaN directly**: Hook testing shows the attention output is already NaN in failed runs

5. **Torch cache clearing doesn't help**: The bug is in NKI runtime state, not PyTorch state

6. **The bug is in persistent Neuron runtime state**: Fresh subprocess for each inference works correctly - the bug is NOT in the NKI functions or kernel code, but in process-level state corruption

7. **Reset functions don't help**: Tested torch.neuron.empty_cache(), torch_neuronx.reset_dynamo_metrics(), torch_neuronx.kernel_registry.reset_table() - none prevent the bug

8. **Fresh subprocess confirms**: All consecutive inferences work when run in separate processes

### What Happens

1. First inference with small seq: NKI kernels compile and execute (may return varying values but compensates)
2. Inference with larger seq or second run: Memory state corruption causes NaN
3. The corruption persists in NKI runtime state until process restart

### Hypotheses Tested

| Hypothesis | Test Result | Conclusion |
|------------|-------------|------------|
| QK norm numerical instability | ❌ Not the cause | Disabling QK norm didn't fix the issue |
| NKI SBUF memory conflict | ❓ Partial | MLP kernel is stable, so conflict is not general |
| NKI state corruption | ✅ **Confirmed** | First run works, subsequent fail |
| Fused RoPE + QK norm interaction | ❌ Not the cause | Issue occurs even with these disabled |

## Impact

- **MLP fusion**: Safe to use, works correctly
- **Attention fusion**: **DO NOT USE** - produces NaN after first run
- **Both fusions**: **DO NOT USE** - same issue as attention only

## Options to Resolve the Issue

### Option 1: Use Only MLP Fusion ✅ RECOMMENDED

The MLP fusion (`NeuronRMSNormMLP`) works correctly with all sequence lengths and provides a ~1.86x speedup. Use only this fusion:

```python
def get_kernel_config():
    return KernelConfig({
        (('Qwen3RMSNorm', 'model.layers.*.post_attention_layernorm'), 
         ('Qwen3MLP', 'model.layers.*.mlp')): 'michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP',
    })
```

**Status:** ✅ Tested and confirmed working with seq 128-1024
**Pros:** Safe, works correctly, provides speedup
**Cons:** No attention fusion benefit

### Option 2: Run Each Inference in a Separate Process

Since fresh subprocess for each inference works correctly, you can use a subprocess/multiprocessing approach to avoid the state corruption:

```python
import subprocess
import sys

def run_inference_in_subprocess(inputs):
    script = '''
import os
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig

# ... inference code here ...
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True)
    return result
```

**Status:** ⚠️ Works but has overhead for process creation
**Pros:** Attention fusion works correctly
**Cons:** High overhead, not suitable for real-time inference

The MLP fusion (`NeuronRMSNormMLP`) works correctly with all sequence lengths and provides a ~1.86x speedup. Use only this fusion:

```python
def get_kernel_config():
    return KernelConfig({
        (('Qwen3RMSNorm', 'model.layers.*.post_attention_layernorm'), 
         ('Qwen3MLP', 'model.layers.*.mlp')): 'michaelbenayoun/qwen3-neuron-kernels:NeuronRMSNormMLP',
    })
```

**Status:** ✅ Tested and confirmed working with seq 128-1024
**Pros:** Safe, works correctly, provides speedup
**Cons:** No attention fusion benefit

### Option 3: Keep Sequence Length ≤ 128 ⚠️ PARTIAL WORKAROUND

For short sequences (≤ 128 tokens), the attention fusion works correctly for multiple consecutive inferences. However, output values vary between runs and may not be deterministic.

**Status:** ⚠️ Works for short sequences only
**Pros:** Can use attention fusion for short inputs
**Cons:** Not reliable for longer sequences, non-deterministic outputs

### Option 4: Implement Attention Kernel Without NKI

Replace the NKI attention kernel with a standard PyTorch implementation:

```python
def attention_forward(self, hidden_states, position_embeddings, ...):
    # Use standard PyTorch attention instead of NKI
    qkv = torch.matmul(hidden_states, self.fused_qkv_proj.weight.T)
    # Apply QK norm manually
    # Apply RoPE manually
    # Use torch.nn.functional.scaled_dot_product_attention
```

**Status:** ⚠️ Not tested - requires significant changes
**Pros:** Bypasses buggy NKI attention functions
**Cons:** Loses performance benefit of NKI attention fusion

### Option 5: File Bug Report with AWS Neuron Team ✅ RECOMMENDED

Document the issue thoroughly for potential fixes:

**Information to include in bug report:**
- NKI version: 0.4.0b4+25404641739.g5cabf9dc
- torch-neuronx: 2.9.0.2.12.22436+0f1dac25
- Platform: AWS Trainium 2 (actual hardware)
- Affected functions: `nki_qkv`, `nki_attention_cte`
- Stable functions: `nki_mlp`
- Trigger conditions:
  - Sequence length ≥ 192 tokens (immediate failure)
  - OR second consecutive inference with seq > 128 tokens
- Root cause: Memory-related state corruption in NKI attention kernel

**Minimal reproduction case:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=KernelConfig({
        'Qwen3Attention': 'michaelbenayoun/qwen3-neuron-kernels:NeuronQwen3Attention',
    }),
    device_map='neuron',
)
# First run - works
result1 = model(inputs)  # ✅ Works
# Second run - NaN (for seq > 128)
result2 = model(inputs)  # ❌ NaN
```

## Recommended Next Steps

1. **Immediate:** Use MLP fusion only (Option 1) for production workloads
2. **Short-term:** File bug report with AWS Neuron team (Option 4) with sequence length threshold details
3. **Medium-term:** If attention fusion is required, implement with standard PyTorch (Option 3)
4. **Long-term:** Wait for torch-neuronx fix from AWS Neuron team

## Additional Findings

- **Attention output non-deterministic**: Even when seq 128 works, attention output values vary between runs (can be zeros, large values, or normal range)
- **MLP is deterministic**: MLP fusion produces consistent outputs across all runs
- **Memory threshold**: ~128 tokens is the safe threshold for attention fusion stability
- **NKI functions work in isolation**: Direct calls to `nki_qkv` and `nki_attention_cte` work correctly - the bug is in model integration
- **Persistent Neuron runtime state corruption**: The bug is in process-level state, not in NKI functions or kernel code. Fresh subprocess for each inference works correctly.
- **Not a QK norm issue**: Disabling QK norm does not fix the bug
- **Not a RoPE issue**: Disabling RoPE does not fix the bug

## Files to Modify

- `qwen3-neuron-kernels/build/torch-neuron/__init__.py` - The fused kernel implementation

## Environment Information

- Platform: AWS Trainium 2 (actual hardware)
- NKI version: 0.4.0b4+25404641739.g5cabf9dc
- torch-neuronx: 2.9.0.2.12.22436+0f1dac25 (preview package)
- transformers: 4.56.2
- Model: Qwen/Qwen3-0.6B (28 layers, 16 heads, head_dim=128)

**Note:** `torch_neuronx` is a preview package with a confirmed bug in the NKI attention kernel. No workaround is available for attention fusion.