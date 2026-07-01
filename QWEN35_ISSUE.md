# Bug Report: `torch.compile(backend="neuron")` hangs / segfaults on Qwen3.5 GatedDeltaNet

## Severity

**Critical** — the native crash (hang / SIGSEGV) is not catchable by Python
`try/except`, so any script that applies `torch.compile(backend="neuron")` to this
model dies without recovering. Benchmarks or pipelines that run multiple
configurations in one process abort *after* the first compile-attempt config,
discarding all prior results.

## Environment

- Model: `Qwen/Qwen3.5-0.8B-Base` (architecture `Qwen3_5GatedDeltaNet` — a
  hybrid linear-attention / gated DeltaNet model)
- `torch`: 2.9.0
- `torch-neuronx`: 2.9.0.2.12.22436+0f1dac25
- `neuronx-cc`: 2.23.6484.0+3b612583
- `transformers`: 5.12.1
- `kernels`: 0.15.2
- Platform: AWS Trainium2 (`trn2.48xlarge`, LNC=2)
- `flash-linear-attention` (fla): **not installed** → transformers falls back to a
  pure-torch GatedDeltaNet implementation (the fallback warning appears on every
  model load)

## Summary

`torch.compile(model, backend="neuron")` fails during **lazy compilation** (first
forward after calling `torch.compile`). The Neuron compiler — `neuronx-cc`
("narwhal" backend) — produces an **internal error** (`NCC_IBTN006`) that manifests
as either a **hang** (process never returns from the first `model(**inputs)`) or a
**SIGSEGV** (process killed by the OS after several seconds). In both cases the
failure is *native* — a C-level crash, not a Python exception — so `try/except`
wrappers cannot contain it. The whole process dies.

**Scope:** the failure affects **all** compile paths — with or without custom Neuron
kernels (`use_kernels=True` / `use_kernels=False`, with and without the
`NeuronGatedDeltaNet` kernel config). It is the `torch.compile(backend="neuron")`
codepath itself that is broken for this model in this SDK version.

**Non-compile paths work correctly:** eager execution (baseline) and the fused
GatedDeltaNet kernel (`NeuronGatedDeltaNet`) without `torch.compile` both produce
valid outputs with argmax agreement = 1.000 and no NaN / inf.

## Symptom

When `torch.compile(backend="neuron")` is applied, the first forward call (including
warmup in a benchmark loop) either:

- **Hangs** — the process consumes 100% CPU on one core and never returns. It may
  eventually print the Neuron error after a long delay, or it may remain silent
  until killed (`SIGKILL` / timeout → exit 124).

- **Segfaults** — the process exits after several seconds with `SIGSEGV` (exit code
  139 / `-11`). This is the symptom observed in `fused_qwen35_example.py`.

Both behaviours are observed on the same environment; the variance is likely
timing- or runtime-dependent.

In `fused_qwen35_example.py` the compile config (`compiled`) is the **2nd** of five
configs. The first config (`baseline`, eager) runs correctly, but when `compiled`
calls `torch.compile` + warmup the whole process is killed — all later configs
(`fused`, `separated`, `fused_compiled`) are never reached and the baseline results
are lost.

## Error message

When the Neon compiler manages to report the failure before crashing:

```
ERROR:torch_neuronx.neuron_dynamo_backend.backend:
  Execution failed: Compilation error occurred on Neuron for operation=torch_compile;
  error message="COMPILATION FAILED: [INTERNAL_ERROR] [NCC_IBTN006] narwhal backend
  error - Please open a support ticket at
  https://github.com/aws-neuron/aws-neuron-sdk/issues/new. You may also be able to
  obtain more information using the 'XLA_IR_DEBUG' and 'XLA_HLO_DEBUG' environment
  variables."
```

`NCC_IBTN006` is a generic "narwhal backend error" code — the Neuron XLA compiler
(neuronx-cc) failed internally while lowering / compiling the model graph, with no
specific operation named. The error suggests the model's ops (likely the
GatedDeltaNet fallback + hybrid linear-attention ops) are not supported by the
Neuron compiler in this SDK version.

## Reproduction (minimal, single process)

```python
import os; os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", dtype=torch.bfloat16, device_map="neuron",
)
model.eval()
model = torch.compile(model, backend="neuron")        # ← compile happens lazily

inputs = model(**tokenized_inputs)                     # ← hangs / segfaults HERE
```

The failure occurs on the **first call** after `torch.compile` (triggering lazy
compilation). No custom kernels are needed — the crash reproduces with a completely
stock model loaded without `use_kernels`.

## Root cause analysis

### Compile path vs. eager path

| Config (single load, isolated)               | compile | kernel | Result (exit)      |
|---------------------------------------------|---------|--------|--------------------|
| eager (plain model)                         | no      | no     | ✅ works (0)       |
| kernel, no compile (`fused`)                | no      | yes    | ✅ works (0)       |
| compile, `use_kernels=True`, no kernel cfg  | yes     | no     | 💀 NCC_IBTN006 → hang |
| compile, `use_kernels=False` (plain eager)  | yes     | no     | 💀 hangs            |
| compile + kernel (`fused_compiled`)         | yes     | yes    | 💀 hangs            |

- **Non-compile paths** (rows 1–2): run correctly, no NaN, argmax agrees with
  baseline to 1.000. The custom `NeuronGatedDeltaNet` kernel replaces the
  GatedDeltaNet layer and works (1.45× speedup vs. eager baseline).
- **Compile paths** (rows 3–5): **always fail**, regardless of `use_kernels` flag
  or kernel config. The crash is in the Neuron XLA compiler / runtime itself, not
  in any particular model layer or kernel.

### Why a segfault kills the entire process

A SIGSEGV is an OS-level signal (signal 11). Python installs a default signal
handler that terminates the process — it is **not** raised as a Python exception
and is not catchable by `try/except`. As a result, `safe_run`-style wrappers (which
only catch `Exception`) cannot isolate the crash from the rest of the configs.

### Why non-compile paths with the same kernel work

The custom `NeuronGatedDeltaNet` kernel runs on the Neuron runtime **without**
`neuronx-cc` lowering — it bypasses the `narwhal` backend entirely at the graph
level. `torch.compile(backend="neuron")`, by contrast, routes the **entire** model
graph through the Neuron XLA compiler, which encounters an internal error on the
ops present in this model (likely the GatedDeltaNet fallback or derivative hybrid
linear-attention ops).

## Workaround

**Isolate each `torch.compile` config in a separate subprocess** with a hard
timeout, so a native crash / hang in one config cannot kill the parent (and the
already-completed configs). The parent detects the timeout / non-zero exit and
reports the config as failed without aborting the full comparison.

This is the approach implemented in `fused_qwen35_example.py` (see
`run_config_isolated`). The subprocess isolation follows the original `safe_run`
philosophy — "one broken config shouldn't abort the whole comparison" — but makes
it robust to native crashes.

### Trade-offs

- **Cost:** each compile config that hangs will block for the full per-config
  timeout (`CFG_TIMEOUT`, default 600 s). With two compile configs that is ~20 min
  of waiting at default settings. Lower `CFG_TIMEOUT` and/or `NUM_RUNS` for a
  faster turnaround.
- **Memory:** the NKI kernel is re-loaded in each subprocess (no shared model
  cache). Kernel compile caches on disk are reused.
- **Logits transport:** each worker saves its reference logits to a temp file
  (~509 MB bf16). The 3 working configs produce ~1.5 GB transient disk usage,
  cleaned up after the run.

## Recommendations

1. **Production:** use the custom NKI kernel (`NeuronGatedDeltaNet`, no
   `torch.compile`) — it works correctly and gives a 1.45× speedup over eager.
2. **AWS Neuron support ticket:** open an issue at
   https://github.com/aws-neuron/aws-neuron-sdk with the `NCC_IBTN006` error and
   the reproduction script above, referencing the GatedDeltaNet / hybrid
   linear-attention model architecture and torch-neuronx 2.9.0.
3. **Long-term:** when neuronx-cc / torch-neuronx adds GatedDeltaNet compile
   support, test `fused_compiled` (kernel + compile) for additional speedup.

Separately, the `nki_qkv` `qkv_cte` bug described in `ISSUE.md` is a
kernel-internal prefill-path defect (unrelated to `torch.compile`). The
`NeuronGatedDeltaNet` kernel used here replaces the layer entirely and does not
call `nki_qkv`; it is not affected by that bug.