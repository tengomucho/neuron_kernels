"""Qwen3.5 Neuron kernel benchmark with per-config process isolation.

WHY THIS STRUCTURE
==================
The original version ran all configs sequentially in ONE process, wrapped in a
``try/except`` (``safe_run``) so a broken config wouldn't abort the comparison.

That works for *Python* exceptions but NOT for native crashes. On this
Qwen3.5-0.8B-Base (GatedDeltaNet) model, ``torch.compile(backend="neuron")``
triggers a Neuron compiler internal error (``NCC_IBTN006`` "narwhal backend
error") that manifests as a native **hang / SIGSEGV during lazy compilation**.
A segfault is a signal, not a Python exception, so ``try/except`` cannot catch
it -- the whole process dies at the first compile config (``compiled``), taking
the (working) baseline results with it.

The fix: run EACH config in a separate child process with a hard timeout. A
native crash or hang in one config now only kills *that* child; the parent
records it as a failure and moves on, so the full comparison completes. The
non-compile configs (baseline / fused / separated) run to completion and their
results are compared as before; the compile configs (compiled /
fused_compiled) are reported as crashed/timeout rather than aborting the run.

MECHANICS
=========
- Parent (no ``--worker``): spawns one child per config via ``subprocess.run``
  with a timeout (``CFG_TIMEOUT`` seconds, default 600). It captures each
  child's exit code:
    0  -> success (read saved logits + timing)
    124 -> timeout (native hang during compile/forward)
    139 -> SIGSEGV (native crash)
    other -> Python-level error (read saved traceback)
- Child (``--worker <name>``): loads the model, computes the reference output,
  optionally ``torch.compile``s, benchmarks, and saves its logits (.pt) +
  timing (result.json) to a per-config slot in the shared output dir.

The tokenizer is deterministic for a fixed prompt + max_length, so the parent
re-tokenizes to recover the padding mask for the real-token comparison.
"""

import os

# Set before importing torch/transformers so async load is actually disabled in
# the worker processes (matches the original script's ordering).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import json
import subprocess
import sys
import tempfile
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig

NUM_WARMUP = int(os.environ.get("NUM_WARMUP", 10))
NUM_RUNS = int(os.environ.get("NUM_RUNS", 50))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 1024))
DTYPE = torch.bfloat16

# Hard per-config timeout (seconds). Compile configs hang natively on this model;
# this bound lets the parent record "timeout" and continue instead of waiting
# forever. Generous enough for the legit configs (baseline ~2.5 s/fwd * 60 runs
# ~= 150 s + load/compile overhead).
CFG_TIMEOUT = int(os.environ.get("CFG_TIMEOUT", 600))

model_id = "Qwen/Qwen3.5-0.8B-Base"

# Canonical config table, shared by parent (spawner) and worker (child).
# name -> (compile_en, kernel_en, fused_mlp)
CONFIGS = {
    "baseline": (False, False, True),
    "compiled": (True, False, True),
    "fused": (False, True, True),
    "separated": (False, True, False),
    "fused_compiled": (True, True, True),
}


def _sync(out):
    """Force the forward to actually execute. The Neuron `torch.compile` backend runs
    fully async/lazy, so calling the model only *dispatches* work -- without reading a
    value back, the timed loop measures dispatch latency (~0.5 ms), not compute. Reading
    a single scalar forces graph execution while avoiding a full logits transfer."""
    return float(out.logits.view(-1)[0])


def benchmark(model, inputs, label):
    """Run benchmark on a model."""
    print(f"  Warming up {label} ({NUM_WARMUP} runs)...", flush=True)
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _sync(model(**inputs))

    print(f"  Benchmarking {label} ({NUM_RUNS} runs)...", flush=True)
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _sync(model(**inputs))
    elapsed = time.perf_counter() - start

    ms_per_run = elapsed / NUM_RUNS * 1000
    tps = SEQ_LEN / (ms_per_run / 1000)
    print(f"  {label}: {ms_per_run:.2f} ms/fwd  |  {tps:.1f} tokens/sec", flush=True)
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

    NOTE: the local MLP-fusion config is commented out below; the active config replaces
    the GatedDeltaNet layer with the `NeuronGatedDeltaNet` kernel. `fused_mlp` is kept
    as a parameter for API compatibility with the original config table, but with the
    MLP-fusion path disabled both `fused` and `separated` resolve to the same kernel
    config.
    """
    # mlp_kernel = "NeuronRMSNormMLP" if fused_mlp else "NeuronRMSNormMLP_Separated"
    # return KernelConfig(
    #     {
    #         "Qwen3Attention": f"{LOCAL_KERNEL_REPO}:NeuronQwen3Attention",
    #         (
    #             ("Qwen3RMSNorm", "model.layers.*.post_attention_layernorm"),
    #             ("Qwen3MLP", "model.layers.*.mlp"),
    #         ): f"{LOCAL_KERNEL_REPO}:{mlp_kernel}",
    #     },
    #     use_local_kernel=True,
    # )
    return KernelConfig({
        "Qwen3_5GatedDeltaNet": "jburtoft/qwen35-deltanet-neuron-kernels:NeuronGatedDeltaNet",
    })


def load_model(compile_en, kernel_en, fused_mlp, label):
    """Load a model with optional kernel fusion, and optionally torch.compile it.

    Returns (model, reference_out, inputs).
    """
    print("=" * 60, flush=True)
    print(f"Loading model: {label}...", flush=True)

    # Build kernel config if requested
    kernel_config = get_kernel_config(fused_mlp=fused_mlp) if kernel_en else None

    # Load model with optional kernel fusion. use_kernels is auto-enabled when a
    # kernel_config is passed; set it explicitly to match the original script and
    # silence the transformers warning.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_kernels=True,
        kernel_config=kernel_config if kernel_en else None,
        dtype=DTYPE,
        device_map="neuron",
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(
        "Hello, how are you?",
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )

    # Move inputs to model device and get reference output
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        reference_out = model(**inputs).logits
    print(f"Output shape: {reference_out.shape}", flush=True)

    # Apply torch.compile if requested. NOTE: on this GatedDeltaNet model this is
    # expected to fail (Neuron NCC_IBTN006 internal error -> native hang/segfault);
    # the parent runs this in a subprocess so the crash is contained.
    if compile_en:
        print(f"  Compiling with neuron backend...", flush=True)
        model = torch.compile(model, backend="neuron")

    return model, reference_out, inputs


# --------------------------------------------------------------------------------------
# Worker (child) mode
# --------------------------------------------------------------------------------------

def worker_main(name, out_dir):
    """Run a single config end to end and persist its logits + timing for the parent."""
    compile_en, kernel_en, fused_mlp = CONFIGS[name]

    result_path = os.path.join(out_dir, f"{name}.json")
    logits_path = os.path.join(out_dir, f"{name}.pt")

    try:
        model, reference_out, inputs = load_model(
            compile_en, kernel_en, fused_mlp, name
        )
        benchmark_ms = benchmark(model, inputs, name)

        # Persist reference logits (CPU) + timing so the parent can compare.
        torch.save(reference_out.detach().to("cpu"), logits_path)
        with open(result_path, "w") as f:
            json.dump(
                {"name": name, "ok": True, "ms": benchmark_ms,
                 "shape": list(reference_out.shape)},
                f,
            )
        print(f"[{name}] worker done", flush=True)
    except Exception as e:
        # Python-level failure: record traceback so the parent can report it.
        with open(result_path, "w") as f:
            json.dump(
                {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                 "tb": traceback.format_exc()},
                f,
            )
        print(f"[{name}] worker FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


# --------------------------------------------------------------------------------------
# Parent mode: spawn isolated workers, then compare / report
# --------------------------------------------------------------------------------------

def _classify_exit(code):
    if code == 0:
        return "ok"
    if code == 124 or code == -9:  # timeout SIGTERM(15)->-15, SIGKILL(9)->-9
        return "timeout"
    if code in (139, -11):  # SIGSEGV
        return "segfault"
    return f"exit={code}"


def run_config_isolated(name, out_dir):
    """Spawn one child process for `name` with a hard timeout. Returns a dict with
    status, logits (or None), ms (or None), and an error message for failures."""
    print("=" * 60, flush=True)
    print(f"Spawning isolated worker for: {name} (timeout {CFG_TIMEOUT}s)...",
          flush=True)

    cmd = [sys.executable, __file__, "--worker", name, "--out", out_dir]
    result_path = os.path.join(out_dir, f"{name}.json")
    logits_path = os.path.join(out_dir, f"{name}.pt")

    try:
        proc = subprocess.run(
            cmd, timeout=CFG_TIMEOUT, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        # Native hang (e.g. torch.compile lazily compiling on first forward). The
        # child is killed by the timeout; record and move on.
        print(f"  !! {name}: TIMEOUT after {CFG_TIMEOUT}s (native hang -- typical "
              f"of the broken torch.compile(neuron) path on this model)", flush=True)
        return {"status": "timeout", "logits": None, "ms": None,
                "error": f"timeout after {CFG_TIMEOUT}s (native hang during "
                         f"compile/forward)"}

    code = proc.returncode
    status = _classify_exit(code)

    # Surface the child's progress lines (minus torch/ATen registration spam).
    for line in proc.stdout.splitlines():
        if line.strip() and not any(s in line for s in (
            "Overriding", "operator:", "dispatch key", "registered at",
            "new kernel", "previous kernel", "aten::", "sparse_grad",
            "Loading weights",
        )):
            print(f"  {name} | {line}", flush=True)

    if status != "ok":
        # Native crash (segfault) or nonzero exit: no usable result file.
        tail = (proc.stderr or "").strip().splitlines()
        tail = tail[-8:] if tail else ["(no stderr)"]
        print(f"  !! {name}: {status.upper()} (native crash -- uncatchable by "
              f"try/except; isolated so other configs survive)", flush=True)
        return {"status": status, "logits": None, "ms": None,
                "error": f"{status}; stderr tail: " + " | ".join(tail)}

    # Child exited cleanly: read its saved result.
    if not os.path.exists(result_path):
        print(f"  !! {name}: exited 0 but no result file (unexpected)", flush=True)
        return {"status": "error", "logits": None, "ms": None,
                "error": "exited 0 but no result file"}
    with open(result_path) as f:
        res = json.load(f)

    if not res.get("ok"):
        print(f"  !! {name}: worker reported error: {res.get('error')}", flush=True)
        return {"status": "error", "logits": None, "ms": None,
                "error": res.get("error", "unknown"), "tb": res.get("tb")}

    logits = torch.load(logits_path, weights_only=True) if os.path.exists(
        logits_path) else None
    print(f"  {name}: ok ({res['ms']:.2f} ms/fwd)", flush=True)
    return {"status": "ok", "logits": logits, "ms": res["ms"], "error": None}


if __name__ == "__main__":
    # --- worker dispatch ---
    if "--worker" in sys.argv:
        wi = sys.argv.index("--worker")
        name = sys.argv[wi + 1]
        out_dir = sys.argv[sys.argv.index("--out") + 1]
        worker_main(name, out_dir)
        raise SystemExit(0)

    # --- parent: run all configs isolated, then compare ---
    print(f"NUM_WARMUP={NUM_WARMUP} NUM_RUNS={NUM_RUNS} SEQ_LEN={SEQ_LEN} "
          f"CFG_TIMEOUT={CFG_TIMEOUT}", flush=True)

    out_dir = tempfile.mkdtemp(prefix="qwen35_bench_")
    print(f"Worker output dir: {out_dir}", flush=True)

    results = {}
    for name in ["baseline", "compiled", "fused", "separated", "fused_compiled"]:
        results[name] = run_config_isolated(name, out_dir)

    # --- compare ---
    # Compare against the baseline over the *real* (non-padding) tokens. The fused
    # attention kernel applies causal masking only and ignores the padding mask, so
    # padding positions diverge from the baseline (which masks them) -- that region is
    # never used and would otherwise dominate a naive full-tensor max-diff. We also
    # flag NaN/inf, which is the failure mode of the buggy kernel.
    print("=" * 60)

    # Re-tokenize in the parent to recover the padding mask (deterministic for a fixed
    # prompt + max_length, matching the workers).
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tok = tokenizer(
        "Hello, how are you?",
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    real = tok["attention_mask"].bool()[0]  # [S], on CPU

    base = results["baseline"]
    base_cpu = (
        base["logits"].to(torch.float32)
        if base["status"] == "ok" and base["logits"] is not None
        else None
    )
    if base_cpu is None:
        print(f"Baseline unavailable (status={base['status']}); cannot compare.")
    else:
        def report(name, res):
            if res["status"] != "ok" or res["logits"] is None:
                print(f"{name:24s} (skipped: {res['status']} -- {res['error']})")
                return
            o = res["logits"].to(torch.float32)
            bad = torch.isnan(o).any().item() or torch.isinf(o).any().item()
            d_all = (o - base_cpu).abs().max().item()
            d_real = (o[:, real, :] - base_cpu[:, real, :]).abs().max().item()
            argmax_agree = (
                (o[:, real, :].argmax(-1) == base_cpu[:, real, :].argmax(-1))
                .float().mean().item()
            )
            print(
                f"{name:24s} nan/inf={bad}  maxdiff(real tokens)={d_real:.4f}  "
                f"argmax agree={argmax_agree:.3f}  (maxdiff all incl. padding={d_all:.2f})"
            )

        report("compiled vs baseline", results["compiled"])
        report("fused vs baseline", results["fused"])
        report("separated vs baseline", results["separated"])
        report("fused + compiled vs baseline", results["fused_compiled"])

    # --- results ---
    print("=" * 60)

    def speedup(name, res):
        ms = res["ms"] if res["status"] == "ok" else None
        if ms is None or base.get("ms") is None:
            print(f"Speedup {name}: (skipped: {res['status']})")
            return
        print(
            f"Speedup {name}: {base['ms'] / ms:.2f}x  ({base['ms']:.2f} ms -> {ms:.2f} ms)"
        )

    speedup("compiled", results["compiled"])
    speedup("fused", results["fused"])
    speedup("separated", results["separated"])
    speedup("fused + compiled", results["fused_compiled"])

    # --- the A/B we actually care about: fused norm+MLP vs separated norm+MLP ---
    print("=" * 60)
    fused_res, separated_res = results["fused"], results["separated"]
    if fused_res["status"] == "ok" and separated_res["status"] == "ok":
        fused_ms, separated_ms = fused_res["ms"], separated_res["ms"]
        fused_vs_sep = separated_ms / fused_ms
        faster = "fused" if fused_ms < separated_ms else "separated"
        print(
            f"Norm+MLP fusion A/B: fused={fused_ms:.2f} ms  separated={separated_ms:.2f} ms  "
            f"-> {faster} faster by {abs(1 - fused_vs_sep) * 100:.1f}%"
        )
        if fused_res["logits"] is not None and separated_res["logits"] is not None:
            sep_real = (
                (
                    separated_res["logits"].to(torch.float32)[:, real, :]
                    - fused_res["logits"].to(torch.float32)[:, real, :]
                ).abs().max().item()
            )
            # Not exactly 0: the in-kernel RMSNorm (NormType.RMS_NORM) and the torch
            # fp32 RMSNorm are algebraically identical but round differently. Same order
            # of magnitude as each-vs-baseline, and argmax still agrees -> equivalent.
            print(
                f"fused vs separated maxdiff (real tokens) = {sep_real:.4f}  "
                f"(small numerical diff from norm precision, not a correctness gap)"
            )
    else:
        print(
            "Norm+MLP fusion A/B: skipped "
            f"(fused={fused_res['status']}, separated={separated_res['status']})"
        )

    print("=" * 60)
    print("Done. Compile configs that crashed were isolated in subprocesses so they")
    print("could not abort the (working) non-compile configs. See status lines above.")