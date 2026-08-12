# ADR-0002 — ONNX Runtime as the on-device inference runtime

**Status:** Accepted · **Date:** 2026-08-12

## Context

The embedding matcher needs a runtime that executes a sentence-embedding model locally on
Arm64 client hardware, supports int8 quantization, and installs without a toolchain
build. Candidates: ONNX Runtime, ExecuTorch, LiteRT / TensorFlow Lite, llama.cpp, and
PyTorch directly.

Three constraints decide it:

1. The target is a **client device**, not a server.
2. The benchmark must be **reproducible by a reader on their own machine**. A measurement
   nobody else can re-run is an anecdote, so anything requiring a bespoke toolchain build
   is disqualified regardless of its merits.
3. **Quantization must be the only deliberate transformation** between the fp32 and int8
   arms. Not "the same kernels" — quantization necessarily rewrites the graph and
   dispatches quantized operators, and that kernel substitution *is* the mechanism by which
   any speedup would occur. What must be held constant is everything around it: the same
   source model and revision, tokenizer, pooling, runtime product and version, inputs,
   thread settings and measurement method.

## Decision

**ONNX Runtime, CPU execution provider.**

- Installs from a wheel on Arm64 macOS and Linux with no build step.
- Dynamic int8 quantization is available through `onnxruntime.quantization` without a
  separate toolchain or calibration dataset.
- Sentence-embedding models export to ONNX through well-trodden paths.
- It is a mainstream client-side inference runtime for Arm targets, so the result is
  useful to other people rather than being an artefact of an exotic stack.

## Consequences

- Reproducible by anyone with `pip` and an Arm64 machine — a hard requirement for the
  benchmark to mean anything.
- Measurements reflect ONNX Runtime's CPU kernels specifically. Numbers are reported as
  *"model X under ONNX Runtime version Y"*, never as a universal claim about the model.
- Runtime version is recorded with every run (PRD MR-6); a version bump can move the
  numbers.
- Kernel-level Arm-specific tuning is not directly exercised. Accepted for v1 — a
  runtime-comparison arm is a candidate RFC, not v1 scope.

## Alternatives rejected

**ExecuTorch** — the stronger on-device story and arguably the more idiomatic choice for
mobile Arm targets. Rejected on shape, not quality: its deployment model is ahead-of-time
export to a `.pte` artefact embedded in an application binary. That is right for shipping
an app and wrong for a `pip`-installable harness a reader runs on their own machine, which
constraint 2 makes non-negotiable. Recorded as the leading candidate for a future
runtime-comparison arm, where comparing it against ONNX Runtime would itself be the
measurement rather than a confound.

**llama.cpp / GGML** — excellent Arm kernels, but built around autoregressive LLM
inference. Wrong shape for a small embedding model.

**LiteRT / TFLite** — viable, but the quantization workflow for a PyTorch-origin
sentence embedder is longer than the ONNX path.

**PyTorch directly** — largest install, slowest CPU inference of the options, and no
straightforward int8 dynamic quantization story on Arm. It is the *baseline of
convenience* the optimization is measured against, not the shipping runtime.
