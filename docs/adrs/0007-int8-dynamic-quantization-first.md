# ADR-0007 — int8 dynamic quantization as the first optimization pass

**Status:** Accepted · **Date:** 2026-08-12

## Context

Once the fp32 embedding matcher works, several optimizations are available: dynamic int8
quantization, static int8 with a calibration set, int4, pruning, distillation, operator
fusion, and swapping to a smaller model.

Applying several at once would make the result unattributable. If a stacked arm is
faster, smaller and less accurate, nothing tells you which change bought what — and a
characterised trade-off is the entire contribution here. **Each optimization has to be
isolated to be measured**, so v1 does exactly one, properly controlled, and later
optimizations become additional arms rather than a bundle.

## Decision

**Dynamic int8 quantization, via `onnxruntime.quantization`, as the first and only v1
optimization pass.**

## Consequences

- It requires **no calibration dataset**, which matters because our fixtures are
  synthetic (ADR-0004) — calibrating on synthetic data would produce a quantization
  tuned to a distribution that does not exist.
- It moves all four measured axes at once: model size on disk, memory, latency, and
  (possibly) accuracy. That is the full trade-off surface in a single change, which is
  exactly what the project set out to characterise.
- The change is small and surgical, which keeps it attributable: the two arms share a
  source model, tokenizer, pooling, runtime version, inputs and measurement method, so
  quantization is the only deliberate difference. It is **not** true that they share a code
  path or kernels — quantization rewrites the graph and dispatches quantized operators,
  which is precisely how a speedup would arise. Which operators were actually quantized is
  therefore part of the result, not an implementation detail.
- It is the standard first move, so the result is directly comparable to what other
  practitioners will have done.
- **The accuracy cost may exceed the latency gain.** That outcome gets published as
  prominently as the favourable one. A negative result with numbers behind it is a
  result, and it is more useful than the absence of one.
- Deeper Arm-specific work — kernel-level tuning, alternative runtimes, SME/SVE paths —
  is deferred. The README must not imply optimization was exhausted.

## Alternatives rejected

**Static int8 with calibration.** Usually more accurate, but needs a representative
calibration set. See above — with synthetic fixtures, calibration would encode a fiction.

**int4.** Larger size and latency win, materially worse accuracy risk on short strings,
and less mature tooling for this model class. Candidate for a follow-up once int8 is
measured.

**Distillation or fine-tuning.** Would likely beat everything here, and is explicitly a
non-goal (PRD NG3). It confounds the measurement: a fine-tuned model wins because of the
fine-tuning, telling you nothing about the quantization trade.

**Swapping to a smaller model.** A model-selection question, not an optimization one.
Belongs in a separate arm with its own controls.
