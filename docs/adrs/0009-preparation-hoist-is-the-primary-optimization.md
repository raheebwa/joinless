# ADR-0009 — The preparation hoist is the primary optimization; quantization is secondary

**Status:** Accepted · **Date:** 2026-08-12 · **Refines:** ADR-0007 · **Forces:** RFC-0001 Q2

## Context

ADR-0007 selected int8 dynamic quantization as the first optimization pass. Treating it as
the *only* optimization creates two problems.

**It may not pay.** Dynamic int8 through ONNX Runtime does not reliably reduce latency for
a small encoder on Apple Silicon. Quantize/dequantize nodes add overhead that can exceed
the matmul saving; fp32 throughput and memory bandwidth on M-series parts are strong; and
a 22M-parameter model may already be resident in cache and bound by something other than
weight bandwidth. Smaller-but-slower is a live outcome, not a remote risk.

**It produces no transferable knowledge.** `quantize_dynamic(...)` is one call, applies
unchanged to any ONNX model, and teaches nothing about how *this* workload behaves. Whether
it helps here is worth measuring, but the measurement is the contribution, not the change.

There is a second optimization already implicit in the design that has neither problem, and
it was never named as one. Under grid blocking, each listing is compared against every
candidate in its own cell plus eight neighbours. **A single name therefore participates in
many comparisons.** The naive implementation — the one most people write first — embeds
inside the comparison loop, recomputing the same vector repeatedly. Hoisting embedding out
to once per record, with batched preparation, removes that redundancy entirely.

That is algorithmic, it is specific to how this workload behaves under a blocking scheme,
it is ours, and it cannot fail to be a win.

## Decision

**The preparation hoist is the primary optimization. Quantization is the second,
subordinate one.** Both are measured and reported; the headline is the hoist.

Three consequences follow immediately:

1. **Batched preparation becomes contractual**, not optional. This settles RFC-0001's open
   question 2. Per-record transformer calls are not a representative client implementation
   and would understate every neural arm.
2. **The naive per-comparison path must be implemented and measured as a control.** An
   optimization reported without its unoptimized baseline is an assertion, not a
   measurement. The same matcher runs in both call patterns; only the call pattern differs.
3. Preparation cost and comparison cost are reported separately, never conflated — which
   RFC-0002 already requires, and which this decision makes load-bearing.

## Consequences

- **A measured optimization remains even if int8 does not pay.** Should quantization turn
  out not to help, a real, measured speedup still stands, and the int8 result becomes a
  documented negative result — *the standard recommended optimization does not pay for
  small encoders on this hardware, and here is which operators were and were not
  quantized*. That is useful to other developers rather than an absence.
- It answers "what did you optimize?" with an algorithmic change rather than a flag.
- The magnitude of the hoist's win is a function of average bucket occupancy, so bucket
  density must be reported alongside it. A benchmark on sparse fixtures would understate
  it; one on artificially dense fixtures would inflate it. Report the occupancy
  distribution with the result.
- The classical arms are unaffected by the hoist — token and character comparison are cheap
  enough that per-comparison recomputation costs little. **That asymmetry is itself part of
  the finding:** the neural arms need the optimization to be viable at all, which is a
  statement about deployment cost, not about accuracy.

## Alternatives rejected

**Keep quantization as the sole headline.** Stakes the entire result on one library call
behaving well on hardware where it is known to be unreliable. Single point of failure, and
the measurement teaches less: `quantize_dynamic()` applies unchanged to any ONNX model,
while the hoist is specific to how this workload behaves under blocking.

**Treat the hoist as merely an implementation detail.** It was framed this way in RFC-0001
— correct as engineering, wrong as reporting. An optimization that is not measured against
its absence cannot be claimed, and this one is both the larger effect and the more
defensible contribution.

**Do both and call them equal.** Blurs the narrative. One headline, one supporting result,
both measured.
