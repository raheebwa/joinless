# RFC-0004 — Quantization spike protocol

**Status:** Draft · **Date:** 2026-08-12 · **Serves:** ADR-0007, ADR-0009

## Summary

A bounded, scripted feasibility check on int8 dynamic quantization, run **before** the
resolver and benchmark harness are built, with an explicit abort condition.

## Motivation

ADR-0007 selects dynamic int8 quantization and ADR-0009 demotes it to the secondary
optimization, but neither establishes that it works here at all. Several things can be true
and are not knowable by reading documentation:

- the quantized graph may fail to load, or expose different input or output signatures;
- ONNX Runtime may quantize only a subset of operators, leaving the heavy matmuls in fp32;
- quantize/dequantize overhead may exceed the matmul saving, making the int8 graph
  **slower** on this hardware;
- resident memory may not fall even when on-disk size does, if the runtime dequantizes to
  fp32 buffers at load;
- output similarity may degrade enough to change matching behaviour materially.

Building a resolver, fixture generator, evaluation harness and four-arm benchmark before
discovering any of these would mean discovering them at the point where they are most
expensive to act on. The dependency runs the other way: the spike constrains the design, so
it runs first.

## Protocol

Eight steps. Each writes to the spike record; none is skipped because it "obviously" passes.

1. **Select one model.** Record its identity, revision, checksum, and the licence stated on
   its model card. One model only — model selection is a separate question (PRD NG5).
2. **Export or acquire the fp32 ONNX graph** via a fully scripted, re-runnable command.
   Record the command and the tool versions that produced the artefact.
3. **Produce the int8 graph** with `quantize_dynamic`, again fully scripted. Record the
   exact call and its parameters.
4. **Load both graphs on the target machine.** Confirm they expose equivalent input and
   output signatures. A difference here is a finding, not a detail to work around.
5. **Inspect which operators were actually quantized.** Enumerate the operator types in
   both graphs and diff them. If the matmuls are untouched, any observed timing difference
   has some other cause and the "quantization" arm is mislabelled.
6. **Run a fixed smoke set** of name pairs through both graphs. Record cosine similarity
   between fp32 and int8 embeddings per pair, and the maximum divergence.
7. **Measure**, in a fresh child process per arm:
   - cold start — new process, model load through first inference
   - warm single-pair scoring
   - warm batched preparation at documented batch sizes
   - peak resident set size, with the measurement method recorded
   - artefact bytes on disk
8. **Write the raw record** to `benchmarks/` in the standard schema, including full
   environment capture: hardware, OS, Python, ONNX Runtime version, thread count, power
   mode.

## Abort condition

The spike **fails** if any of the following holds:

- either graph does not load or run reliably on the target;
- no operator of consequence was quantized;
- no measured resource — cold start, warm latency, peak RSS, artefact size — changes
  materially in either direction;
- fp32 and int8 embeddings diverge so far that the two arms are not comparable
  implementations of the same model.

On failure, **do not** proceed to wire quantization through the package. The recorded
outcome becomes the reported result for that arm — *the standard first-choice optimization
does not pay for a small encoder on this hardware, and here are the operators that were and
were not converted* — and ADR-0009's primary optimization carries the work. A negative
result obtained under a pre-registered protocol is evidence; the same result discovered
late and quietly dropped is not.

A slower-but-smaller int8 graph is **not** a failure. It is a genuine trade-off and one of
the more useful outcomes available, since artefact size and memory are first-class
constraints on a client device.

## Outputs

- one spike record in `benchmarks/`
- the two model artefacts, fetched not committed (ADR-0005)
- the scripted export and quantization commands, committed
- an operator-level diff between the two graphs
- a go / no-go entry against the abort condition above

## Open questions

1. Which operator types must be quantized for the arm to be honestly labelled "int8"?
   Proposal: the matmuls dominating encoder cost; anything less is reported as partial.
2. Does ONNX Runtime's default `quantize_dynamic` operator selection differ between
   platforms? If so the operator diff must be recorded per platform, not once.
3. Should the smoke set be drawn from the development split, or fixed independently?
   Development split, to avoid touching calibration or sealed data (ADR-0011).
