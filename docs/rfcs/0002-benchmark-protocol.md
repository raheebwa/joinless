# RFC-0002 — Benchmark protocol

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** PRD MR-1 … MR-12 · **Bound by:** ADR-0011, ADR-0013, ADR-0014

## Summary

Define how the four arms are measured so that the resulting numbers mean something and
can be reproduced by a reader on their own machine.

## Motivation

The project's entire contribution is a comparison. A comparison is worth exactly as much
as its methodology. Loose measurement — one run, no warm-up, mean latency, an evaluation
set of easy cases — would produce numbers that look authoritative and are worthless.

## Design

### Evaluation set

A labelled set of record pairs with known-correct outcomes, constructed to include the
cases that separate the arms rather than the cases both handle:

| Category | Example shape | Expected to separate? |
|---|---|---|
| Exact match | identical names | no — both should pass |
| Suffix noise | `X Ltd` vs `X Limited` | no — classical handles via suffix stripping |
| Word order | `A B Traders` vs `Traders A B` | no — token sets are unordered |
| Abbreviation | `Intl Bus Machines` vs `International Business Machines` | **yes** |
| Transliteration | spelling variants of the same name | **yes** |
| Near-miss negative | two genuinely different entities sharing most tokens | **yes — in the other direction** |
| Coordinate-less | record with no lat/lon | tests pass-through, not matching |

Near-miss negatives matter most. An evaluation set of positives only rewards a matcher
for saying yes, and the neural arm will say yes more often. Precision is where the
interesting failure lives.

Composition is documented and committed alongside the set, because it materially shapes
the result (ADR-0004).

**The expected winner per family is recorded before the run** (ADR-0011 rule 4). Families
where the outcome contradicts the expectation are reported as findings — an expectation
that survives every family is weaker evidence than one that breaks somewhere, because a
benchmark that can only confirm its author has not tested anything.

### Splits

Three disjoint roles, generated deterministically (ADR-0011 rule 1):

| Role | Use | Reported? |
|---|---|---|
| development | designing and debugging fixtures and matchers | no |
| calibration | threshold selection only | no |
| sealed test | the reported result | once |

No model is trained here, so this is a tuning/test split rather than a train/test split.
Tuning is where inflation enters, and it enters per arm at different magnitudes — which is
what makes the *ranking* an artefact rather than only the absolute numbers.

### Threshold governance

Each arm's threshold is selected on calibration data alone, by an identical documented
procedure, and frozen before the sealed run. Procedure and selected value go into the run
record. Scores are matcher-specific and are never transferred between arms (RFC-0001).

### Metrics

| Metric | Definition | Why |
|---|---|---|
| Precision | correct matches / proposed matches | catches over-eager matching |
| Recall | correct matches / true matches | catches missed links |
| F1 | harmonic mean | single comparable number |
| Cold start — interpreter start | process launch to the first line of user code, new process per arm | identical across all four arms; not attributable to the matcher |
| Cold start — import | `import joinless` plus the arm's own imports | attributable — the neural arms pay ONNX Runtime's import cost that the classical arms never incur (ADR-0014) |
| Cold start — session creation | inference session construction from the artefact | attributable; `null` for the classical arms, which construct no session (MR-17) |
| Cold start — tokenizer load | tokenizer construction from the artefact | attributable; `null` for the classical arms, which load no tokenizer (MR-17) |
| Cold start — first inference | first embedding call, uninitialised caches | attributable |
| Cold start (total) | sum of the five phases above | derived, never itself measured — exactly one set of quantities is recorded |
| Warm scoring p50 / p99 | per-comparison, after warm-up | steady-state cost; p99 is what a user feels |
| Batched preparation | per-record embedding at documented batch sizes | the production path under ADR-0009 |
| Naive preparation | per-comparison embedding | the control the hoist is measured against |
| Peak RSS | max resident set, isolated child process per arm | the client-device ceiling |
| Artefact size | actual bytes on disk, per artefact | install and distribution cost |
| Bucket occupancy | distribution of candidates per cell | the hoist's win scales with it (ADR-0009) |

Mean latency is deliberately **not** reported. It hides the tail, and the tail is what
makes an interface feel slow.

Cold start is reported separately rather than amortised into per-comparison latency. They
are different costs with different regimes: a long batch job amortises cold start to
nothing, while a short interactive run is dominated by it. Smearing them together produces
a number that describes neither.

Cold start is itself decomposed into phases rather than reported as one number, because a
single figure bundles costs with different owners. Interpreter start is identical across all
four arms and is not attributable to the matcher — it is paid by the process, not the arm, so
folding it into a per-arm figure credits or blames whichever arm happens to be measured for a
cost none of them controls. Import, session creation, tokenizer load and first inference are
each attributable: import cost differs because the neural arms pull in ONNX Runtime and the
classical arms do not (ADR-0014); session creation, tokenizer load and first inference exist
only for the arms that load a model and are reported as `null` with that reason for the arms
that do not (MR-17). The total is derived by summing the phases and is never measured on its
own — exactly one set of quantities is recorded per run, the phase figures, and the total is
arithmetic on them rather than a second measurement that could disagree with the first.

### Method

1. **Separate `prepare` from `matches` in the accounting.** Report per-record preparation
   cost and per-comparison cost separately. Conflating them is the most common way this
   kind of benchmark misleads (see RFC-0001).
2. **Warm-up before timing.** First-call cost includes runtime initialisation and lazy
   graph loading; that is a real cost but a different one, so it is reported separately
   as start-up cost rather than smeared into per-comparison latency.
3. **Repeat and report the distribution**, never a single run.
4. **Identical inputs across arms.** Same records, same order, same seed.
5. **Record the environment** — hardware, OS, Python version, ONNX Runtime and `rapidfuzz`
   versions, model identity, revision and checksum, quantized operator list, thread count,
   warm-up count, repetition count, power mode — into the run record.
6. **Randomise arm order across repeats.** Sustained runs on a laptop throttle, so a fixed
   order systematically favours whichever arm runs while the machine is coolest.
7. **Isolate every resource metric.** Cold start (and its phases), warm scoring, preparation
   cost and peak RSS are each taken in a fresh child process per arm, not only memory. The
   reason is the same for all of them: shared imports and retained allocations contaminate
   whichever metric is being read in that process, not memory alone — a warm allocator and a
   warm page cache make whichever arm runs second look faster, and a runtime already
   imported by an earlier arm makes the next arm's import cost disappear. Arms measured in
   one process report the high-water mark, or the warmed-up cost, of whichever ran first,
   never their own. ADR-0014's install-profile invariant is what makes the isolation
   enforceable rather than assumed: a fresh child process per arm is the only way "the
   classical arm did not import the runtime" is a fact about that process, not a hope about
   which arm happened to run first.
8. **One command** reproduces everything.

### Output

Each run writes a durable record to `benchmarks/` containing environment, evaluation
set identity, per-arm metrics, and the exact command. The README results table is
generated from those records — never hand-written.

## Open questions

1. How many repeats before the p99 stabilises on this workload?
2. Should energy be measured? It is the most honest client-device metric and the hardest to
   measure portably. Out of scope for v1; named here as a known gap rather than omitted.
3. ~~Should thermal state be controlled for?~~ **Settled: randomise arm order across
   repeats** (method step 6). Full thermal control is not achievable on a laptop;
   randomisation converts a systematic bias into variance that the repeat spread exposes.
4. What is the smallest bucket-occupancy figure at which the hoist's advantage is worth
   reporting? Below roughly one candidate per comparison there is nothing to hoist.

## Alternatives considered

**Report accuracy only.** Would hide the entire cost side of the trade — which is the
question.

**Use an existing entity-resolution benchmark suite.** Better external validity, but
those suites are built on real corpora, which ADR-0004 rules out, and they measure whole
systems rather than the single matcher swap isolated here.
