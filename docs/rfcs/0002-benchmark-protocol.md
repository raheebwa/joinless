# RFC-0002 — Benchmark protocol

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** PRD MR-1 … MR-12 · **Bound by:** ADR-0010, ADR-0011, ADR-0013, ADR-0014

## Summary

Define how the four arms are measured so that the resulting numbers mean something and
can be reproduced by a reader on their own machine.

## Motivation

The project's entire contribution is a comparison. A comparison is worth exactly as much
as its methodology. Loose measurement — one run, no warm-up, mean latency, an evaluation
set of easy cases — would produce numbers that look authoritative and are worthless.

The same failure mode does not stop at measurement. A report that can drift from the
record it claims to summarise, or a single ranked winner that silently weighs accuracy
against memory against latency on the reader's behalf, both produce something that looks
authoritative for reasons that have nothing to do with whether it is true.

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

### Per-seed accuracy reporting

The corpus is generated under several deterministic seeds (ADR-0011 rule 3), and the
sealed test in the table above is scored two ways from that same set of seeds, because
pooling and per-seed reporting answer different questions:

| Figure | Answers |
|---|---|
| Pooled | what an arm scores across every seed's sealed-test pairs, combined into one split — more data, the same "reported result" the splits table above names, and the number a single ranked comparison between arms is drawn from |
| Per-seed, with variation | whether that pooled figure depends on one seed's particular draw, or holds across every seed the corpus was generated under |

**Both are kept, deliberately, rather than replacing one with the other.** A pooled-only
report is exactly the defect this section exists to close: after pooling there is one
number per family, so a reader cannot tell a stable result from an artefact of one draw.
A per-seed-only report drops the pooled figure's larger sample for no reason — pooling
does not corrupt anything ADR-0011 rule 2 requires, since every seed's sealed-test pairs
carry that seed as part of their pair id (`joinless.corpus`), so no pair collides across
seeds and pooling never mixes one seed's row into another's. Reporting only one of the
two would answer only one of the two questions above and leave the other for a reader to
reconstruct by hand from a record not shaped to hold it.

The threshold itself is not re-opened by any of this. One threshold per arm, selected once
from the pooled calibration split by the identical procedure threshold governance below
already requires, is applied to the pooled sealed test and to every seed's own sealed test
alike — ADR-0011 rule 2's "identical procedure" has to hold seed to seed for the same
reason it already has to hold arm to arm: a comparison between seeds scored under
different procedures would attribute variation to the draw that actually came from the
scoring, not the data.

The seed-to-seed spread itself is the sample standard deviation of each family's
precision, recall and F1 across whichever seeds produced a defined value for that metric
— undefined, not zero, with fewer than two defined values to compare (ADR-0013): a single
seed's figure has no spread to report, and reporting `0.0` would claim a stability nothing
measured. Every persisted pooled figure carries this variation alongside it; a run record
cannot report one without the other (`joinless.evaluation.SealedTestAccuracy`).

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

**`report` is a pure function of the run record.** It re-renders whatever a run already
wrote; it never re-measures, and there is no argument, flag, or edit point by which a
metric can be supplied to it by hand. This is what turns `benchmarks/README.md`'s claim
that "every number published traces to a record here" from an aspiration into an
enforceable property: without the rule, a hand-typed figure and a measured one are
indistinguishable once both are on the page, and the entire reason for writing a durable
record is to make that distinction checkable. With the rule, a number in a report that
does not trace to a record is a defect with a one-line description — which record it was
supposed to come from, and why it is not there — rather than something a reader has to
take on faith.

### Decision output

The four arms are measured on several dimensions that do not share a unit — accuracy,
warm latency, resident memory, artifact size. Collapsing those into one ranked winner
means choosing an exchange rate between an F1 point and a megabyte of resident memory,
and any rate this project picked would encode this project's priorities, not the
reader's. A mobile client bound by peak RSS and a batch job bound by artifact size can
rank the same four arms in opposite orders, and both orderings are correct for the
constraint that produced them.

**The decision output is therefore the Pareto frontier under constraints the reader
states** — a memory ceiling, a latency ceiling, an accuracy floor, any combination of
these — never a generic winner row. An arm sits on the frontier if no other arm beats it
on every stated dimension at once; everything not on the frontier is dominated by
something that costs no more on any axis and can be dropped without loss. **"No arm
qualifies" is a valid result** for a constraint set nothing measured here satisfies, and
it is more useful than a winner produced by quietly relaxing the constraint the reader
actually stated — it reports the true state of the evidence instead of a comforting
approximation of it.

This is the same boundary ADR-0010 draws for the project's claims generally: what the
evidence supports is a statement of the form *given the model is more accurate, what does
it cost on the reference machine, and does a regime exist where the cheaper arm is still
the right call* — not a single figure that resolves that trade-off on the reader's
behalf.

**Frontier constraints are four: a memory ceiling on peak RSS, a latency ceiling on
warm p50, a false-positives ceiling, an accuracy floor on F1.** Peak RSS and warm p50
are each measured once per arm, independent of family; F1 and false positives are each
read per family (below). An arm whose accuracy for a family is undefined — `null`, with
a reason, never `0.0` (ADR-0013) — cannot be compared against a *stated* floor one way
or the other, so it is excluded from that family's frontier with the record's own
reason attached, not silently scored as failing the floor.

False positives was added after the first three (issue #106). `semantic alias` and
`near-miss negative` are all-negative by design (below), so precision and recall have
no denominator and F1 is undefined for every arm, always — the frontier had no axis to
place an arm on, and reported "no arm qualifies" on the family the four arms separate
most clearly, indistinguishable from a constraint set nothing satisfies. False positives
— `predicted_positives - true_positives`, a plain count with no empty-denominator case
to be undefined by — is defined on every family, every arm, every run, exactly as peak
RSS and warm p50 are, and is bounded by a ceiling the same way: lower is better, like
cost, not like F1. With no floor stated, an arm no longer needs a defined F1 to reach the
frontier; it competes on false positives and cost instead, and its `f1` is recorded as
undefined rather than omitted or coerced into a number. A *stated* floor still excludes
it — an absent F1 still cannot be compared against one — so "no arm qualifies" keeps
meaning exactly what it always meant: a constraint set nothing satisfies, never a
frontier with no axis to place an arm on.

Domination is decided on whichever of the four axes are actually comparable: F1
participates only when every arm being compared has a defined F1, decided once per
family rather than pair by pair (an undefined value cannot be collapsed into "equal" or
"worse" without breaking domination as an ordering); the two cost axes and false
positives always participate, since they are always defined. On a family where every
arm's F1 is defined, this changes nothing — false positives simply joins the axes
already compared, and is `0` for every arm on an all-positive family, where a predicted
positive is a true positive by definition, so it neither helps nor distorts there.

**The frontier is computed per family, never from the whole-record aggregate.** A
constraint set is a statement about the reader's own data, and the aggregate describes
this benchmark's chosen mixture of families, not the reader's — a frontier computed from
it would be answering a question about the corpus, not about the family the reader's
data actually resembles.

### The global threshold, and per-family operating points alongside the frontier

One arm's threshold is selected once, from the pooled calibration split, by the
identical procedure every arm shares (ADR-0011 rule 2), and applied unchanged to every
family in the sealed test — there is no per-family retuning, because a caller deploying
this arm sets one threshold, and reporting only per-family thresholds would describe a
configuration no deployment has.

That single threshold does not serve every family equally. `character noise` is the
clearest case: across all five corpus seeds, every arm proposes zero matches there at
its global threshold, on a family that carries 40 real positives in the sealed split —
not because the family is unlabelled (it is not: `semantic alias` and `near-miss
negative` are the two families that are all-negative *by design*,
`joinless/corpus.py`'s module docstring), but because none of the four arms' calibrated
thresholds happen to admit anything on this family's particular perturbation shape. The
per-family F1 there is `null`, and it stays `null` — reported as a stated property of
the run: *this arm proposes nothing on this family at its global threshold*, not *this
arm was not measured*. Two alternatives were rejected on their merits: changing the
selection objective to stop easy families dominating would trade one measured weakness
for a hidden one, since this project has no principled weighting to offer between
families any more than it has one between accuracy and cost; and replacing the global
threshold with per-family thresholds would answer a question nobody deploying a single
matcher asked.

**The frontier reports per-family operating points alongside it, resolving the same
tension a different way.** A reader whose data resembles `character noise` is not served
by a threshold selected for a mixture they do not have — but *reporting* that arm's own
figure for that family, at the run's frozen threshold, is a different act from *retuning*
the threshold to that family. Every family block in the frontier's output carries each
arm's own precision, recall and F1 for that family — the same frozen-threshold figure
`accuracy.pooled.per_family` already records, grouped by family instead of pooled — set
beside that arm's per-run cost figures, so a reader meeting a `null` for their family sees
exactly what produced it and can read every other arm's figure for the same family in the
same place. This is not per-arm tuning and does not breach ADR-0011 rule 2: the scoring
procedure and the threshold are identical for every arm and every family, which is what
keeps the comparison about the matchers rather than about how each one was tuned.

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

**Emit a single winner row.** Requires assigning weights across accuracy, latency,
memory and artifact size on the reader's behalf and not disclosing that a choice was
made in doing so — the same failure ADR-0010 names for claims about this project more
generally. A row computed under one weighting is wrong for a reader operating under a
different one, and the record does not say which weighting produced it. The frontier
reports what is actually known — which arms are and are not dominated — and leaves the
weighting to whoever has the constraint.
