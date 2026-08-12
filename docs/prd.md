# joinless — Product Requirements

**Status:** draft · **Owner:** Aheebwa Ramadhan · **Last updated:** 2026-08-12

---

## 1. Problem

Merging two datasets that describe the same real-world entities is trivial when they
share an identifier and intractable when they do not. The no-shared-key case is the
common one in public-record, civic and open-data work: a geographic source and a
descriptive source overlap in coverage, disagree in structure, and have no key between
them.

Three failure modes sink naive implementations:

1. **Quadratic blow-up.** Comparing every row in A against every row in B is O(n²). At
   ten thousand rows a side that is 10⁸ comparisons for a merge that should take seconds.
2. **Silent data loss.** Restricting the match to rows that both carry coordinates
   discards exactly the coordinate-less rows that hold the facts the geographic source
   lacks — deleting the reason the second source was added.
3. **Identifier collisions.** Deriving a row id from `name + coordinates` collapses two
   same-named coordinate-less rows into one, losing data a second way.

A classical resolver solves all three in a few hundred lines: a coordinate grid for
near-linear candidate generation under sparse buckets, a token-overlap coefficient for
dirty names, a distance tie-break, and a dual hashing scheme so coordinate-less rows stay
addressable.

Where it struggles is names. Token overlap cannot see that `INTL BUSINESS MACHINES` and
`International Business Machines Corp` are the same entity, or that a transliterated name
matches its source-language form. Embedding the names addresses that class of miss — at
the cost of a model, a runtime, memory and latency.

That embeddings are more accurate than string matching on record linkage is established
prior art and is taken here as a starting point, not produced here as a finding.
[LinkTransformer](https://arxiv.org/abs/2309.00789) (Arora & Dell) evaluates four
linkage tasks — company aliases among them — and reports off-the-shelf
sentence-transformer models outperforming Levenshtein edit-distance matching "typically
by a wide margin".

What that work does not report is the cost side on a client machine. Its scaling story is a
FAISS backend extensible to GPUs, and its deployment advice to users new to LLMs is a cloud
service optimized for deep learning, so that dependencies need not be resolved locally. It
reports no cold start, no warm latency, no resident memory and no artifact size for the
machine in the operator's hands. That is the gap this project addresses:

> Given the model is more accurate, what does it cost on an Arm64 client — and is there a
> regime where the cheap matcher is still the right call?

What may and may not be claimed from the resulting evidence is fixed by ADR-0010, and
binds this document as much as any other.

## 2. Who this is for

| User | Need |
|---|---|
| **Data engineer merging public-record sources** | A correct, fast resolver that does not lose coordinate-less rows, and a defensible reason to pick a matcher |
| **Developer with privacy or contractual constraints** | Linkage that provably never sends records off the machine |
| **Practitioner evaluating on-device inference** | Honest numbers on what a small embedding model costs on a client device, against a real non-trivial baseline |

## 3. Goals

- **G1** Ship a correct keyless entity resolver: candidate generation that is linear in
  the number of records under bounded bucket occupancy, retention of coordinate-less rows,
  and stable namespaced identifiers with defined behaviour for exact duplicates.
- **G2** Make the name matcher a swappable component, so matching strategy is a
  configuration decision rather than a rewrite.
- **G3** Ship four matcher implementations — token-overlap, character-aware classical,
  embedding fp32, embedding int8.
- **G3a** Measure the preparation hoist against a naive per-comparison control, as the
  primary optimization.
- **G4** Publish a reproducible benchmark measuring all four arms on identical data across
  accuracy, latency, memory and model size.
- **G5** Run entirely locally on Arm64 client hardware. No network calls at match time.
- **G6** Report what each arm costs on the named reference machine, per perturbation
  family, and the conditions under which each arm is the reasonable choice **for this
  benchmark** — including the families where a classical arm wins.
- **G7** Make the procedure runnable on someone else's labelled pairs, so the transferable
  artefact is the method rather than the number.

## 4. Non-goals

- **NG1** Not a distributed or clustered system. Single process, single machine.
- **NG2** Not a general-purpose ML framework. One task: deciding whether two names refer
  to the same entity.
- **NG3** Not shipping a fine-tuned model. Stock pretrained embeddings only; fine-tuning
  is a separate question and would confound the measurement.
- **NG4** No real-world corpus in the repository. Fixtures are synthetic (see ADR-0004).
- **NG5** Not chasing state-of-the-art accuracy. The contribution is the measured
  trade-off, not the top of a leaderboard.
- **NG6** Not a full-system linkage tool. Multi-field probabilistic matching, clustering,
  active learning, review workflows and connectors are owned by mature projects; adding
  any of them enlarges the system and leaves the measurement where it was (ADR-0012).

## 5. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Resolve two record sets with no shared key into one merged set | P0 |
| FR-2 | Candidate generation via coordinate grid bucketing. Expected linear in record count **under bounded bucket occupancy**; degrades toward quadratic as occupancy grows, since a listing is compared against every candidate in its cell and the eight neighbours. Occupancy distribution is measured and reported, not assumed | P0 |
| FR-3 | Coordinate-less records are **retained as unmatched source records**, never dropped. They are not linked: a record that never enters a candidate set is unreachable by any matcher, so no matcher — classical or neural — changes this. Linking them would require a separate name-only blocking strategy, which is out of scope (§10) | P0 |
| FR-4 | Dual hashing so coordinate-less same-named rows get distinct stable ids. Identity is defined by a canonical serialisation over a named source namespace; **a hash is collision-resistant, not collision-free**, and two byte-identical rows from one source cannot be separated by content alone — behaviour for exact duplicates is defined explicitly (source ordinal participates) rather than left to the hash | P0 |
| FR-5 | Merge policy: more-populated record wins; geographic coordinates always win; provenance fields unioned | P0 |
| FR-6 | Two-part matching protocol: a scorer that returns a similarity value for a name pair, and a single thresholding adapter that turns a score into a decision. Thresholding lives in exactly one place, so substituting a scorer cannot change blocking, identity, distance or merge policy | P0 |
| FR-7 | Token-overlap scorer — token-overlap coefficient, standard library only | P0 |
| FR-7a | Character-aware scorer — Jaro-Winkler / token-set similarity via `rapidfuzz` (ADR-0008) | P0 |
| FR-8 | Embedding scorer — ONNX Runtime, configurable model | P0 |
| FR-8a | Batched preparation as part of the scorer contract (ADR-0009) | P0 |
| FR-8b | Naive per-comparison call path retained as the optimization control, and run for **all four arms**, not only the neural ones. The reported effect of the hoist is the difference between arms (ADR-0009); measuring one side of that difference leaves nothing to compare it against | P0 |
| FR-9 | int8 dynamic quantization path for the embedding scorer | P0 |
| FR-10 | Scorer selection by configuration, no code change | P1 |
| FR-11 | Command-line interface as the reproduction surface — `compare`, `benchmark`, `report`, `resolve`, `doctor` — running with no network access. The demonstrated action is the same documented command another developer runs, and `doctor` reports architecture, execution provider, installed profile and offline status so the Arm64/CPU-only and no-network properties are checkable rather than asserted | P0 |
| FR-11a | Optional read-only viewer over recorded runs, bounded by a must-not list: no measurement performed inside a web request, no editing of recorded results or thresholds, no implicit model download, no transmission of names, and never required to reproduce a result. A live threshold control exists in `compare` only — benchmark thresholds are fixed by the calibration policy (ADR-0011) | P2 |
| FR-12 | Installable package with declared dependencies and two install profiles: a base install that carries no neural dependencies, and an opt-in extra that carries the neural runtime (ADR-0014) | P1 |
| FR-12a | Classical-only execution never initialises ONNX Runtime, as an invariant with a test rather than a convention. Without the boundary, a classical arm's measured cost inherits the neural arm's import cost and "the classical arm is cheap" is unmeasurable (ADR-0014) | P0 |
| FR-13 | Fail closed, never silently degrade (ADR-0013): an arm that cannot initialise is recorded as unavailable with a reason rather than omitted from the table, and a missing or checksum-mismatched model artefact refuses to run rather than fetching one | P0 |
| FR-14 | Labelled-pairs input path — CSV or JSONL carrying `left_name`, `right_name` and `label`, with optional `category` and `pair_id` (ADR-0012, RFC-0005) | P1 |
| FR-15 | Schema validation whose failure messages name the offending row and column, and a load-time disjointness check over whichever split assignment is in force, so calibration and sealed test can never overlap — including where one pair is assigned to two splits. Whether a user-supplied split is honoured at all, or splits are always derived deterministically from the file, is settled in RFC-0005 | P1 |
| FR-16 | One code path: the built-in synthetic corpus is a producer of the labelled-pairs schema, not a parallel implementation. Generated and user-supplied pairs share validation, splitting, threshold governance, scoring and record writing; a divergence between the two is a defect (ADR-0012) | P0 |

## 6. Measurement requirements

| ID | Requirement | Priority |
|---|---|---|
| MR-1 | Labelled evaluation set with known-correct pairings, covering the perturbation families named in ADR-0011 — exact, formatting, word order, abbreviation, character noise, semantic alias, transliteration, near-miss negative. That list is canonical and this requirement does not restate a shorter one. Character noise and semantic alias are load-bearing: the first is invisible to token overlap, and the second is where an embedding model is expected to overmatch | P0 |
| MR-2 | Report precision, recall and F1 per arm **per family**. The aggregate is a derived summary of the per-family table, never the reported form — fixture composition determines an aggregate ranking and an aggregate conceals that | P0 |
| MR-3 | Report p50 and p99 **per-comparison** latency. Any per-1 000-comparison figure is derived from it, so exactly one quantity is measured | P0 |
| MR-4 | Report peak resident memory per arm | P0 |
| MR-5 | Report model size on disk per arm | P0 |
| MR-6 | Record hardware, OS, Python and runtime versions with every run | P0 |
| MR-7 | Benchmark reproducible from a single documented command (FR-11) | P0 |
| MR-8 | Runs archived as durable records, not just printed. The report is a pure function of the record: it re-renders, never re-measures, and there is no manual metric override path. A figure in a report that is not in a record is a defect | P1 |
| MR-9 | Report the accuracy/cost Pareto frontier under stated constraints, naming the conditions under which each arm is the reasonable choice for this benchmark. "No arm qualifies" is a valid result, and there is no generic winner row | P1 |
| MR-10 | Report hoisted vs naive preparation cost per arm, with candidate-bucket occupancy distribution | P0 |
| MR-11 | Report cold start separately from warm scoring, decomposed into phases — interpreter start, import, session creation, tokenizer load, first inference — so cost that is not attributable to the arm is visible as such | P0 |
| MR-12 | Record which operators were actually quantized in the int8 graph | P0 |
| MR-13 | Corpus generated into three disjoint roles — development, calibration, sealed test — with disjointness enforced rather than assumed (ADR-0011) | P0 |
| MR-14 | Each arm's threshold selected on calibration data alone, by an identical documented procedure, and frozen before the sealed test runs. The procedure and the selected value are recorded in the run record (ADR-0011) | P0 |
| MR-15 | Corpus generated under several deterministic seeds, with per-family results reported across seeds and the seed-to-seed variation stated, so a result that depends on one draw is visible as one (ADR-0011) | P0 |
| MR-16 | Expected winner recorded per family **before** the run; families whose outcome contradicts the expectation are reported as findings rather than smoothed away (ADR-0011) | P0 |
| MR-17 | An undefined metric is reported as `null` with a stated reason, never as `0` — a zero and an undefined are different facts. A run whose threshold was selected using sealed-test data is marked invalid, not merely warned (ADR-0013) | P0 |
| MR-18 | User-supplied names are excluded from durable run records by default: records carry `pair_id` or an index plus aggregate counts, and raw names appear only under an explicit local error-analysis flag. Run records are meant to be shared, and user pairs may be personal data (ADR-0012) | P0 |

## 7. Success criteria

1. A reader can run one command on an Arm64 machine and reproduce the published table.
2. The four arms are measured on identical inputs with identical scoring.
3. What each arm costs on the named reference machine is stated with numbers behind it,
   per perturbation family, together with the conditions under which each arm is the
   reasonable choice for this benchmark.
4. Cases where a classical arm beats the model are reported as prominently as the
   cases where it loses.
5. Nothing in the repository requires network access at match time.
6. A reader can run the same protocol over their own labelled pairs and obtain a record in
   the same shape as the published one.

## 8. Milestones

| # | Milestone | Contents |
|---|---|---|
| M1 | Baseline | Resolver + token-overlap and character-aware scorers + synthetic fixtures + tests, all green, reachable through `compare`, `resolve` and `doctor` |
| M2 | Evaluation | Labelled eval set (MR-1) under the split and threshold governance of MR-13 – MR-16, scored by `benchmark` and rendered by `report` |
| M3 | Neural arm | Embedding scorer (fp32) behind the same protocol, batched, installed through the opt-in extra (FR-12) |
| M4 | Optimization | Preparation hoist measured against the naive control across all four arms (primary); int8 quantization measured (secondary) |
| M5 | Result | Pareto analysis, published per-family table, README results section |
| M6 | Viewer | Optional read-only viewer over recorded runs (FR-11a) |
| M7 | Bring-your-own pairs | Labelled-pairs input path (FR-14 – FR-16) through the same code path as the generated corpus |

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Fixture composition decides the ranking | The reported order is a property of the corpus, not the matchers | Compose the corpus under MR-1 and MR-16: families where cheap comparison is expected to win, expected winner pre-registered per family, results reported per family across seeds |
| Threshold tuning inflates one arm more than another | The comparison measures tuning effort | Calibration-only selection by an identical procedure, frozen before the sealed test (MR-14) |
| Stock sentence embeddings are weak on short business names | Neural arm underperforms | That is a publishable finding, not a failure — report it |
| Quantization accuracy loss is larger than the latency gain | The secondary optimization does not pay | Report it; the preparation hoist is the primary optimization (ADR-0009), and a negative int8 result with numbers is still a result |
| An arm fails to initialise, or a metric denominator is undefined | A broken run reads as a bad arm | Fail closed (FR-13, MR-17): unavailable arms and undefined metrics are recorded as such, never as a zero |
| Runtime version differences change the numbers | Non-reproducible | Pin and record every version with each run (MR-6) |
| Scope creep into fine-tuning or a larger model zoo | Nothing finishes | NG3, NG5 and NG6 are load-bearing; hold them |

## 10. Out of scope for v1

Multi-language name matching beyond what stock embeddings provide · blocking strategies
other than coordinate grid · streaming or incremental resolution · a persistence layer ·
GPU or NPU execution providers · fine-tuning · distributed execution · schema mapping ·
clustering · active learning · review queues · source connectors · serving infrastructure.

The final six — schema mapping onward — are the scope ADR-0012 rejects by name. If the
labelled-pairs path starts to require any of them, that scope is re-entering through it.

## 11. Open questions

1. Which embedding model gives the best accuracy-per-megabyte on short entity names?
2. Is cosine similarity over mean-pooled embeddings the right comparison for short entity
   names, or does a different pooling or similarity function do better? (The
   character-aware half of this question is now an arm of the benchmark, FR-7a, rather
   than an open question.)
3. Replacing candidate generation with a vector index is out of scope for v1 (§10) and
   would be a different system; it stays a question only in the sense that it would need
   its own measurement before anyone should prefer it here.
4. Does int8 quantization hurt short-string similarity more than it hurts general
   sentence similarity?
