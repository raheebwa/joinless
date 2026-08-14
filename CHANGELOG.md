# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Semantic Versioning rule 4 permits a `0.y.z` release to break anything without notice.
This project does not take that latitude: before `1.0.0` as after it, a breaking change
bumps the MINOR version and is listed under `Changed` or `Removed`. "0.x, caveat emptor"
warns nobody.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-08-14

First release. Everything below was built in this cycle, so the whole entry is
`Added` apart from the defects this project found in its own measurements and
corrected before publishing them.

### Added

- Package manifest (`pyproject.toml`) with a PEP 639 MIT licence expression.
- `dev` and `neural` optional install profiles.
- `joinless` console entry point.
- The importable `joinless` package.
- `export` optional install profile (torch, transformers, `optimum[onnxruntime]`,
  onnx, huggingface_hub) for the scripted model-export tooling under `spikes/`.
- `rapidfuzz` as a runtime dependency, carrying the character-aware classical arm.
- Full line and branch coverage as an enforced floor rather than a reported number.
- `joinless.records` — `Record`, and a dual hashing scheme giving every row a stable
  identifier that does not collapse same-named rows carrying no coordinates.
- `joinless.scoring` — the `Scorer` protocol, the `ThresholdMatcher` adapter, and the
  `overlap` and `fuzzy` classical arms, selectable by name.
- `joinless.corpus` — a deterministic synthetic corpus over the eight perturbation
  families, split into three disjoint roles.
- `joinless.resolver` — `resolve`, merging two record sets that share no key.
- `joinless resolve`, `joinless compare` and `joinless doctor` on the command line.
  `doctor` reports architecture, execution provider, installed profile and offline status,
  so the Arm64, CPU-only and no-network properties are checkable rather than asserted.
- `joinless.evaluation` — per-family precision, recall and F1 reported as defined values
  or as null with a stated reason, never as a zero standing in for an undefined figure;
  threshold selection on the calibration role; and pre-registered expectations, whose
  failures are returned as first-class contradictions rather than folded into a log line.
- `joinless.measurement` — warm latency, peak resident memory and a five-phase cold-start
  decomposition, each measured in a fresh child interpreter so one arm's cost cannot leak
  into another's. An arm whose artifact or runtime is absent is reported unavailable with
  the reason, and keeps its row.
- `joinless.runrecord` — the run record: hardware, OS, interpreter and runtime versions,
  model identity and checksum, evaluation-set identity, and every arm's result, written
  to a new file that is never overwritten.
- `joinless benchmark` on the command line, writing one such record per run.
- `joinless.embedding` — the `embed-fp32` arm: a MiniLM sentence encoder behind the same
  `Scorer` protocol as the classical arms, with mean pooling and cosine similarity, and
  batched preparation as part of the protocol so a per-pair caller cannot silently opt
  out of the batching the measurement assumes.
- `tokenizers` in the `neural` profile, loaded from a local `tokenizer.json`, so an
  inference-only install carries no model-hub tooling.
- Artifact size on disk recorded per arm, making the storage cost of an arm part of the
  record rather than a figure a reader has to go and measure.
- The `embed-int8` arm, on RFC-0004's spike record recording a "go": the same
  `EmbeddingScorer` class the `embed-fp32` arm uses, over the dynamically-quantized
  graph, sharing the fp32 arm's tokenizer file. A run can now load both arms' model
  checksums at once, each recorded under its own name rather than one overwriting the
  other. The int8 arm's per-family F1 divergence from the fp32 arm, computed from both
  arms' own accuracy reports in the same run, is a new field on every run record.
- The int8 arm's matmul-conversion census — how many of each candidate operator type
  converted and how many remain fp32, not just which replacement operator types are
  present — read from its graph fresh on every run rather than carried over from the
  spike record, and checked against the exact census that graph is recorded to have: a
  run whose graph does not match it, on operator types or on counts, writes no record
  at all.
- Per-family accuracy is now reported both pooled across every seed and per seed, with
  the seed-to-seed variation (sample standard deviation of precision, recall and F1)
  stated as a figure rather than left for a reader to compute — a pooled figure and its
  variation are structurally inseparable in the record: the type they are recorded on
  cannot be constructed with one and not the other.
- `joinless.report` and `joinless report` — the per-family results table rendered from a
  run record and nothing else. The command re-renders what a run already wrote: it
  measures nothing, constructs no scorer, and has no argument through which a figure
  could arrive instead of being read.
- A false positive count on every family for every arm. It is defined wherever precision
  and recall are not, which is what lets the two all-negative families report a result at
  all: on the lexical trap family the arms separate 115 / 5 / 1 / 0, where every one of
  them previously reported `null`.
- `joinless.frontier` — the Pareto frontier per family under constraints a caller states,
  over accuracy, false positives, resident memory and warm latency. "No arm qualifies" is
  a first-class result naming why each arm was excluded, and no generic winner row is
  emitted anywhere.
- A generated results section in the README, produced from one named record by a committed
  command, carrying the reference machine, the corpus, the seeds and the scope the
  evidence supports.

### Fixed

- The preparation-cost worker now times `joinless.resolver`'s own `prepare_hoisted` and
  `prepare_naive` functions — the same two `score_candidates` calls to score every
  candidate pair — instead of a second, hand-written copy of both loops. The reported
  hoist speed-ups now describe the call pattern the library actually ships.
- Preparation cost is now measured over repeated, warmed-up samples of each path and
  reported at the median and the 99th percentile, the same shape `measure_warm_latency`
  already established for warm scoring, instead of a single untimed draw. A single draw
  of this figure was noise large enough to flip which arm looked faster; the median is
  stable across independent runs where the single draw was not, and `warmup_count`/
  `repetition_count` now travel on the figure itself.

[Unreleased]: https://github.com/raheebwa/joinless/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/raheebwa/joinless/releases/tag/v0.1.0
