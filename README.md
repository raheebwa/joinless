# joinless

[![CI](https://github.com/raheebwa/joinless/actions/workflows/ci.yml/badge.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

The coverage badge states an **enforced floor**, not a measured number reported by a
service. `pytest` is configured with `fail_under = 100` over line *and* branch coverage, so
the figure is either 100% or the build above it is red. Nothing is uploaded anywhere to
produce it — see
[ADR-0016](docs/adrs/0016-tests-assert-behaviour-and-cover-every-path.md), which also
explains why a coverage floor is a completeness check rather than a quality one.

**Record linkage when there is no join key — and a measured answer to whether a neural matcher is worth it on-device.**

Two datasets describe the same real-world entities. They share no identifier. One is
coordinate-located but fact-poor; the other is fact-rich but often has no coordinates at
all. You need one merged view, without an O(n²) cross product, and without silently
dropping the rows that carry the facts you added the second source for.

The classical answer is cheap string and geometry work: bucket by coordinate grid,
compare names with a token-overlap coefficient, break ties by distance. It runs anywhere,
needs no model, and fits in a few hundred lines.

The modern answer is to embed the names and compare vectors. It handles abbreviations,
transliterations and word-order noise that token overlap cannot.

**That embeddings win on accuracy is settled, and not by us.**
[LinkTransformer](https://arxiv.org/abs/2309.00789) (Arora & Dell) evaluates four linkage
tasks — company aliases among them — and reports off-the-shelf sentence-transformer models
outperforming Levenshtein edit-distance matching "typically by a wide margin", with
custom-trained models ahead of both. Take that as the starting point, not the finding.

**What nobody publishes is what it costs you.** That work's scaling story is a FAISS
backend extensible to GPUs, and its deployment advice to users new to LLMs is a cloud
service optimized for deep learning, so that dependencies need not be resolved locally.
Accuracy is measured. Cold start, warm latency, resident memory and artifact size on the
machine in your hands are not.

## The question

> Given that the model is more accurate, what does it cost on an Arm64 client — and is
> there a regime where the cheap matcher is still the right call?

Four interchangeable matchers behind one interface, measured on identical data:

| Arm | Matcher | Dependencies |
|---|---|---|
| `overlap` | Token-overlap coefficient over normalised name tokens | standard library |
| `fuzzy` | Character-aware similarity — Jaro-Winkler / token-set ratio | `rapidfuzz` |
| `embed-fp32` | Stock sentence-embedding similarity | ONNX Runtime |
| `embed-int8` | Same model, dynamically quantized | ONNX Runtime |

Two classical arms, not one, and deliberately so. Token overlap is character-blind —
`BRIGHTWATR` and `BRIGHTWATER` share no token — so measuring a transformer against it alone
would be measuring against a matcher nobody defends for name work. `fuzzy` is the arm that
makes "classical" mean something.

## The optimization

Under grid blocking, a single name participates in many comparisons — its own cell plus
eight neighbours. The obvious implementation embeds inside the comparison loop and
recomputes the same vector over and over. **Hoisting preparation out of the comparison
loop, with batching, is the primary optimization measured here**, and the naive
per-comparison path is implemented alongside it as the control — an optimization reported
without its unoptimized baseline is an assertion rather than a measurement.

Quantization is the second, subordinate optimization. It is the one everyone reaches for
first, and whether it actually pays for a small encoder on this hardware is an open
question this benchmark answers rather than assumes.

Notably the classical arms barely benefit from the hoist — token and character comparison
are cheap enough to recompute. That asymmetry is part of the finding: the neural arms need
the optimization to be viable at all.

Measured on: pair precision / recall / F1, cold start, warm scoring, batched preparation,
peak resident memory, and artifact size on disk — reported per perturbation family, not as
a single aggregate.

Results are scoped to a disclosed synthetic benchmark on named hardware. They characterise
these matchers on a controlled distribution; they are not a universal ranking.

## Related work

| | |
|---|---|
| [LinkTransformer](https://arxiv.org/abs/2309.00789) | Record linkage as text retrieval with transformer LMs; off-the-shelf HF models reported as outperforming edit-distance matching, typically by a wide margin, with custom-trained models ahead of both. Establishes the accuracy result this project takes as given. Does not report client-side deployment cost. |
| [Splink](https://moj-analytical-services.github.io/splink/) · [dedupe](https://github.com/dedupeio/dedupe) · [Zingg](https://github.com/zinggAI/zingg) | Full-system probabilistic and learned linkage at scale. `joinless` does not replace them — use them for blocking, multi-field linkage, clustering and review. |
| [Ditto](https://arxiv.org/abs/2004.00584) · [WDC Products](https://arxiv.org/html/2301.09521) | Fine-tuned neural entity matching and its benchmarks. `joinless` deliberately does no fine-tuning; the question is what a *stock* model costs. |

`joinless` sits after any candidate generator and answers one operator decision: should the
pairwise name scorer carry an on-device embedding model?

## Why on-device

Record linkage means holding records about real people and real businesses. Sending them
to a hosted matching API is often the single least acceptable step in the pipeline —
legally, contractually, or ethically. Everything in `joinless` runs locally: no network
calls at match time, no telemetry, no data leaving the machine.

That constraint is also what makes the measurement interesting. On a server you can
afford the model and never think about it. On a client device the trade is real.

## Status

All four arms — `overlap`, `fuzzy`, `embed-fp32`, `embed-int8` — run end to end behind
the `resolve`, `compare`, `doctor` and `benchmark` commands, at an enforced 100% line and
branch coverage floor. `benchmark` runs RFC-0002's protocol over the built-in synthetic
corpus and writes one record per run to [`benchmarks/`](benchmarks/): per-family
precision, recall and F1, the four resource measurements, and the int8 arm's per-family
accuracy divergence from fp32, with the exact hardware and runtime versions recorded
alongside every run.

<!-- BEGIN GENERATED RESULTS: do not edit by hand — regenerate with `uv run python scripts/render_readme_results.py <record> README.md` -->

## Results

Generated from [`benchmarks/20260813T130904Z-benchmark.json`](benchmarks/20260813T130904Z-benchmark.json) by `uv run python scripts/render_readme_results.py benchmarks/20260813T130904Z-benchmark.json README.md` — every figure below traces to that one run record.

**Reference machine:** Darwin 25.5.0 (arm64), 12 cores, 32.0 GiB RAM, Python 3.14.5

**Corpus:** the built-in synthetic corpus (`joinless.corpus`), pooled across seeds 1, 2, 3, 4, 5; families: abbreviation, character noise, exact, formatting, near-miss negative, semantic alias, transliteration, word order

Results are scoped to this disclosed synthetic benchmark on the reference machine named above. They describe how these four matchers compare on this corpus, what the embedding arms cost on this hardware, and whether the preparation hoist and quantization change those costs — nothing wider (ADR-0010). They are not a universal ranking. In particular: this is not a location of "the classical/neural crossover" — two classical matchers and one embedding model do not locate a frontier across either family of technique; it is not a claim that embeddings beat string matching in general, which is already established prior art (see LinkTransformer, cited below), not re-asserted here; and it is not a claim that these results transfer to real corpora unchanged, since the corpus is synthetic by construction.

### Aggregate

| arm | aggregate F1 | warm p50 | peak RSS | artifact |
|---|---|---|---|---|
| `embed-fp32` | 0.871 | 25.00µs | 279.1 MB | 91.1 MB |
| `embed-int8` | 0.866 | 25.25µs | 208.9 MB | 59.3 MB |
| `fuzzy` | 0.886 | 0.96µs | 24.2 MB | — (classical arms carry no model artifact) |
| `overlap` | 0.721 | 0.21µs | 22.7 MB | — (classical arms carry no model artifact) |

### exact

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | 1.000 | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.2 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### formatting

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | 1.000 | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.2 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### word order

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | 1.000 | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.2 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### abbreviation

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 0.769 | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | 0.750 | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | 0.974 | 0 | 0.96µs | 24.2 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (f1=0.974, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (f1=0.974, false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### character noise

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | null (precision is undefined: no predicted positives) | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | null (precision is undefined: no predicted positives) | 0 | 0.96µs | 24.2 MB |
| `overlap` | null (precision is undefined: no predicted positives) | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=null, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### semantic alias

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | null (precision is undefined: no predicted positives) | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | null (precision is undefined: no predicted positives) | 0 | 0.96µs | 24.2 MB |
| `overlap` | null (precision is undefined: no predicted positives) | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=null, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.2 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### transliteration

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | 1.000 | 0 | 25.25µs | 208.9 MB |
| `fuzzy` | 0.961 | 0 | 0.96µs | 24.2 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)
- `embed-int8`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs)

### near-miss negative

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 25.00µs | 279.1 MB |
| `embed-int8` | null (recall is undefined: no actual positives) | 1 | 25.25µs | 208.9 MB |
| `fuzzy` | null (recall is undefined: no actual positives) | 5 | 0.96µs | 24.2 MB |
| `overlap` | null (recall is undefined: no actual positives) | 115 | 0.21µs | 22.7 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `embed-fp32` (f1=null, false_positives=0), `embed-int8` (f1=null, false_positives=1), `fuzzy` (f1=null, false_positives=5), `overlap` (f1=null, false_positives=115).

<!-- END GENERATED RESULTS -->

## Target platform

Arm64 client devices. The reference platform is Apple Silicon (M2 Max). Every published
number names the machine that produced it, and no platform is described without a run
record in [`benchmarks/`](benchmarks/).

The code is plain Python and ONNX Runtime, so Windows on Arm, Arm64 Linux laptops and
Arm single-board computers with sufficient memory are *expected* to work. That is an
expectation from the dependency set, not a measurement, and it stays labelled as one
until a run record exists for the platform in question.

Note that the mainstream on-device inference tutorials for Arm target Android and LLM
workloads. This project is neither, so it takes the framework guidance (ONNX Runtime as
a client runtime) without a template to follow. Deviations are documented in
[`docs/adrs/`](docs/adrs/).

## Documentation

| Document | Contents |
|---|---|
| [`docs/prd.md`](docs/prd.md) | What this is for, who it serves, what is in and out of scope |
| [`docs/adrs/`](docs/adrs/) | Decisions taken, with the reasoning and the alternatives rejected |
| [`docs/rfcs/`](docs/rfcs/) | Designs proposed before implementation |
| [`benchmarks/`](benchmarks/) | Run records — every published number traces to one |

## Prior art

The classical resolver is a reimplementation of the algorithm described in
[`entity-resolution-no-keys`](https://github.com/raheebwa/entity-resolution-no-keys)
(MIT, same author) — grid bucketing, token-overlap name matching, haversine tie-break,
and a dual hashing scheme that keeps coordinate-less rows addressable. That repository
remains the reference description of the algorithm. This one asks a different question:
what happens when you swap the matcher for a model.

## Licence

MIT — see [LICENSE](LICENSE).
