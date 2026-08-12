# joinless

[![CI](https://github.com/raheebwa/joinless/actions/workflows/ci.yml/badge.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)

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

Early. The benchmark harness and the classical baseline come first; the embedding arms
follow. Results land in [`benchmarks/`](benchmarks/) as they are produced, with
the exact hardware and runtime versions recorded alongside every run.

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
