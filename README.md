# joinless

[![CI](https://github.com/raheebwa/joinless/actions/workflows/ci.yml/badge.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/raheebwa/joinless/actions/workflows/ci.yml)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

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

## Install

Two install profiles, both `pip install`:

| Profile | Command | Adds | Unlocks |
|---|---|---|---|
| Base | `pip install .` (from a clone) | `rapidfuzz` — a base dependency, not optional | `overlap`, `fuzzy`; `resolve`, `compare`, `doctor`, `benchmark`, `report` |
| Neural | `pip install ".[neural]"` | `onnxruntime`, `onnx`, `tokenizers` | the *ability* to construct `embed-fp32`/`embed-int8` — see below |

Installing `.[neural]` is not, by itself, enough to run the two embedding arms: they also
need a fetched, checksummed model artefact on disk (see "Reproducing the neural arms").
Without one, they report themselves `Unavailable` with a reason (ADR-0013) — a clear
message naming the missing piece — rather than crashing, and every other arm and command
is unaffected.

**`doctor`, `resolve` (against `overlap`/`fuzzy`) and `compare` all work on the base
profile, with no model artefact present.** So does `benchmark`: it attempts all four arms
every run and records the two neural ones as unavailable rather than omitting them, so a
base-profile run still writes a complete record.

For development — adds `pytest`, `ruff`, `mypy`, `hypothesis`:

```sh
pip install -e ".[dev]"
```

## Quickstart

Every command below was run from a clone with `pip install -e ".[dev]"` (base profile —
see "Install"). Output is copied verbatim from a real run; only the machine-specific
lines (paths, timings, the interpreter's own version) are exactly what that machine
printed and will differ on yours.

### `doctor` — what's installed

```sh
$ joinless doctor
architecture: arm64
operating system: Darwin 25.5.0
python version: 3.14.5
joinless version: 0.1.0
execution provider: cpu (ADR-0006: no GPU or NPU provider is ever configured)
installed profile: base
offline status: no command in this package opens a network connection
benchmark run record: none (doctor reports the environment, not a run; see benchmarks/ for run records)
```

`installed profile` reads `neural` instead of `base` the moment `onnxruntime` is
importable — `doctor` checks that it's on `sys.path` without importing it (ADR-0014), so
running this command never pays, or risks, the import cost it is reporting on.

### `compare` — score one name pair

```sh
$ joinless compare "Northbridge Hardware" "Northbridge Hardware Ltd"
scorer: overlap
threshold: 0.8
score: 1.0000
decision: match
elapsed: 0.0085 ms (illustrative timing for this single comparison only, never benchmark evidence — see `joinless benchmark`)
```

`--scorer` selects an arm not yet available on your profile and reports why, rather than
crashing:

```sh
$ joinless compare "Northbridge Hardware" "Northbridge Hardware Ltd" --scorer embed-fp32
Scorer 'embed-fp32' is unavailable: the 'onnxruntime' package is not installed (No module named 'onnxruntime'); install with `pip install 'joinless[neural]'`
```

### `resolve` — merge two record sets

**Input schema.** Each side is a JSON Lines file — one JSON object per line — read by
`joinless.cli._read_records`. A row's `name` is the only required field; `latitude`,
`longitude` and `fields` (a string-to-string map) are all optional and default to absent
or empty. `source` and `ordinal` are never supplied in the file: `source` is the input
file's own stem, and `ordinal` is a row's position among that file's non-blank lines.
Blank lines are skipped, not counted, so a trailing newline never shifts a later row's
ordinal.

**Output schema.** One JSON object per line: a matched pair is `{"status": "matched",
"name", "latitude", "longitude", "fields", "sources"}` — the merge of both sides under
the FR-5 policy (coordinates come from whichever side has them; the more-populated side's
name and field values win a genuine disagreement; every field key present on only one
side survives regardless). An unmatched record is `{"status": "unmatched", "source",
"ordinal", "name", "latitude", "longitude", "fields", "reason"}` — kept, with the reason
it did not match, rather than dropped.

[`examples/left.jsonl`](examples/left.jsonl) and [`examples/right.jsonl`](examples/right.jsonl)
are one small synthetic record set each — invented names, no real business, person,
address or coordinate. Resolving them:

```sh
$ cat examples/left.jsonl
{"name": "Northbridge Hardware", "latitude": 0.31, "longitude": 32.58, "fields": {"category": "hardware"}}
{"name": "Meridian Foods", "latitude": 0.42, "longitude": 32.61}

$ cat examples/right.jsonl
{"name": "Northbridge Hardware Ltd", "latitude": 0.31, "longitude": 32.58, "fields": {"phone": "+000 000 000 000", "hours": "08:00-18:00"}}
{"name": "Unrelated Traders"}

$ joinless resolve --left examples/left.jsonl --right examples/right.jsonl --output examples/merged.jsonl
resolved 2 left record(s) and 2 right record(s) under 'overlap': 1 matched pair(s), 2 unmatched record(s) written to examples/merged.jsonl

$ cat examples/merged.jsonl
{"fields": {"category": "hardware", "hours": "08:00-18:00", "phone": "+000 000 000 000"}, "latitude": 0.31, "longitude": 32.58, "name": "Northbridge Hardware Ltd", "sources": ["left", "right"], "status": "matched"}
{"fields": {}, "latitude": 0.42, "longitude": 32.61, "name": "Meridian Foods", "ordinal": 1, "reason": "no candidate record shares its nine-cell grid neighbourhood", "source": "left", "status": "unmatched"}
{"fields": {}, "latitude": null, "longitude": null, "name": "Unrelated Traders", "ordinal": 1, "reason": "no coordinates: a record without them can never enter a candidate set (FR-3)", "source": "right", "status": "unmatched"}
```

One line of each kind this schema produces: `Northbridge Hardware` and `Northbridge
Hardware Ltd` share every token but one, so `overlap`'s default threshold (0.8) matches
them, and the merge keeps the more-populated side's name (`Ltd`, three `fields` keys)
while carrying over `category` from the side that would otherwise have lost it.
`Meridian Foods` sits over 0.1° from anything on the other side — outside the nine-cell
grid neighbourhood a candidate must share — and `Unrelated Traders` carries no
coordinates at all, so it can never enter a candidate set (FR-3); each is kept with the
reason that applies to it. A test
(`tests/test_cli.py::test_resolve_reproduces_the_committed_example_output`) reruns this
exact command over the committed pair and asserts the output matches
`examples/merged.jsonl` byte-for-byte, so the committed file cannot drift from what the
command actually produces.

A record missing `name` is reported with the file and the line it came from, rather than
an uncaught traceback:

```sh
$ printf '{"latitude": 0.31, "longitude": 32.58}\n' > /tmp/bad.jsonl
$ joinless resolve --left /tmp/bad.jsonl --right examples/right.jsonl --output /tmp/out.jsonl
/tmp/bad.jsonl:1: record has no 'name' field
```

### `pytest` — the test suite

```sh
$ python -m pytest
[...]
================================ tests coverage ================================
_______________ coverage: platform darwin, python 3.14.5-final-0 _______________

Name    Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------
TOTAL    1901      0    438      0   100%

13 files skipped due to complete coverage.
Required test coverage of 100.0% reached. Total coverage: 100.00%
============================= 639 passed in 20.85s =============================
```

`639 passed` is the base profile's count. Installing `.[dev,neural]` on top adds the tests
that exercise a real, importable `onnxruntime`/`tokenizers` pair — `653 passed` on the
same machine, same suite, still 100% line and branch coverage either way.

### `benchmark` — run the protocol, write a record

```sh
$ joinless benchmark
```

Not run in this walkthrough: a published run record is produced once, on the pinned
interpreter CONTRIBUTING names ("a measurement is only comparable to another measurement
made the same way"), and checked into `benchmarks/`; a run from an arbitrary quickstart
would not become that record, only an uncommitted extra file next to it. Its behaviour is
stated instead from the source that implements it (`joinless/cli.py`'s `_cmd_benchmark`,
and `tests/test_cli.py::test_benchmark_records_an_arm_without_a_configured_cache_dir_as_unavailable`,
which exercises exactly the base-profile case): it attempts all four arms regardless of
which profile is installed, recording an unavailable one with a reason rather than
omitting it (ADR-0013), so a base-profile run still writes a complete record; it prints
`wrote benchmarks/<UTC-timestamp>-benchmark.json`, then a contradictions line
(`contradictions: none — every pre-registered expectation held` when every
pre-registered expectation held) and a preparation-hoist-asymmetry line — both quoted
verbatim from `joinless/cli.py`, not paraphrased; and it writes exactly the record shape
`report` (below) reads.

### `report` — render a run record's table

```sh
$ joinless report benchmarks/20260813T130904Z-benchmark.json
cannot render benchmarks/20260813T130904Z-benchmark.json: record carries schema 'benchmark-v6'; this build renders 'benchmark-v7'. Re-run `joinless benchmark` to produce a record this version can read.
```

That is the real, current output of that exact command against the record actually
checked in below — not a hypothetical. `report` checks a record's `schema` field before
rendering anything (`joinless.report.RENDERABLE_SCHEMA`) and refuses one written under an
earlier schema with a message naming both, rather than rendering a table shape the record
does not match. A record `benchmark` writes on this build carries the schema `report`
renders, so `joinless benchmark` followed by `joinless report` on the path it just printed
always succeeds.

## Reproducing the neural arms

`embed-fp32` and `embed-int8` score names with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0), exported to ONNX and,
for the int8 arm, dynamically quantized. Every value below is a literal pinned in
[`joinless/embedding.py`](joinless/embedding.py) — read from there, not retyped from
memory, so this table cannot drift from what the code actually checks.

**Cache directory.** One environment variable names a writable directory nothing else in
this project assumes exists:

```sh
export JOINLESS_MODEL_CACHE_DIR=/path/to/a/writable/directory
```

Its layout, and the checksum each file is verified against before an arm will construct
(ADR-0013's fourth fail-closed rule — a mismatch is reported, never repaired by fetching a
replacement):

| Arm | File | Path under `$JOINLESS_MODEL_CACHE_DIR` | SHA-256 |
|---|---|---|---|
| `embed-fp32` | model | `fp32/model.onnx` | `e3fe9a9a8c877bd5ca0deebb6303aba138acc6818440211377afaca1ba78b511` |
| `embed-fp32` | tokenizer | `fp32/tokenizer.json` | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` |
| `embed-int8` | model | `int8/model.onnx` | `eebed71d4f7671a4d8093decee1fb23018992e139813f30d502bf16ee408208e` |
| `embed-int8` | tokenizer | `fp32/tokenizer.json` (shared — RFC-0004) | same as `embed-fp32`'s |

**Producing the artefact.** Fetching and exporting the model is separate, one-shot tooling
under [`spikes/quantization/`](spikes/quantization/) (its own
[README](spikes/quantization/README.md) is the authoritative Setup section) — never run
by anything under `joinless/` itself, so an inference-only install never pays its cost:

```sh
pip install -e ".[dev,export]"
python -m spikes.quantization.model          # fetch + record the model's identity
python -m spikes.quantization.export_fp32    # export the fp32 ONNX graph
python -m spikes.quantization.quantize_int8  # int8-quantize it (only if you need embed-int8)
```

Then, to run the arms themselves:

```sh
pip install -e ".[dev,neural]"
```

**`neural` and `export` are deliberately two different extras**, not one
(ADR-0014). `neural` (`onnxruntime`, `onnx`, `tokenizers`) is the inference runtime whose
*cost* this project measures — folding build-time tooling into it would change what that
measurement means. `export` (`torch`, `transformers`, `optimum[onnxruntime]`, `onnx`,
`huggingface_hub`) is that build-time tooling: nothing it installs ever runs at scoring
time, and nothing under `joinless/` imports any of it.

That separation is also why an open, disclosed dependency exposure stays bounded to a
minority of installs.
[Issue #92](https://github.com/raheebwa/joinless/issues/92) records three `transformers`
advisories reachable only through the `export` extra — every fix ships as a `5.x` release,
and `optimum-onnx`'s own `transformers<4.58` cap currently refuses to admit any of them, so
no installable fix exists yet. Nothing under `joinless/` imports `transformers`; it is absent from both
the base profile and `neural`, so installing either to *use* this library never acquires
it. The one consumer is `spikes/quantization/export_fp32.py`, run by hand, against one
model named and checksummed in the run record it produces, which refuses a mismatch
(ADR-0013) rather than loading whatever the exposure's threat model assumes — a known,
checksummed load, not an arbitrary one. That bounds the exposure to a maintainer running
export tooling; it does not eliminate it.

<!-- BEGIN GENERATED RESULTS: do not edit by hand — regenerate with `uv run python scripts/render_readme_results.py <record> README.md` -->

## Results

Generated from [`benchmarks/20260813T151146Z-benchmark.json`](benchmarks/20260813T151146Z-benchmark.json) by `uv run python scripts/render_readme_results.py benchmarks/20260813T151146Z-benchmark.json README.md` — every figure below traces to that one run record.

**Reference machine:** Darwin 25.5.0 (arm64), 12 cores, 32.0 GiB RAM, Python 3.14.5

**Corpus:** the built-in synthetic corpus (`joinless.corpus`), pooled across seeds 1, 2, 3, 4, 5; families: abbreviation, character noise, exact, formatting, near-miss negative, semantic alias, transliteration, word order

Results are scoped to this disclosed synthetic benchmark on the reference machine named above. They describe how these four matchers compare on this corpus, what the embedding arms cost on this hardware, and whether the preparation hoist and quantization change those costs — nothing wider (ADR-0010). They are not a universal ranking. In particular: this is not a location of "the classical/neural crossover" — two classical matchers and one embedding model do not locate a frontier across either family of technique; it is not a claim that embeddings beat string matching in general, which is already established prior art (see LinkTransformer, cited below), not re-asserted here; and it is not a claim that these results transfer to real corpora unchanged, since the corpus is synthetic by construction.

### Aggregate

| arm | aggregate F1 | warm p50 | peak RSS | artifact |
|---|---|---|---|---|
| `embed-fp32` | 0.871 | 24.87µs | 275.3 MB | 91.1 MB |
| `embed-int8` | 0.866 | 24.71µs | 208.9 MB | 59.3 MB |
| `fuzzy` | 0.886 | 0.96µs | 24.0 MB | — (classical arms carry no model artifact) |
| `overlap` | 0.721 | 0.21µs | 22.5 MB | — (classical arms carry no model artifact) |

### exact

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | 1.000 | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.0 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (f1=1.000, false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### formatting

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | 1.000 | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.0 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (f1=1.000, false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### word order

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | 1.000 | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | 1.000 | 0 | 0.96µs | 24.0 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (f1=1.000, false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'fuzzy' (f1=1.000, false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### abbreviation

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 0.769 | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | 0.750 | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | 0.974 | 0 | 0.96µs | 24.0 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'fuzzy' (f1=0.974, false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `embed-int8`: dominated by 'fuzzy' (f1=0.974, false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### character noise

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | null (precision is undefined: no predicted positives) | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | null (precision is undefined: no predicted positives) | 0 | 0.96µs | 24.0 MB |
| `overlap` | null (precision is undefined: no predicted positives) | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=null, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### semantic alias

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | null (precision is undefined: no predicted positives) | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | null (precision is undefined: no predicted positives) | 0 | 0.96µs | 24.0 MB |
| `overlap` | null (precision is undefined: no predicted positives) | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=null, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'fuzzy' (false_positives=0, peak RSS=24.0 MB, warm p50=0.96µs)
- `fuzzy`: dominated by 'overlap' (false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### transliteration

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | 1.000 | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | 1.000 | 0 | 24.71µs | 208.9 MB |
| `fuzzy` | 0.961 | 0 | 0.96µs | 24.0 MB |
| `overlap` | 1.000 | 0 | 0.21µs | 22.5 MB |

**On the frontier** (no stated constraints — none of these is dominated by another): `overlap` (f1=1.000, false_positives=0).

- `embed-fp32`: dominated by 'embed-int8' (f1=1.000, false_positives=0, peak RSS=208.9 MB, warm p50=24.71µs)
- `embed-int8`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)
- `fuzzy`: dominated by 'overlap' (f1=1.000, false_positives=0, peak RSS=22.5 MB, warm p50=0.21µs)

### near-miss negative

| arm | f1 | false_positives | warm p50 | peak RSS |
|---|---|---|---|---|
| `embed-fp32` | null (precision is undefined: no predicted positives) | 0 | 24.87µs | 275.3 MB |
| `embed-int8` | null (recall is undefined: no actual positives) | 1 | 24.71µs | 208.9 MB |
| `fuzzy` | null (recall is undefined: no actual positives) | 5 | 0.96µs | 24.0 MB |
| `overlap` | null (recall is undefined: no actual positives) | 115 | 0.21µs | 22.5 MB |

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
