# ADR-0017 — `tokenizers` as the standalone tokenizer for the neural profile

**Status:** Accepted · **Date:** 2026-08-13 · **Refines:** ADR-0002, ADR-0014 · **Serves:** RFC-0001

## Context

The embedding arms take a name string and must produce what the ONNX graph actually
consumes: `input_ids`, `attention_mask` and `token_type_ids`, each `int64`, as recorded
against the exported artefact in `benchmarks/20260812T181752Z-quantization-spike.json`. A
graph that scores embeddings has no opinion about how a string becomes those three arrays —
something upstream of the graph has to hold that opinion, and an inference-only install
needs it too. `EmbeddingScorer.prepare` (RFC-0001) is where it lives.

The spike that produced the artefact used `transformers.AutoTokenizer`, because the spike
also needed `transformers` to run the export itself. That is not evidence it belongs in the
arm that only runs inference: the spike's tokenizer choice was downstream of a tool already
present for an unrelated reason, not a decision about what a reader installs to score a
name.

Three candidates were considered, and each is rejected below on what it would do to the
install profile, the graph, or the accuracy comparison — never on the effort required to
build it.

## Decision

**`tokenizers` — the standalone Rust library, not `transformers` — goes into the `neural`
extra.** The arm constructs it with `Tokenizer.from_file("tokenizer.json")`, reading the
tokenizer artefact already produced alongside the model, and never calls
`Tokenizer.from_pretrained`.

That last clause is load-bearing, not a style choice. `Tokenizer.from_pretrained` reaches
`huggingface_hub`'s hub client to resolve and fetch a repo; `Tokenizer.from_file` reads a
path this machine already has. Measured directly: importing `tokenizers` alone leaves
`huggingface_hub`, `requests`, `urllib3` and `tqdm` all absent from `sys.modules`, and the
import itself costs about 7.8 ms. `from_pretrained` is the only member of the package that
would put a network client on the path a matcher call takes, and this arm's `prepare` never
calls it.

## Alternatives rejected

**`transformers.AutoTokenizer`.** The library already lives in the `export` extra, and the
comment above that extra in `pyproject.toml` states why it is separate: `neural` names the
inference runtime whose cost the benchmark measures, and folding export tooling into it
would change what the measurement means (ADR-0014). `AutoTokenizer` would fold exactly that
tooling back in through the tokenizer alone, and it would do so to reach a wrapper over the
library chosen here anyway: `transformers` lists `tokenizers` among its own dependencies
(`uv.lock`), so `AutoTokenizer` is a layer above `Tokenizer`, not an alternative to it.
Taking that layer means taking nine further packages — `filelock`, `huggingface-hub`,
`numpy`, `packaging`, `pyyaml`, `regex`, `requests`, `safetensors` and `tqdm` — into the
install profile whose import cost the benchmark reports, in exchange for an interface onto
a library the profile would already contain. Issue #92 sharpens the same boundary from a different angle: three open advisories
against `transformers` are reachable today only through `export`, bounded to a maintainer
running export tooling against a checksummed artefact, because nothing under `joinless/`
imports it. Adding `AutoTokenizer` to `neural` would move that exposure into every reader's
inference-time install, which is the install profile ADR-0014 built to be minimal and
verifiable in the first place.

**A hand-written WordPiece implementation over `vocab.txt`.** Removes the dependency
entirely, and trades it for a tokenizer that has to agree with the one the model was
trained under, exactly, or it does not. This is a measurement project, and the failure mode
is the kind measurement is bad at catching: a reimplementation that disagrees on some
subword boundaries does not raise or fail a test — it produces slightly different token ids,
which produce a slightly different embedding, which produces a slightly lower score on
exactly the pairs closest to the decision threshold. There is no oracle in this benchmark
that would flag that as *the tokenizer is wrong* rather than *the embedding arm is weaker
than expected* — it would be read as an accuracy result about the model, when it was
actually a bug in code this project wrote. ADR-0002 constraint 3 already rules out any
uncontrolled difference between what a reader's tokenizer does and what the model's own
tokenizer does; a reimplementation is exactly that kind of difference, chosen for no reason
the measurement needs.

**`onnxruntime-extensions`, tokenizing inside the graph.** Fuses tokenization into the ONNX
graph as a custom op, so a name goes in and a score comes out with no separate tokenizer
call. That changes two things RFC-0002 depends on being separate. First, the Metrics table
lists tokenizer load and warm scoring as distinct, separately attributable phases, and
Method step 1 requires per-record preparation cost kept apart from per-comparison cost;
fusing tokenization into the graph collapses tokenizer-load cost into session creation and
per-comparison tokenization cost into warm scoring, and neither figure can be pulled back
apart once they share one graph execution. Second, it changes the
artefact: the model file recorded with a revision and a checksum (ADR-0013's fourth rule)
would stop being weights-and-graph and become weights-and-graph-and-preprocessing, which is
a different artefact than the one the quantization spike already produced and recorded —
re-exporting it is a new decision, not a drop-in swap of the tokenizer this arm calls.

## Consequences

- The no-network property ADR-0013 requires of match-time execution extends cleanly to
  tokenization: `import tokenizers` reaches neither `huggingface_hub` nor `requests`, and
  `Tokenizer.from_file` never does either, so a classical-only process and a neural process
  that only scores are both still processes that touch no network.
- The dependency is nonetheless present in the installed set, and stating the no-network
  property should not be read as stating its absence: `tokenizers` depends on
  `huggingface_hub`, which depends on `requests` (`uv.lock`), and both sit in
  `site-packages` of any `neural` install whether or not `from_pretrained` is ever called.
  What is true is narrower and checkable — neither import nor exercise on the path this
  arm's code takes — not that the packages are missing.
- The tokenizer import belongs inside the embedding arm's own construction path, behind the
  same lazy-import boundary ADR-0014 already draws around ONNX Runtime itself. A base
  install that never constructs an embedding arm never imports `tokenizers` either, and the
  `sys.modules` invariant ADR-0014 states for the runtime holds for the tokenizer by the
  same mechanism, not a second one.
- Licence is Apache-2.0 (`tokenizers`' own classifier), compatible with this project's MIT
  licence (ADR-0005).
- `tokenizer.json` ships as one of the artefact files alongside the ONNX graph, so its
  presence and checksum are governed by the same `ArtifactRequirement` /
  `verify_artifact` path (`joinless/measurement.py`) that already refuses a mismatched or
  missing model file rather than fetching a replacement — one checksum-verification
  mechanism for every artefact file an arm depends on, not a second one specific to the
  tokenizer.
