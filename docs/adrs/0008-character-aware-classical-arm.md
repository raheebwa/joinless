# ADR-0008 — Add a character-aware classical arm

**Status:** Accepted · **Date:** 2026-08-12 · **Extends:** ADR-0003

## Context

ADR-0003 established that the classical matcher is a permanent first-class arm because a
comparison needs a maintained control. It assumed one classical matcher: the token-overlap
coefficient inherited from the prior-art resolver.

That assumption is weak in a specific, well-known way. **Token overlap is character-blind.**
`BRIGHTWATR` and `BRIGHTWATER` share no token, so the coefficient is zero and the pair is
rejected. Every single-character typo, transposition, and concatenation is invisible to it.

Meanwhile, the classical methods practitioners actually use for name matching are
character-aware: Jaro-Winkler (the canonical choice in census and administrative linkage),
edit-distance ratios, and q-gram/trigram similarity. Comparing a transformer against token
overlap alone and calling the result "classical versus neural" would be comparing against
a matcher nobody defends for this task.

The consequence is that the comparison cannot separate *embeddings beat classical name
matching* from *embeddings beat one weak heuristic*, which is the question the benchmark
exists to answer.

## Decision

**Add a character-aware classical arm using `rapidfuzz`.** Four arms total:

| Arm | Matcher | Dependencies |
|---|---|---|
| `overlap` | Token-overlap coefficient — the inherited transparent floor | standard library |
| `fuzzy` | Character-aware similarity (Jaro-Winkler / token-set ratio) | `rapidfuzz` |
| `embed-fp32` | Stock sentence-embedding similarity | ONNX Runtime |
| `embed-int8` | Same model, quantized | ONNX Runtime |

`rapidfuzz` is MIT-licensed, C++-backed, and operates at microsecond scale — it does not
meaningfully move the cost axis, which is precisely what makes it a hard opponent.

## Consequences

- **The comparison becomes defensible.** "Classical" now means something a practitioner
  would recognise, not a strawman.
- **It raises the chance of the more interesting result.** Stock sentence embeddings are
  trained on sentence semantics, not short proper nouns, and are known to conflate
  semantically adjacent but distinct entities — *Kampala Traders Ltd* and *Kampala
  Merchants Ltd* are near-identical in embedding space and different companies. A
  microsecond-scale character matcher holding its own against a quantized transformer at a
  fraction of the cost is a considerably better finding than confirming what
  [LinkTransformer](https://arxiv.org/abs/2309.00789) already published.
- **The zero-dependency property now belongs to `overlap` alone.** State that precisely
  rather than claiming it for the project.
- One more arm across every slice, seed and timing boundary. Accepted: it is the arm that
  makes the other three worth reporting.
- `rapidfuzz`'s licence and version are pinned and recorded with each run like any other
  dependency.

## Alternatives rejected

**Keep three arms.** Cheaper, but the resulting comparison cannot separate the model's
advantage over character-aware matching from its advantage over a character-blind
heuristic. The saving costs the result its meaning.

**Add several classical matchers** — phonetic, Monge-Elkan, multiple q-gram variants.
Turns a decision experiment into a survey of string metrics, which is a different project
and one with substantial existing literature. One credible character-aware representative
is enough to make "classical" honest.

**Replace `overlap` with `fuzzy`.** Loses the inherited production-derived matcher, which
is the arm with a real provenance story. Both stay.
