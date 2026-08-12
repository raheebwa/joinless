# RFC-0001 — Matcher protocol

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** PRD FR-6, FR-7, FR-8, FR-10

## Summary

Define the seam that lets any name-matching strategy be substituted without touching the
resolver, so that comparing strategies is a configuration change rather than a rewrite.

## Motivation

The resolver's candidate generation (coordinate grid), tie-break (haversine distance),
merge policy and identifier scheme are orthogonal to *how two names are compared*. Only
the name comparison differs between the arms. If that comparison is inlined into
the resolver, each arm becomes a separate resolver and the comparison becomes invalid —
differences in the surrounding code would contaminate the result.

## Design

A single narrow protocol:

```python
class Matcher(Protocol):
    def prepare_all(self, names: Sequence[str | None]) -> list[Prepared]:
        """Normalise raw names into whatever this matcher compares.

        Batched and called once per record at load time, never inside the
        comparison loop (ADR-0009). Returns opaque, matcher-specific values:
        token frozensets for the classical arms, embedding vectors for the
        neural ones.
        """

    def prepare(self, name: str | None) -> Prepared:
        """Single-record preparation.

        Retained deliberately as the naive control the hoist is measured
        against (ADR-0009), not as the production path.
        """

    def score(self, a: Prepared, b: Prepared) -> float:
        """Similarity in [0.0, 1.0]. Higher means more likely the same entity.

        Returns a score, never a decision. Thresholding is the caller's
        responsibility (ADR-0011) — a matcher that decided internally could not
        be calibrated, could not produce a precision/recall curve, and could not
        be compared against another arm at a matched operating point.
        """

    @property
    def name(self) -> str:
        """Stable identifier used in benchmark records."""
```

Comparability across arms requires the range to be fixed, not the semantics: an overlap
coefficient, a Jaro-Winkler ratio and a cosine similarity are all in `[0, 1]` but are not
interchangeable quantities. Thresholds are therefore selected **per arm** and are never
transferred between them.

The split between `prepare` and `score` is load-bearing. Embedding a name is
comparatively expensive and must happen **once per record**, not once per comparison —
the grid scans each listing against up to nine cells, so a name participates in many
comparisons. Folding embedding into `score` would measure the wrong thing entirely and
make the neural arms look far worse than they are.

The resolver holds a `Matcher` and calls only these members. It never inspects
`Prepared`.

### Implementations

| Class | `prepare` returns | `score` computes |
|---|---|---|
| `OverlapMatcher` | `frozenset[str]` of normalised tokens | overlap coefficient `|A ∩ B| / min(|A|,|B|)` |
| `FuzzyMatcher` | normalised string | character-aware ratio via `rapidfuzz` (ADR-0008) |
| `EmbeddingMatcher` | `np.ndarray` — L2-normalised mean-pooled embedding | cosine similarity, rescaled from `[-1, 1]` to `[0, 1]` |

`EmbeddingMatcher` is constructed with a model path, a threshold, and a runtime session;
the fp32 and int8 arms are the same class with different model artefacts. That keeps the
comparison honest — the only difference between those two arms is the weights.

## Open questions

1. ~~Should the protocol return a score rather than a bool?~~ **Settled: it returns a
   score.** Calibration (ADR-0011) requires a continuous value — a matcher that thresholds
   internally cannot be tuned on calibration data, cannot yield a precision/recall curve,
   and cannot be compared against another arm at a matched operating point.

   The resolver still applies `score >= threshold` and then breaks ties by distance,
   preserving the prior-art tie-break semantics exactly. Ranking candidates by name
   similarity instead of distance would change resolver behaviour and remains out of scope.
2. ~~Should `prepare` be batched?~~ **Settled by ADR-0009.** Batched preparation is
   contractual — per-record transformer calls are not a representative client
   implementation and would understate every neural arm. The protocol gains
   `prepare_all(names) -> list[Prepared]`, and the per-record path is retained only as the
   naive control the hoist is measured against.
3. ~~Where does the threshold live?~~ **Settled: outside the matcher**, in configuration,
   set from calibration output (ADR-0011). The scale remains matcher-specific and
   non-transferable, but ownership sits with the caller so that selection is an auditable
   artefact rather than a constant buried in an implementation.

## Alternatives considered

**Pass a comparison function.** Too thin — there is per-record preparation state that a
bare function cannot hold without a closure, and closures make the benchmark's
per-arm accounting opaque.

**Subclass the resolver per matcher.** Exactly the contamination this RFC exists to
prevent.
