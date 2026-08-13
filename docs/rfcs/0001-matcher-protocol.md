# RFC-0001 — Scorer protocol and thresholding adapter

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** PRD FR-6, FR-7, FR-8, FR-10

## Summary

Define the seam that lets any name-matching strategy be substituted without touching the
resolver, so that comparing strategies is a configuration change rather than a rewrite.
The seam is two types: a scorer that returns a similarity, and one adapter that turns a
similarity into a decision.

## Motivation

The resolver's candidate generation (coordinate grid), tie-break (haversine distance),
merge policy and identifier scheme are orthogonal to *how two names are compared*. Only
the name comparison differs between the arms. If that comparison is inlined into
the resolver, each arm becomes a separate resolver and the comparison becomes invalid —
differences in the surrounding code would contaminate the result.

## Design

Two types, not one. A `Scorer` answers *how alike are these two names*; a
`ThresholdMatcher` answers *are they the same entity*. The second is written once and
wraps any instance of the first.

### The scorer

```python
class Scorer(Protocol):
    def prepare_all(self, names: Sequence[str | None]) -> list[Prepared]:
        """Normalise raw names into whatever this scorer compares.

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

        Returns a score, never a decision. A scorer that decided internally
        could not be calibrated, could not produce a precision/recall curve,
        and could not be compared against another arm at a matched operating
        point (ADR-0011).
        """

    @property
    def name(self) -> str:
        """Stable identifier used in benchmark records."""
```

### The thresholding adapter

```python
@dataclass(frozen=True)
class ThresholdMatcher:
    """Turns any scorer into a decision. The only place a threshold is applied."""

    scorer: Scorer
    threshold: float

    def matches(self, a: Prepared, b: Prepared) -> bool:
        return self.scorer.score(a, b) >= self.threshold
```

One adapter, one comparison, four arms. This is the whole reason for the split.

With thresholding inside each scorer there would be four implementations of `>=`, and
"substituting a scorer changes nothing but the score" would be a convention every one of
them had to keep. Conventions are kept until they are not, and the failure is silent: an
arm that also adjusted a tie-break, or rounded a score before comparing, would produce a
benchmark number attributed to its *scoring* that was partly its *thresholding*. There is
no measurement that separates those two afterwards.

With one adapter the property is structural rather than promised. It holds because there
is only one place it could be broken, and that place is not reachable from a scorer.

### The invariant

**Substituting a scorer cannot change blocking, identity, distance or merge policy.**

Those four belong to the resolver and are not exposed to the seam:

| Concern | Owner | Reachable from a `Scorer`? |
|---|---|---|
| Candidate generation — coordinate grid bucketing | resolver | no |
| Record identity — the dual hashing scheme | resolver | no |
| Tie-break among candidates over threshold — haversine distance | resolver | no |
| Merge policy for a matched pair | resolver | no |
| Decision — score against threshold | `ThresholdMatcher` | no, it is the adapter |
| Similarity of two prepared names | `Scorer` | yes, and only this |

A scorer receives two `Prepared` values and returns a float. It is handed no record, no
coordinate, no identifier and no configuration, so there is nothing in the surrounding
pipeline it is able to reach even by accident. That is what makes a four-arm comparison
a comparison of four scorers rather than of four resolvers.

The resolver holds a `ThresholdMatcher`. It asks whether a candidate pair matches, and
among those that do, it breaks ties by distance exactly as the prior art does — that
tie-break is resolver policy and stays put when the scorer changes.

## Comparability

Comparability across arms requires the range to be fixed, not the semantics: an overlap
coefficient, a Jaro-Winkler ratio and a cosine similarity are all in `[0, 1]` but are not
interchangeable quantities. Thresholds are therefore selected **per arm** and are never
transferred between them.

The split between `prepare` and `score` is load-bearing. Embedding a name is
comparatively expensive and must happen **once per record**, not once per comparison —
the grid scans each listing against up to nine cells, so a name participates in many
comparisons. Folding embedding into `score` would measure the wrong thing entirely and
make the neural arms look far worse than they are.

A scorer is called only through these members, and `Prepared` is never inspected by
anything but the scorer that produced it.

### Implementations

| Class | `prepare` returns | `score` computes |
|---|---|---|
| `OverlapScorer` | `frozenset[str]` of normalised tokens | overlap coefficient `|A ∩ B| / min(|A|,|B|)` |
| `FuzzyScorer` | normalised string | character-aware ratio via `rapidfuzz` (ADR-0008) |
| `EmbeddingScorer` | `tuple[float, ...] \| None` — L2-normalised mean-pooled embedding, `None` for a blank name | cosine similarity, rescaled from `[-1, 1]` to `[0, 1]` |

`EmbeddingScorer` is constructed with a name, an already-built tokenizer and an
already-opened runtime session — and no threshold, which sits in the adapter. Resolving a
model path, loading the tokenizer and opening the session is the caller's job (one factory
function per arm); the class itself holds only what it needs to prepare and score. The
fp32 and int8 arms are the same class with different model artefacts. That keeps the
comparison honest — the only difference between those two arms is the weights.

## Open questions

1. ~~Should the protocol return a score rather than a bool?~~ **Settled: it returns a
   score.** Calibration (ADR-0011) requires a continuous value — a scorer that thresholds
   internally cannot be tuned on calibration data, cannot yield a precision/recall curve,
   and cannot be compared against another arm at a matched operating point.

   `ThresholdMatcher` applies `score >= threshold`, and the resolver then breaks ties by
   distance, preserving the prior-art tie-break semantics exactly. Ranking candidates by name
   similarity instead of distance would change resolver behaviour and remains out of scope.
2. ~~Should `prepare` be batched?~~ **Settled by ADR-0009.** Batched preparation is
   contractual — per-record transformer calls are not a representative client
   implementation and would understate every neural arm. The protocol gains
   `prepare_all(names) -> list[Prepared]`, and the per-record path is retained only as the
   naive control the hoist is measured against.
3. ~~Where does the threshold live?~~ **Settled: outside the scorer**, as a field of
   `ThresholdMatcher`, set from calibration output (ADR-0011). The scale remains
   scorer-specific and non-transferable, but ownership sits in one named place so that
   selection is an auditable artefact rather than a constant buried in four
   implementations.

## Alternatives considered

**Pass a comparison function.** Too thin — there is per-record preparation state that a
bare function cannot hold without a closure, and closures make the benchmark's
per-arm accounting opaque.

**Subclass the resolver per scorer.** Exactly the contamination this RFC exists to
prevent.

**One type that scores and decides.** The shape this RFC replaces. It reads as simpler
because it is one class instead of two, and it costs the only property that makes a
four-arm comparison mean anything: that the arms differ in their scoring and in nothing
else. A benchmark built on it reports differences it cannot attribute.
