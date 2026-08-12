# ADR-0011 — Evaluation protocol integrity

**Status:** Accepted · **Date:** 2026-08-12 · **Binds:** RFC-0002

## Context

Every matcher here compares a similarity value against a threshold. Where that threshold is
chosen determines the reported accuracy, and there are three ways to get a number that
looks authoritative and means nothing.

**Threshold overfitting.** Selecting a threshold that maximises F1 on the same pairs used
to report F1 makes the reported figure an upper bound on a quantity that does not
generalise. Worse for a comparison: the amount of inflation differs per arm — an arm with a
sharper similarity distribution gains more from tuning — so the *ranking* between arms
becomes an artefact of the tuning procedure rather than a property of the matchers.

**Unequal tuning effort.** Comparing one arm at its tuned optimum against another at a
library default measures which arm received attention.

**Fixture composition determining the outcome.** The fixtures are synthetic and authored by
someone who knows what each matcher does (ADR-0004). A set weighted toward abbreviation and
transliteration favours embeddings; one weighted toward high-token-overlap near-misses
favours precise character comparison. Whoever picks the mixture picks the winner, and an
aggregate score conceals that entirely.

None of these are presentational concerns. Each produces a number that is arithmetically
correct and substantively false.

## Decision

**Four rules, binding on every reported result.**

**1. Three-way split.** The corpus is generated into disjoint roles:

| Role | Use |
|---|---|
| development | designing and debugging fixtures and matchers; freely inspected |
| calibration | threshold selection only; never reported |
| sealed test | reported once, after thresholds are frozen |

No model is trained, so this is not a train/test split — it is a *tuning*/test split, and
tuning is where the inflation enters.

**2. Threshold governance.** Each arm's threshold is selected on calibration data alone, by
a documented and identical procedure, and frozen before the sealed test runs. The procedure
and the selected value are recorded in the run record. Every arm receives the same
selection treatment.

**3. Per-family reporting, multiple seeds.** Results are reported per perturbation family —
exact, formatting, word order, abbreviation, character noise, semantic alias,
transliteration, near-miss negative — never as a single aggregate. The corpus is generated
under several deterministic seeds and variation across seeds is reported, so a result that
depends on one draw is visible as one.

**4. The corpus must contain families where cheap comparison is expected to win.** A
fixture set built only from cases the model handles well measures the author's expectations
rather than the matchers. Without families where token or character comparison is expected
to succeed — and where an embedding model is expected to overmatch, such as semantically
adjacent but distinct entities — the benchmark cannot distinguish *the model is better*
from *only questions the model answers were asked*. The expected winner per family is
recorded **before** the run, and families where the outcome contradicts the expectation are
reported as findings rather than smoothed away.

## Consequences

- The reported number is lower than a tuned-on-test number would be, and is the one that
  generalises to the disclosed distribution.
- Threshold selection becomes a documented artefact, reproducible and criticisable.
- Fixture composition becomes an explicit, reviewable design decision instead of an
  invisible one.
- Rule 4 means the corpus is built partly from cases where the newer method is expected to
  do worse. That is the point: a comparison that cannot come out the other way is not a
  comparison.
- More generation and bookkeeping than a single-pass benchmark. This is the cost of the
  result meaning anything.
- A pre-registered expectation that turns out wrong is a genuine finding and is reported as
  one — it is evidence the benchmark can surprise its author.

## Alternatives rejected

**Single set, tuned and reported together.** Simplest, and produces the specific failure
this record exists to prevent.

**Fixed thresholds from the literature, no calibration.** Avoids overfitting but measures
whether published defaults suit this distribution, which is a different question and a less
useful one.

**Aggregate F1 only.** Compact, and hides the mechanism by which fixture composition
determines the ranking. The per-family breakdown *is* the result; the aggregate is a
summary of it.
