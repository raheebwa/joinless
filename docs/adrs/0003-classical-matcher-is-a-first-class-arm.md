# ADR-0003 — The classical matcher is a first-class arm, not a legacy path

**Status:** Accepted · **Date:** 2026-08-12

**Superseded in part by ADR-0008.** The consequence below reading *"Carrying two
implementations forever is a real maintenance cost"* assumed one classical implementation
next to the neural family. ADR-0008 adds a second classical implementation (`fuzzy`),
making it three implementations, four arms. The decision and the rest of this record's
reasoning are unaffected; only that count is superseded.

## Context

Once an embedding matcher exists, the obvious move is to treat the token-overlap matcher
as the old way — kept for compatibility, quietly deprecated, eventually deleted.

That would destroy the only thing this project has to say.

## Decision

**The classical matcher is a permanent, first-class, supported arm.** It is measured in
every benchmark run, it is a valid production configuration, and it is never marked
deprecated.

The project's claim is a *comparison*. A comparison needs a control arm that is genuinely
maintained — not a strawman kept around to lose.

## Consequences

- Every benchmark reports every arm. No run publishes neural numbers alone.
- The classical matcher gets the same test rigour and the same optimization attention as
  the neural arms. Tuning the model while leaving the baseline naive would rig the result.
- Configurations where the classical matcher wins are documented as **recommendations**,
  not as caveats.
- Carrying two implementations forever is a real maintenance cost. Accepted deliberately:
  it is the cost of the project's thesis.
- The zero-dependency path stays available for users who cannot or will not add an
  inference runtime — a genuine deployment constraint, not a hypothetical.

## Alternatives rejected

**Deprecate after the neural arm lands.** Assumes the conclusion. The measurement may
well show the classical matcher winning under realistic client latency budgets — that
outcome must remain expressible.

**Keep it as a fallback only.** "Fallback" framing invites under-maintenance, and an
under-maintained control arm makes the whole comparison dishonest.
