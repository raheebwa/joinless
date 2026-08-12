# ADR-0001 — Greenfield package; prior art cited, not forked

**Status:** Accepted · **Date:** 2026-08-12

## Context

The classical keyless-resolution algorithm this project builds on already exists as a
published MIT reference implementation:
[`entity-resolution-no-keys`](https://github.com/raheebwa/entity-resolution-no-keys) —
grid bucketing, token-overlap name matching, haversine tie-break, dual hashing for
coordinate-less rows. Roughly 300 lines, standard library only.

This project asks a different question — *when is a neural matcher worth its cost on a
client device* — and needs a pluggable matcher interface, an inference runtime, a
labelled evaluation set, and a benchmark harness. None of those belong in a repository
whose value is being a minimal, dependency-free reference.

Three options: fork it, take it as a dependency, or start fresh and cite it.

## Decision

**Start fresh. Cite the prior work as prior art.**

The classical matcher is reimplemented here against the documented behaviour of the
reference implementation, not copied from it. `entity-resolution-no-keys` is credited in
the README and here, and is neither a fork parent, a submodule, nor a dependency.

## Consequences

- This repository has its own history and its own design, unconstrained by the reference
  implementation's deliberate zero-dependency minimalism.
- The reference implementation stays clean — it does not acquire an ONNX Runtime
  dependency to serve a question it was not built to answer.
- Reimplementing duplicates work already done once, and risks silent behavioural drift
  from the reference. Mitigated by property tests asserting the documented invariants —
  coordinate-less rows retained, identifiers stable and namespaced under reordering —
  rather than by trusting the two implementations to look alike. Complexity is not among
  them: property tests cannot establish asymptotic behaviour, so candidate-comparison
  counts are instrumented and measured across increasing input sizes and a deliberately
  dense bucket instead.
- Attribution obligation is permanent: the README and this record must keep the credit
  even as the code diverges.

## Alternatives rejected

**Fork.** Inherits history and framing, and implies the fork will track upstream. It will
not — the two answer different questions.

**Depend on it.** It is published as a readable reference implementation, not a versioned
library with a stable API. Depending on it would freeze its internals as a contract and
constrain both repositories.
