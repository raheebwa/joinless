# ADR-0005 — MIT licence

**Status:** Accepted · **Date:** 2026-08-12

## Context

The project is intended to be read, copied, and reused — its value is the measurement and
the pattern, not exclusivity. It also reimplements an algorithm already published by the
same author under MIT (ADR-0001).

## Decision

**MIT.** `LICENSE` at the repository root, detectable by GitHub, visible in the About
section.

Every dependency must carry an MIT/Apache/BSD-compatible licence; this is checked before
a dependency is added. Model weights are **fetched at setup, never committed** — they
carry their own licences, which are recorded in the documentation rather than absorbed
into this repository's terms.

## Consequences

- Maximum reuse, minimum friction. Anyone can lift the harness into their own work.
- Consistent with the prior-art repository, so the reimplementation raises no licence
  question.
- No copyleft protection: someone can build a closed product on this. Accepted — the
  contribution is the published finding, which cannot be un-published.
- Ongoing obligation: any vendored snippet or ported algorithm from elsewhere must be
  licence-checked and credited before it lands.

## Alternatives rejected

**Apache 2.0** — the patent grant is a real advantage, but MIT is shorter, more widely
understood, and matches the prior-art repository. No patent exposure is anticipated here.

**GPL** — actively counterproductive. Copyleft would deter exactly the reuse this project
wants.
