# ADR-0012 — Bring-your-own labelled pairs is the only scope addition

**Status:** Accepted · **Date:** 2026-08-12

## Context

A benchmark whose result can only be reproduced on the author's fixtures answers a question
about the author's fixtures. Results on a synthetic corpus are bounded to that corpus by
construction (ADR-0004, ADR-0011), so a reader with their own names cannot learn from this
project whether an embedding matcher would pay *for them* — which is the only version of
the question they actually have.

The transferable artefact is therefore not the number. It is the **procedure**: identical
pairs, identical splits, identical threshold governance, identical timing boundaries, run
across the same four arms.

The established tools in this space — [Splink](https://moj-analytical-services.github.io/splink/),
[dedupe](https://github.com/dedupeio/dedupe), [Zingg](https://github.com/zinggAI/zingg),
[OpenRefine](https://openrefine.org/) — occupy full-system linkage: blocking, multi-field
probabilistic matching, clustering, active learning, review workflows. Growing into that
space would duplicate mature work and would dilute the one measurement this project exists
to make.

## Decision

**Add exactly one surface: a labelled-pairs input path. Add nothing else.**

Input schema:

| Field | Required | Meaning |
|---|---|---|
| `left_name` | yes | name from source A |
| `right_name` | yes | candidate name from source B |
| `label` | yes | `1` same entity, `0` different entity |
| `category` | no | user-defined perturbation family for per-family reporting |
| `pair_id` | no | user-owned identifier, so reports need not echo raw names |

**The binding architectural constraint: the built-in synthetic corpus is one producer of
this schema, not a parallel path.** The synthetic benchmark and a user-supplied file enter
the same validation, the same split logic, the same threshold governance, the same arms and
the same record writer. If the two ever require separate code paths, the design has gone
wrong and the divergence is a defect, not a feature.

User data stays local: never uploaded, never committed, and excluded from the durable run
record by default — reports identify pairs by `pair_id` unless an explicit local
error-analysis mode is requested.

## Consequences

- The contribution becomes a method rather than a number, and the method is testable by
  anyone on data that matters to them.
- Every rule in ADR-0011 applies to user data automatically, because it is the same code.
  A user cannot accidentally tune on their test set using this tool.
- The schema is a compatibility surface once published. It is deliberately minimal — three
  required fields — to keep it cheap to honour.
- This is product surface, not algorithmic scope. No matcher family, blocking strategy,
  model, training loop, persistence layer or connector is added.
- Validation and clear failure messages become real work, since malformed input is now a
  normal case rather than an internal one.

## Explicitly rejected

Multiple blocking strategies · address and phone comparison · clustering · active learning ·
review queues · connectors · model training or fine-tuning · a model zoo · serving
infrastructure · schema mapping.

Mature projects own each of these. Adding any of them enlarges the system while leaving the
measurement exactly where it was.

**The tripwire:** if the labelled-pairs path starts to need schema mapping, candidate
generation, a persistence layer or a UI, the rejected scope is re-entering through it. Stop
and reconsider rather than continue.
