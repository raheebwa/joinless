# RFC-0005 — Labelled-pairs interface

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** ADR-0012

## Summary

One input schema and one command, so the benchmark protocol can be run against a user's own
labelled pairs. The built-in synthetic corpus becomes one producer of that schema rather
than a separate path.

## Motivation

Results on a synthetic corpus are bounded to that corpus by construction (ADR-0004,
ADR-0011). A reader with their own names cannot learn from a published table whether an
embedding matcher would pay for *their* data — which is the only version of the question
they have. The transferable artefact is the procedure, not the number.

## Schema

CSV or JSONL:

| Field | Required | Type | Meaning |
|---|---|---|---|
| `left_name` | yes | string | name from source A |
| `right_name` | yes | string | candidate name from source B |
| `label` | yes | `0` \| `1` | `1` same entity, `0` different entity |
| `category` | no | string | user-defined perturbation family, for per-family reporting |
| `pair_id` | no | string | user-owned identifier, so reports need not echo raw names |

Three required fields, deliberately. The schema is a compatibility surface once published,
so it should be cheap to honour and hard to get wrong.

## Command

```text
joinless benchmark --pairs my-pairs.csv --output run.json
```

Behaviour:

1. validate schema, types and label domain; fail with a message naming the row and column
2. split deterministically into calibration and sealed test (ADR-0011) unless the file
   supplies its own split column
3. run every arm over identical pairs
4. select each arm's threshold on calibration data alone, then freeze
5. report precision, recall and F1 on the sealed split, per `category` where supplied
6. report cost per arm using the same timing boundaries as the built-in benchmark
7. write a machine-readable record plus a Markdown decision table

## Privacy

User data is read and never leaves the machine: not uploaded, not copied into the
repository, not written into the durable record. Reports identify pairs by `pair_id`, or by
index where none is supplied.

Raw names appear in output only under an explicit `--include-names` flag intended for local
error analysis. The default is chosen so that a record can be shared without re-reading it
first.

## The constraint that matters

**The synthetic corpus is a producer of this schema, not a parallel implementation.** The
generator emits rows in exactly this shape and feeds the same validation, splitting,
calibration, scoring and record-writing code.

If the two ever require separate code paths, that is a defect. It means either the schema
cannot express what the benchmark needs, or the benchmark has grown behaviour that only
works on data whose generation it controls — and the second failure would silently
invalidate the claim that a user's run is comparable to the published one.

## Open questions

1. Should user-supplied splits be honoured in v1, or only deterministic generated ones?
   Prefer generated-only unless honouring a supplied split is trivial — a user-chosen split
   is one more way for calibration and test to overlap without anyone noticing.
2. What is the minimum viable pair count before a reported F1 is meaningful? The command
   should warn below a stated floor rather than print a confident number from twenty rows.
3. Should class imbalance be reported and warned on? A file that is 95% negatives produces
   a high-looking precision for almost any matcher.
4. Does the decision table need a recommendation line, or only the measurements? A
   recommendation requires assuming a false-positive cost the tool cannot know; proposal is
   to require the user to state a precision floor and report which arms clear it.
