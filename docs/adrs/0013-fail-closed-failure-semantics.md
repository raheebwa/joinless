# ADR-0013 — Fail closed: a failure is recorded as a failure, never degraded into a number

**Status:** Accepted · **Date:** 2026-08-12 · **Extends:** ADR-0011 · **Binds:** RFC-0002

## Context

A measurement harness has two kinds of bad day. It can stop, which is loud and gets fixed.
Or it can carry on with a hole in it and emit a table anyway — and the output of a degraded
run is shaped exactly like the output of a sound one. Once written to a record, the two are
indistinguishable, and the record is what gets read months later.

ADR-0011 sets the rules that make a reported number mean something. Rules without defined
failure behaviour are preferences: each of them has a convenient way to be not-quite-broken
that produces a plausible figure. Four such degradations are specific to this harness, and
each is the behaviour a reasonable implementation reaches for first.

**An empty denominator.** Precision is correct matches over proposed matches. An arm that
proposes nothing has a zero denominator and an undefined precision. Substituting `0.0`
keeps the column aligned and destroys the distinction between two different facts: *the arm
proposed matches and every one was wrong*, and *the arm proposed nothing at all* — which
usually means a mis-set threshold, a failed preparation step, or an empty split. Per-family
reporting (ADR-0011 rule 3) multiplies the opportunity, because a family with no proposals
above threshold is an ordinary occurrence. A table of honest-looking zeros is the result,
and a broken run reads as a bad arm.

**A threshold that saw the sealed test.** ADR-0011 rule 2 requires thresholds selected on
calibration data alone and frozen before the sealed test runs. If violating that produces a
warning, the run still completes, the record is still written, and the figure is still
there to be quoted by someone who never saw the terminal output. The inflation ADR-0011
exists to prevent is back, now with a paper trail that looks like compliance.

**An arm that will not start.** The inference runtime may be absent, a model artefact may
be missing, a platform may have no wheel. Skipping the arm and reporting the rest yields a
table that describes itself as a four-arm comparison and contains two — which is the
failure mode ADR-0003 was written against, arrived at by accident rather than by intent.

**A missing or altered artefact.** Model artefacts are fetched at setup and never committed
(ADR-0005), the run record names each artefact's revision and checksum (RFC-0002 method step
5), and nothing in this project touches the network at match time (PRD G5). An artefact that
is absent, or present with a checksum other than the one the record names, means the run
cannot be the run it claims to be. Fetching a replacement mid-run repairs the symptom while
changing both the measured configuration and the no-network property.

## Decision

**Every failure is either recorded as a first-class outcome or refuses to produce a result.
Nothing degrades quietly into a number.**

| Condition | Behaviour |
|---|---|
| metric denominator is zero | the metric is `null` in the record, with a machine-readable reason and the counts that produced it. Never `0.0` |
| threshold selection touched any sealed-test pair | the run's status is `invalid`; the record is written and kept, and no figure from it is reportable |
| an arm cannot initialise | the arm appears in the results with status `unavailable` and the reason. It is never omitted |
| artefact missing, or checksum ≠ the recorded value | the run aborts before any measurement. No fetch is attempted |

Four points follow from the table.

1. **`null` and `0.0` are different values and stay different everywhere** — in the record,
   in any derived table, and in any aggregate. An aggregate computed across families skips
   nulls and reports how many it skipped; it never coerces them to zero, because a mean
   over silently-zeroed undefined values is a fabricated number.
2. **`invalid` is a property of the run, not a message about it.** It lives in the record
   next to the numbers, so a reader who never saw the run cannot mistake it for a sound
   one. The record is retained rather than discarded — a run that broke a protocol rule is
   evidence about the harness.
3. **`unavailable` keeps the arm in the shape of the result.** A reader can see that four
   arms were configured, which two produced numbers, and why the others did not. That is a
   different statement from a two-arm benchmark, and it must remain a different statement.
4. **Refusing is the correct response to an artefact mismatch**, not a fallback. The
   alternative to a run under unknown conditions is no run, not a run under conditions
   nobody recorded.

## Consequences

- Every reported figure is either a measurement or an explicit absence. There is no third
  category that looks like the first and behaves like the second.
- Failure paths acquire tests of their own: a zero-denominator family, a threshold drawn
  from the wrong split, an uninitialisable arm, and a corrupted artefact each have a
  defined, asserted outcome. These are cheap to test precisely because the outcomes are
  values rather than log lines.
- The record schema carries status fields and reason strings, not only metrics. That is a
  small permanent addition to every record, including the ones where nothing went wrong.
- Runs abort more often than a lenient harness would, and some of those aborts will be
  inconvenient. That is the trade: an inconvenient stop is recoverable, a quietly wrong
  number is not.
- Consumers of a record must handle `null`. Any table generator, any comparison, and any
  future analysis has to distinguish the two cases — which is the point, since a consumer
  that cannot express the distinction would reintroduce the defect downstream.

## Alternatives rejected

**Substitute `0.0` for an undefined metric.** Keeps every column numeric and every
downstream consumer simple. It also encodes a false claim: that an arm proposed matches and
got them all wrong, when it proposed nothing. The simplification is bought entirely with
the reader's ability to tell the two apart.

**Warn on a threshold that saw test data, and continue.** A warning reaches whoever is
watching the process. The record reaches everyone else, forever. If a rule matters enough
to write down, its violation has to survive into the artefact that outlives the session.

**Drop arms that fail to initialise.** Produces a clean table, and a clean table is exactly
the problem: nothing in it indicates that a configured arm is missing. The absence has to
be visible in the same place the presences are.

**Fetch or re-fetch a missing artefact at run time.** Convenient, and it makes the benchmark
depend on a network at the moment it claims not to. It also lets a run silently proceed on a
different artefact than the one under test, which is precisely the confound ADR-0002
constraint 3 exists to exclude.

**A `--strict` flag, lenient by default.** Moves the decision to whoever runs the command,
which means the default behaviour determines what most records contain. A protocol that is
optional is not a protocol; if leniency is ever wanted for local exploration, that is the
development split's job (ADR-0011 rule 1), not a flag on the reported run.
