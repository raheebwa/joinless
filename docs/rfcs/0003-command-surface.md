# RFC-0003 — Command surface, and the optional viewer

**Status:** Draft · **Date:** 2026-08-13 · **Implements:** PRD FR-11, FR-11a

## Summary

The command line is the reproduction surface. Five commands — `resolve`, `compare`,
`benchmark`, `report`, `doctor` — and an optional read-only viewer over records those
commands already wrote.

## Motivation

A benchmark table communicates the trade-off to someone already invested enough to read
it. It does not communicate what the difference *feels* like — that a name pair the
classical matcher rejects is one the neural matcher accepts, and that the neural answer
arrives perceptibly later.

There is also a claim in the README that is easy to assert and hard to believe: nothing
leaves the machine. A command that runs with the network off makes that checkable in a few
seconds.

The demonstrated action should be **the same documented command another developer runs**.
Anything else demonstrates a second implementation, and the reader has no way to tell
which one produced the published numbers.

## Design

### The commands

| Command | Does | Never does |
|---|---|---|
| `resolve` | Merges two record sets into one | Measure anything |
| `compare` | Scores one name pair under a chosen arm | Write a run record |
| `benchmark` | Runs the protocol and writes one record to `benchmarks/` | Print a number it did not write |
| `report` | Renders a record | Re-measure, or compute a figure the record lacks |
| `doctor` | Reports the execution environment | Change it |

`doctor` reports architecture, execution provider, installed profile and offline status.
Those four exist so that ADR-0006's Arm64 and CPU-only claim, and the no-network property,
are **checkable rather than asserted** — a reader who does not trust the README can run one
command and see the same facts the run records carry.

`report` never measures. A renderer that can compute a missing figure will eventually
compute one that was never recorded, and the resulting number traces to nothing.

Every command runs with no network interface available. That is a test, not an aspiration.

### The optional viewer

A read-only view over records already written, bounded by a must-not list:

- **no authoritative benchmarking inside a web request** — a request handler competes with
  the browser for the same cores, so a figure measured there is not the figure the
  documented command produces
- **no editing of recorded results or thresholds** — a record is evidence, and evidence
  that can be edited in the interface that displays it is not evidence
- **no implicit model download** — a viewer that fetches on demand breaks the property the
  project exists to demonstrate
- **no transmission of names** — records may carry `pair_id` precisely so that raw names
  need not leave the machine
- **never required to reproduce a result** — every figure is reachable from the command
  line alone, or the viewer has become the reproduction surface

Not a framework, not a build step, no bundler. If it acquires state management, routing or
a component library, that is the signal it has outgrown its purpose and should be cut back
rather than continued.

## Open questions

1. ~~Terminal UI or local web page?~~ **Settled: the command line, with the viewer
   optional and subordinate to it.** PRD FR-11 fixes the command line as the reproduction
   surface, and the reasoning is that the demonstrated action should be the same documented
   command another developer runs — a separate interactive front end demonstrates a second
   implementation, and a reader cannot tell which one produced the published numbers.

   A graphical front end also adds state, assets and frontend dependencies without
   improving the measurement, and someone choosing between name matchers is already in a
   terminal. The viewer survives as FR-11a: read-only, over records the commands wrote,
   and never required to reproduce anything.
2. ~~Should it expose the threshold as a live control?~~ **Settled: in `compare` only.**
   Thresholds are where most of the behaviour lives, so being able to move one and watch
   the decision change is worth having on a single pair.

   Benchmark thresholds are a different object: ADR-0011 fixes them by calibration, on
   calibration data alone, and freezes them. A control that moved those would make the
   reported figure a function of whatever the last person dragged, and no record could say
   which value produced it. So the viewer has no threshold control at all.
3. ~~Should it accept a small pasted batch to make the latency difference perceptible?~~
   **Settled: no. Single comparisons stay single.**

   Batching to make a difference *feel* larger is measurement design, and doing it in an
   interactive surface produces a number under conditions no run record describes.
   Interactive timings are illustrative and are never benchmark evidence; the figures that
   count come from `benchmark`, under RFC-0002's protocol, in a fresh process.

   The latency difference being imperceptible on one pair is itself part of the finding.
   Hiding that behind a batch size chosen to make it visible would be arguing for a
   conclusion rather than reporting one.

## Alternatives considered

**Skip the viewer, ship the table.** The measurement is the real contribution, and the
commands already make it reproducible. Retained only because `report` has to render a
record somehow, and a read-only view over records that already exist costs little and adds
a way to see per-family results side by side.

**Record a video instead.** A video shows behaviour but cannot be verified. Anyone can run
a command themselves, which is the whole reason the command line is the surface.

**A graphical front end as the primary surface.** Rejected above — it would be a second
implementation of the thing being measured, and the published numbers would not obviously
come from the code a reader runs.
