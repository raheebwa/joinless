# RFC-0003 — Local demo application

**Status:** Draft · **Date:** 2026-08-12 · **Implements:** PRD FR-11, M6

## Summary

A small local application that runs the resolver interactively, so the behaviour can be
seen rather than only read about in a results table.

## Motivation

A benchmark table communicates the trade-off to someone already invested enough to read
it. It does not communicate what the difference *feels* like — that a name pair the
classical matcher rejects is one the neural matcher accepts, and that the neural answer
arrives perceptibly later.

There is also a claim in the README that is easy to assert and hard to believe: nothing
leaves the machine. A demo that runs with the network off makes that checkable in a few
seconds.

## Design

A single-screen local application:

- Two text inputs — a name from each source, optionally with coordinates
- A matcher selector — `overlap` / `fuzzy` / `embed-fp32` / `embed-int8`
- Live output: match or no match, the underlying score, and the elapsed time for the
  comparison
- A side-by-side mode running all four arms on the same input at once, so disagreements
  are visible directly
- Zero network usage after model load

Implementation should be the smallest thing that works — a local web UI served from the
package, or a terminal interface. Not a framework, not a build step, no bundler.

The demo exists to make the resolver's behaviour visible; it is not a product surface. If
it starts acquiring state management, routing or a component library, that is the signal
it has outgrown its purpose and should be cut back rather than continued.

## Open questions

1. Terminal UI or local web page? A web page screenshots better and is easier for a
   reader to run; a terminal UI has no asset story at all and cannot accidentally make a
   network request.
2. Should it expose the threshold as a live control? Illuminating — thresholds are where
   most of the behaviour lives — but invites the impression that results were tuned after
   the fact. If included, the benchmark thresholds must be clearly marked as fixed.
3. Should it accept a small pasted batch rather than single pairs, to make the latency
   difference perceptible? A single comparison is too fast to feel.

## Alternatives considered

**Skip the demo, ship the table.** Cheaper, and the measurement is the real contribution.
Rejected because the no-network property is central to the project's argument and is far
more convincing demonstrated than asserted.

**Record a video instead of shipping an app.** A video shows the behaviour but cannot be
verified. Anyone can check a running app themselves.
