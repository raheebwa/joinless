## What this changes

<!-- The behaviour before, and the behaviour after. -->

## Which record it implements or amends

<!--
Name the ADR in docs/adrs/ or the RFC in docs/rfcs/ that this change carries out, for
example "implements RFC-0001" or "amends ADR-0008".

A change that answers "none" is either outside the scope set by docs/prd.md §4 and §10,
or it is a design decision that has not been written down yet — in which case the RFC
comes first.
-->

## How it was verified

<!--
The commands you ran, and what they reported. Behaviour changes arrive test-first: the
failing test, then the code that makes it pass. Tests written against code that already
existed are characterization tests and are named as such.

If this touches the resolver's invariants — no dropped coordinate-less rows, no id
collisions, candidate generation linear under bounded bucket occupancy — say which
property tests cover them.
-->

## Effect on published figures

<!--
Whether any number in the README or docs/ changes. Every published figure traces to a
record in benchmarks/, so a changed figure means a new record, not an edited one. Write
"none" if nothing published moves.
-->

## Checklist

- [ ] Every commit is signed off (`git commit -s`), adding the `Signed-off-by:` trailer
      required by the Developer Certificate of Origin.
- [ ] Commit subjects follow `<type>(<scope>): <subject>` — present tense, lowercase,
      under 50 characters.
- [ ] New source files carry `# SPDX-License-Identifier: MIT` as their first comment line.
- [ ] Any new dependency is MIT, Apache or BSD licensed, and is declared in
      `pyproject.toml`.
- [ ] Any new fixture data is synthetic, with no real people, businesses, phone numbers,
      addresses, or coordinates that resolve to real premises.
- [ ] `CHANGELOG.md` records the change under `## [Unreleased]`, unless nothing
      user-visible moved.
