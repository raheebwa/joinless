# Contributing to joinless

`joinless` is a library and a measurement. Both halves have to hold: the resolver has to
be correct, and every number published about it has to trace to a run someone else can
reproduce. Contributions are welcome on those terms.

Read [`docs/prd.md`](docs/prd.md) first — §4 (non-goals) and §10 (out of scope for v1) are
load-bearing, and they are the fastest way to tell whether an idea belongs here.

---

## Working on the code

From a clone of the repository, at its root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Use a Python version satisfying `requires-python` in `pyproject.toml`.

That is the base profile — the resolver, the `Matcher` protocol, the `overlap` and `fuzzy`
arms, the benchmark harness and the CLI — plus the tooling used to develop them. The
inference runtime is deliberately not in it. Working on an embedding arm needs the
`neural` extra as well:

```sh
pip install -e ".[dev,neural]"
```

The split is not packaging taste. A base install has to produce a tree in which ONNX
Runtime is not importable, because that is what turns "classical-only execution never
initialises the runtime" into something a test can assert instead of something the
documentation claims — see
[ADR-0014](docs/adrs/0014-optional-neural-install-profile.md). The corollary for anyone
touching the package: no module reachable from `import joinless` may import the runtime at
module level.

Model weights are fetched at setup and are never committed.

## Running the tests

One command, from the repository root:

```sh
python -m pytest
```

That is the command CI runs, on each interpreter version the project supports, and on both
install profiles — the base one, where ONNX Runtime is not importable, and the `neural`
one. If the documented command and the workflow ever diverge, the documented one is correct
and the workflow is the bug.

## Style and types

Three more commands, and CI runs exactly these:

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy .
```

Formatting is checked rather than rewritten in CI, so run `python -m ruff format .` before
you commit. Type checking is done against the `neural` extra installed, because a module
whose imports are absent is reported as an error rather than checked.

## Test policy

**Test-driven development is the policy of this project, and the commit sequence is the
evidence for it.**

1. Write the failing test first. Run it. Confirm it fails, and that it fails for the
   reason you expect — a test that passes before the behaviour exists is testing nothing.
2. Write the minimum code that makes it pass.
3. Refactor with the test green.
4. Commit. One behaviour per commit.

Implementation and its tests are never written in the same pass. Tests written against
code that already exists are **characterization tests**, and are named as such, so that a
later reader can tell which tests specified behaviour and which merely recorded it.

The resolver's correctness properties are invariants, not examples, and get property
tests:

- no record carrying no coordinates is ever dropped from the output;
- identifiers are stable and do not collide across sources, with behaviour for exact
  duplicates defined explicitly;
- candidate generation stays linear in record count under bounded bucket occupancy.

A commit that adds behaviour without a test that exercises it is not complete, and the
history is expected to show that.

## Numbers

Anything numeric added to the README, an ADR, or any other document must trace to a run
record in [`benchmarks/`](benchmarks/), and that record must name the hardware, OS, Python
version and runtime versions that produced it. Estimated, extrapolated, or
plausible-sounding figures are not acceptable in any file. Where a value is not yet
measured, write `TBD`.

Results that make the neural arms look bad are published exactly as prominently as results
that make them look good. A negative result with numbers is a result — see
[`docs/adrs/0010-claim-scope.md`](docs/adrs/0010-claim-scope.md) and
[`docs/adrs/0011-evaluation-protocol-integrity.md`](docs/adrs/0011-evaluation-protocol-integrity.md).

## Fixtures

Fixtures are synthetic and invented ([ADR-0004](docs/adrs/0004-synthetic-fixtures-only.md)).
No real business names, no real people, no real addresses, and no coordinates that resolve
to a real premises. Phone numbers, where a fixture needs one, take the form
`+000 000 000 000` — never a locale-plausible number, because many countries reserve no
range for fictional use and a "made up" number can route to a real person.

## Commits

Conventional Commits:

```
<type>(<scope>): <subject>
```

Types: `feat` `fix` `docs` `style` `refactor` `perf` `test` `chore`. Present tense,
lowercase, subject under 50 characters. The body explains why, not what — the diff already
says what.

**Sign off every commit:**

```sh
git commit -s
```

That appends a `Signed-off-by:` line, which certifies your contribution under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). There is **no
CLA**; the DCO is the whole of it, and an unsigned commit cannot be merged.

`Signed-off-by:` is the only trailer this project uses. Commit messages carry no other
trailers, so that provenance in the history means exactly one thing.

## Changelog

A change a user would notice is recorded in [`CHANGELOG.md`](CHANGELOG.md) under
`## [Unreleased]`, in the pull request that makes it. The format is
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/): ISO dates, and only the six
section types `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` and `Security`.

The file is written by hand and never generated from `git log`. A commit log records how
the work was done; the changelog records what changed for someone depending on the
package, and the two are not the same document.

Versioning is [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html) starting at `0.1.0`. The
latitude rule 4 grants `0.y.z` releases is not used here: before 1.0 as after it, a
breaking change bumps MINOR and is listed under `Changed` or `Removed`. A released version
is immutable — a bad release is superseded by a higher one and marked `[YANKED]`, never
re-cut.

## Pull requests

**Every pull request names the ADR or RFC it implements or amends.** Put it in the
description: *implements RFC-0001*, *amends ADR-0007*.

If the honest answer is "none", the change is one of two things: outside the scope the PRD
fixes, or a design decision nobody has written down yet. In the second case, open the RFC
first — [`docs/rfcs/`](docs/rfcs/) exists so that designs are argued before they are
implemented, and so that a maintainer five years from now can find the reasoning instead
of guessing at it.

Keep a pull request to one behaviour. A branch that changes the resolver, adds a matcher
and edits the benchmark protocol cannot be evaluated as any of the three.

## How changes land, stated plainly

This project has one maintainer. That fact is not hidden behind process language, and the
substitutes it forces are written down here rather than left to inference.

A change lands when it carries **a linked decision record and a green CI run**. Changes
submitted from outside are reviewed by the maintainer before merge. Changes authored by
the maintainer are not reviewed by a second person, and **this project does not claim peer
review it did not have.**

The same directness applies to the practices commonly expected of an open-source project.
Adopting a standard that cannot be satisfied is worse than declining it, because a
half-met checklist reads as a met one:

| Practice | Status here |
|---|---|
| OpenSSF Scorecard `Code-Review` | **Not adopted.** One maintainer. The substitute is stated plainly: every change lands with a linked ADR/RFC and a green CI run. This project does not claim peer review it did not have. |
| OpenSSF Scorecard `Contributors` (3+ organisations) | **Not adopted.** Not achievable by decision. |
| `Fuzzing` / OSS-Fuzz | **Not adopted.** The surface is a string similarity function over local input. Revisit if the package ever parses an untrusted binary format. |
| `Signed-Releases` + SBOM | **Deferred, not declined** — mandatory at `0.1.0`. There is nothing to sign yet, and claiming the practice now would be false. |
| Contributor Licence Agreement | **Not used.** The DCO (`git commit -s`) instead. |
| OpenSSF Best Practices badge enrolment | **Not enrolled.** The criteria are adopted on merit; a badge is not a goal. |

## Licensing

`joinless` is MIT ([`LICENSE`](LICENSE)). By signing off a commit you agree that your
contribution is offered under that licence.

- Every source file begins with `# SPDX-License-Identifier: MIT` as its first comment line.
  One line, machine-readable, and it survives a single file being copied out of the tree.
- Every dependency's licence must be MIT/Apache/BSD-compatible. Check before adding it,
  not after.
- Third-party algorithms are credited in the README and in the ADR that adopts them; a
  third-party snippet carries its own `SPDX-FileCopyrightText` at the point of use.

## Reporting problems

- **Bugs and behaviour questions** — open an issue. For anything involving a measured
  number, include the architecture, OS, Python version and the run record ID. Without
  those four, a reported number cannot be attributed to a machine, which makes it
  unusable.
- **Suspected vulnerabilities** — do not open an issue. Follow
  [`SECURITY.md`](SECURITY.md).
- **Conduct** — [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) applies to every space this
  project uses.
