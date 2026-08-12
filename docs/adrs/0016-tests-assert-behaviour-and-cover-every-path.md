# ADR-0016 — Tests assert behaviour, and cover every path

**Status:** Accepted · **Date:** 2026-08-12 · **Supersedes:** nothing · **Serves:** PRD G1, MR-6

## Context

`CONTRIBUTING.md` already fixes test-driven development as the policy and names three
resolver invariants as property-tested. Two things were missing from that: any statement of
what a test has to do to count, and any check that every path has one.

Both gaps were occupied rather than theoretical.

**The invariant test could not fail.** PRD FR-12a requires that classical-only execution
never initialises the inference runtime *as an invariant with a test rather than a
convention*. The test popped `onnxruntime` from `sys.modules` and then ran `import
joinless` — which by that point is served from `sys.modules` and never re-executes the
module body. Adding a module-level import of the runtime to the package left the test
passing. The most load-bearing test in the repository asserted against an import it had not
performed.

**The quantization spike shipped five defects that its tests could not see.** Four of the
five survived for one reason: the stand-in was easier to satisfy than the thing it stood
for. A `list` where production had a numpy array, so `if embeddings` was fine in the test
and raised in production. A `float` where production had `np.float32`, so the record
serialised in the test and failed to serialise for real. A mocked subprocess, so an
invented command-line flag was never rejected. A mocked API client, so a field the real
host omits came back populated.

Neither gap is the sort a review finds. A hollow test reads exactly like a real one — that
is what makes it hollow — and a path with no test at all is invisible in a diff that only
shows the lines someone wrote.

## Decision

### 1. A test asserts behaviour a caller can observe

Not the shape of the implementation. The check is whether the test would survive a rewrite
that kept the behaviour: if changing how the code works requires changing the test, the
test is describing the code rather than constraining it, and it can never report a
regression.

### 2. A stand-in must be at least as awkward as the thing it replaces

This is the rule the spike needed. Where a double is unavoidable, it models the real
thing's inconvenient behaviour, not its convenient behaviour — an array-like that raises on
`bool()` because numpy does, a subprocess double that rejects flags the real command
rejects, an API double that omits the fields the real host omits.

A double built to make the test pass tests the double.

### 3. Where a double would have to be that elaborate, use the real thing

A child interpreter, a temporary directory, a real file. The boundary test now spawns a
real interpreter, because module-level import behaviour is not observable any other way
from inside a process that has already imported the module.

### 4. Invariants get property tests; examples get example tests

An invariant is a claim over all inputs — no coordinate-less record dropped, no identifier
collision across sources, candidate generation linear under bounded occupancy. An example
test says the claim held for the inputs someone thought of, and the failure modes here are
the ones nobody thinks of. Hypothesis is the tool.

### 5. Every path is executed: 100% line and branch coverage, enforced

`fail_under = 100`, branch coverage on, checked in CI and by the pre-commit hook.

**Coverage is a completeness check, not a quality one.** It says no code path ships
unexecuted. It says nothing about whether the assertion around that path is worth anything
— 100% is reachable by calling every function and asserting nothing. That is precisely why
it is paired with rules 1 to 4 rather than standing alone, and why neither half substitutes
for the other.

**The no-cover pragma is not honoured.** Coverage's default exclude list is replaced rather
than extended. A pragma is the one-line way to make any number reach 100, and a floor that
can be waived inline is not a floor. Excluding a line takes an edit to the list in
`pyproject.toml` — visible in a diff, with a reason attached.

Three exclusions stand, each a line that *cannot* execute under test rather than one nobody
covered: the `__main__` guard, `if TYPE_CHECKING:`, and a bare ellipsis, which is a
Protocol member's body and a declaration rather than code.

### 6. Test names state the behaviour

`test_a_record_without_coordinates_is_never_dropped`, not `test_resolve_2`. A name that
states the claim makes a failing run readable without opening the file.

## Scope

The floor covers `joinless/` — everything the wheel ships. `spikes/` is excluded, and it is
the only exclusion of its kind: it holds one finished experiment whose output is already
recorded in `benchmarks/`, the build backend packages `joinless/` alone so it reaches no
installer, and it will not run again. Held to the floor it would generate tests for code
that can no longer regress.

## Consequences

- A new module lands with its paths executed or CI is red. There is no accumulating debt,
  because there is no state in which the number is 97% and someone means to get to it.
- The floor is cheapest to adopt now, at 28 lines of shipped code, and gets more expensive
  every week it is deferred.
- Some tests will exist mainly to execute a path. That is the honest cost of the floor, and
  rules 1 to 4 are what stop those tests from being the whole suite.
- Coverage measures execution, so it cannot detect an assertion that is too weak. If
  hollowness turns out to survive the rules above, mutation testing is the escalation —
  deliberately not adopted now, because it is slow and there is no evidence yet that the
  rules are insufficient.

## Alternatives considered

**Behaviour-driven syntax — feature files and step definitions.** Gherkin exists to give a
non-technical stakeholder a spec they can read and validate. This project has one
maintainer and no such reader, so the cost — a prose layer, a glue layer, and a regex
between the test and the assertion — buys nothing. It also fits the work badly: the
invariants here are claims over generated inputs, not user journeys, and *given 1,000
randomly generated record sets* fights the syntax. Rule 6 takes the one good idea in it, a
name that states the behaviour, at no cost.

**A coverage percentage as the whole bar.** Rejected as the *only* rule, adopted as one of
six. Alone it rewards tests that execute code and assert nothing, which is the failure this
ADR exists to prevent.

**A lower floor, say 90%.** The uncovered tenth is not randomly distributed — it is the
error paths and the branches nobody wanted to set up, which is where defects live. A
threshold below 100 also has no natural defence: every argument for 90 is an argument for
85 next quarter.

**Uploading coverage to a third-party service for a badge.** Rejected. It adds an external
dependency and a token to a project whose stated position is that nothing leaves the
machine, and it reports a number CI already enforces. The badge states the enforced floor
and links to the workflow that enforces it.

**Excluding hard-to-test paths by pragma.** This is the mechanism the decision above
specifically disables, and the reason is that it is invisible: a pragma looks like a
comment, while an entry in the exclude list is a change to project configuration.
