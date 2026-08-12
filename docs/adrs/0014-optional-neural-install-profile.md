# ADR-0014 — The neural runtime is an optional install profile, not a base dependency

**Status:** Accepted · **Date:** 2026-08-12 · **Refines:** ADR-0002, ADR-0003, ADR-0008

## Context

Two of the four arms need nothing beyond the standard library and `rapidfuzz` (ADR-0008). The
other two need an inference runtime — which arrives with transitive dependencies of its own,
NumPy among them — and a model artefact. The default packaging choice, declaring everything
the project uses as a dependency of the project, makes the inference runtime arrive with the
package whether or not a reader ever runs a neural arm.

That default costs more than disk space.

**It makes "the classical arm is cheap" unmeasurable.** Cheapness is not a background fact
about the classical arms; it is one of the quantities this benchmark reports. Cold start is
measured in a new process, from load through first inference, because it is paid once per
invocation and dominates short runs. If importing the package initialises ONNX Runtime
regardless of the configured arm, then the cold start recorded for `overlap` includes a
runtime `overlap` never calls. The figure would describe the installation, not the matcher —
and it would move whenever the runtime's own import cost moved, for reasons having nothing
to do with name matching. RFC-0002 already isolates peak resident memory in a fresh child
process per arm for the same reason: a measurement that shares a process with another arm
ends up reporting the other arm. Import cost needs the same boundary, and a boundary that
only exists at run time cannot be established by a package that imports the runtime on load.

**It removes the deployment path ADR-0003 commits to.** ADR-0003 records the runtime-free
classical path as a supported configuration for users who cannot or will not add an inference
runtime — a real constraint, not a hypothetical. ADR-0008 fixes its precise form: the
zero-dependency property belongs to `overlap` alone, and `fuzzy` adds `rapidfuzz` and nothing
heavier. A configuration that cannot be installed without the inference runtime is neither of
those, whatever the documentation says.

The distinction only becomes checkable if it is expressed in the packaging metadata, because
that is what determines what a user actually ends up with.

## Decision

**Two install profiles, and one invariant that makes the boundary testable.**

| Profile | Install | Contains |
|---|---|---|
| base | `pip install joinless` | resolver, `Matcher` protocol, `overlap` and `fuzzy` arms, benchmark harness, CLI |
| neural | `pip install joinless[neural]` | the above, plus ONNX Runtime and its transitive dependencies, and the artefact preparation path |

The mechanism is a packaging **extra** — a `[project.optional-dependencies]` entry, which
is published in distribution metadata and can therefore be requested by anyone installing
from an index. A PEP 735 dependency group is the wrong instrument: build backends must not
emit group data as package metadata, so a group cannot be selected by a user installing a
released version. Groups describe local development; this is a user-facing profile.

**The invariant, stated so that it is a test rather than an intention:**

> A process that imports `joinless` and runs only classical arms never initialises ONNX
> Runtime. After any classical-only execution, `onnxruntime` is absent from `sys.modules`.

Two structural rules follow, and both are consequences of the invariant rather than
independent preferences:

1. **No module reachable from package import may import the runtime at module level.** The
   neural imports belong inside the embedding matcher's own module, behind the point where
   an embedding arm is actually constructed. A stray top-level import is not a style
   problem; it silently reattaches the cost the boundary exists to detach.
2. **Requesting a neural arm on a base install is a first-class outcome, not a traceback.**
   It is an arm that cannot initialise, so ADR-0013 governs it: the arm is recorded as
   `unavailable` with a reason, and the reason names the install that would make it
   available.

## Consequences

- The cold-start and import-cost figures for the classical arms become attributable to the
  matchers, which is the only form in which they are worth publishing.
- Both profiles have to be exercised, and each catches a different failure. Where the neural
  extra is installed, the `sys.modules` assertion is what detects an import that has
  regressed from lazy to eager. Where it is not, the test is that importing the package and
  running the classical arms succeeds at all — which is what shows the base profile is
  genuinely installable rather than only declared.
- The runtime-free classical path stops being a claim in a document and becomes an install
  command, which is the only version of it a user can act on.
- The extra is a published compatibility surface. Renaming it later breaks install commands
  in other people's scripts and documentation, so it gets one name and keeps it.
- A base install cannot execute the neural arms at all, so the boundary tests carry real
  weight: they are the mechanism that stops the two profiles from quietly becoming one.
- Two supported configurations is more surface than one. Accepted for the same reason
  ADR-0003 accepts two matcher families: the comparison is the product, and a comparison
  whose cheap side cannot be installed cheaply has already lost half its meaning.

## Alternatives rejected

**One profile; the runtime is a base dependency.** Simplest to document and every arm works
everywhere. It also makes the classical arms' cost a property of the install rather than the
matcher, and forces an inference runtime onto users whose reason for choosing the classical
arm was avoiding one. The measurement and the deployment story fail together, which is a
strong signal that the packaging choice was load-bearing all along.

**Lazy imports alone, with the runtime still declared as a required dependency.** Satisfies
the in-process half of the invariant and none of the rest. The runtime is still installed,
still downloaded, and still present in every environment — so the runtime-free path
remains unavailable and the only thing gained is that the cost is not paid twice.

**A separate `joinless-neural` distribution.** Clean separation, and two version numbers
that must be held in step. The arms are compared at one revision by construction; a version
skew between the classical and neural distributions would be exactly the kind of
uncontrolled difference ADR-0002 constraint 3 rules out. An extra keeps one version for all
four arms.

**Detect the runtime at run time and enable the neural arms if it imports.** Makes the
available arm set a property of whatever else is installed in the environment, so the same
command produces a two-arm or a four-arm result depending on an unrelated package. That is
the silently-dropped-arm defect ADR-0013 forbids, arriving through the front door.
