# ADR-0015 — The project's rules are enforced locally, not only in review

**Status:** Accepted · **Date:** 2026-08-12

## Context

`CONTRIBUTING.md` states rules a change must satisfy: Conventional Commit subjects under
fifty characters, a `Signed-off-by:` trailer and no other, an SPDX header on every source
file, actions pinned to a commit SHA, a `permissions:` block in every workflow, ISO dates in
the changelog, no committed model artefacts, and no phone number outside the documented
placeholder form.

Stating a rule is not enforcing it. Two of these were broken while the rules were in place
and unenforced: a commit reached a branch without its sign-off, and a file carrying a
locale-plausible telephone number was written into the object store. Neither was caught by
reading; both were caught later, when correcting them was more expensive than refusing them
would have been.

CI checks most of these, but CI sees a branch only once its history exists. A malformed
subject or a missing trailer cannot be corrected after a push without rewriting a branch
other people may already have — cheap before the push, disruptive after it. The moment of
the commit is the only point at which the fix costs nothing.

## Decision

The rules that a machine can check are enforced by hooks in `.githooks/`, enabled with
`git config core.hooksPath .githooks`.

- `commit-msg` — subject shape, subject length, sign-off present, and `Signed-off-by:` as
  the only trailer.
- `pre-commit` — refuses the default branch and a branch not named
  `<type>/<short-description>`, then applies the repository rules to the **staged** content.
- `pre-push` — refuses a direct push to the default branch.

The checks are functions of the paths they are given, in `.githooks/checks.sh`, so they run
against fixtures in `tests/hooks/` without needing a repository. A check nobody tested
enforces nothing.

`pre-commit` judges staged content rather than the working tree, because a partially staged
file is committed as staged and that is the version the rules must hold for.

Each hook runs `.git/hooks/<name>.local` when one exists and is executable, so a contributor
can add a personal check without modifying a tracked file.

## Decision — the toolchain

The reference interpreter is pinned in `mise.toml` and the dependency set is locked.

```toml
[tools]
python = "3.14.5"

[env]
_.python.venv = { path = ".venv", create = true }
```

The reasoning is the same as above, applied to a different rule. PRD MR-6 requires the
Python and runtime versions to be recorded with every run, and PRD §9 names runtime version
differences as a way the reported numbers move. Recording a version after the fact
describes what happened; pinning it is what lets a second machine match it. An interpreter
that follows whatever the system package manager last installed is not a reference.

Dependencies are declared in two layers, because they answer two different questions:

- `pyproject.toml` declares **lower bounds only**. This is a library, and a library that
  pins exact versions dictates resolution to everyone who installs it.
- The lock file records the **exact** versions the reference environment resolved to. That
  is what makes a run record reproducible rather than merely descriptive.

The pin is not the supported range. `requires-python` admits 3.11 and later and CI tests
every version in that range; that is what establishes portability. The pin names the one
interpreter the reference machine uses, which is a separate question, and narrowing the CI
matrix to match it would be a mistake.

Export-time tooling is not the inference runtime. Anything needed to produce a model
artefact belongs in its own group rather than in `neural`, because `neural` names the
runtime whose cost the benchmark measures and folding a build-time dependency into it would
make that measurement mean something else.

## Consequences

- A pinned interpreter that the toolchain manager cannot install is not a pin. The version
  chosen here resolves to a standard build; a freethreaded build would be a different
  interpreter with different performance characteristics, and selecting one by accident
  would change every measured number without changing any documented decision.
- Enforcement is per clone, because `git config` is per clone. A contributor who has not run
  the line is unprotected, so **CI remains the authority** and the hooks are the earlier,
  cheaper of two gates rather than a replacement for the later one.
- The checks are shell, and shell is easy to write badly. They are therefore tested, and the
  hook runs its own tests before permitting a commit.
- The rule set is duplicated between the hooks and CI. This is deliberate: they run at
  different times against different inputs, and a rule that holds in only one of those
  places is not enforced. When a rule changes, both change.
- `--no-verify` bypasses all of it. That is a property of git, and the answer to a hook that
  is wrong is to fix the hook rather than to route around it.

## Alternatives rejected

**Enforcement in CI alone.** CI cannot reject a commit message, only report on one that
already exists. The correction it prompts is a branch rewrite, which is precisely what
catching the fault one second earlier avoids.

**commitlint with husky.** Both are Node programs. Adopting them puts a `package.json` and a
second language runtime into a Python library whose documented setup needs neither, and
requires a Node step in CI. In exchange they cover the commit message alone — leaving the
SPDX, classifier, action-pinning, workflow-permissions, changelog, artefact and placeholder
rules to be implemented separately regardless.

**commitizen.** Python-native and a genuine option for the message rules, but it lives in the
`dev` extra and therefore cannot run until the package is installed. A fresh clone, which is
where a first commit is most likely to be malformed, is exactly the state in which it is
absent. It also covers only the message, so it reduces rather than removes the shell.

**A version range for the reference interpreter.** A range is the right answer for
`requires-python`, which describes what the package supports. It is the wrong answer for the
machine that produces published figures, where the whole point is that two runs used the
same thing.

**Pinning exact versions in `pyproject.toml` instead of a lock file.** It would make the
reference set reproducible and simultaneously impose it on every downstream installation,
which is not this project's business.

**The pre-commit framework.** It manages hook environments, which is valuable when hooks have
dependencies. These have none: they are a few lines of shell over `git` and `grep`. Adopting
a manifest format and a bootstrap step to run them would add moving parts without removing
any.
