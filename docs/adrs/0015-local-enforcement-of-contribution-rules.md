# ADR-0015 — Contribution rules are enforced at the commit, not only in review

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

## Consequences

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

**The pre-commit framework.** It manages hook environments, which is valuable when hooks have
dependencies. These have none: they are a few lines of shell over `git` and `grep`. Adopting
a manifest format and a bootstrap step to run them would add moving parts without removing
any.
