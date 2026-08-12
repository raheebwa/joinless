# Security Policy

## Supported versions

`joinless` follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html), and
versioning starts at `0.1.0`. No version has been published yet; `0.1.0` will be the first.

| Version | Security fixes |
|---|---|
| Latest published release | Yes |
| Any earlier release | No — upgrade to the latest |
| Default branch, between releases | No — it is not a supported artifact |

There is no long-term-support branch and no backporting. A security fix ships in a new
version, is listed under `Security` in [`CHANGELOG.md`](CHANGELOG.md), and the version it
supersedes is not patched.

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting:**
[open a draft advisory](https://github.com/raheebwa/joinless/security/advisories/new).
The same form is reachable from the repository's **Security and quality** tab under
**Report a vulnerability**. Either route opens a private draft advisory visible only to
you and the maintainer.

**Do not open a public issue, pull request, or discussion for a suspected vulnerability.**
A public report is a public disclosure, and it starts the clock for everyone using the
package before there is anything to upgrade to.

Reports are received privately through GitHub rather than by email. A published address
attracts more automated noise than reports and gives no way to attach a draft advisory to
the fix.

A useful report includes:

- what the defect is, and the version or commit it is present in;
- the input or sequence that triggers it — the smallest one you can find;
- the architecture, OS and Python version you observed it on, plus the runtime versions if
  an embedding arm is involved;
- what an attacker gains, in your assessment.

## What to expect

| Stage | Window |
|---|---|
| Acknowledgement that the report was received | 5 working days |
| Initial assessment — accepted, needs more information, or out of scope | 10 working days |
| Fix released, or a stated plan with dates if the work is larger | 90 days from acknowledgement |

The project is maintained by one person, and these windows are what one person can hold
to without a rota. A shorter window that is routinely missed is worse than a longer one
that is met, because a reporter deciding when to disclose needs a date that means
something. If a window is going to be missed, you will be told in the advisory thread
rather than left waiting.

When a fix ships, the advisory is published through GitHub with a CVE requested where the
defect warrants one. Reporters are credited by the name they ask to be credited by, or not
at all if they prefer.

## Scope

`joinless` is a local library. It makes no network calls at match time and sends no
telemetry; the one network operation is fetching model weights at setup. That shape
determines what a vulnerability here can be.

**In scope:**

- Code execution, path traversal, or unintended file writes triggered by input records,
  fixture files, labelled-pair files, or configuration.
- Catastrophic backtracking or unbounded resource consumption reachable from ordinary
  input to name normalisation, matching, or parsing.
- Unsafe deserialisation anywhere in the load path, including model artifacts.
- Fetching model weights over an unauthenticated or unverified channel, or accepting an
  artifact whose integrity is not checked.
- Leaking record contents outside the process — to disk in an unexpected location, to a
  log, or over a network connection the package should never open. The no-network-at-match-time
  property is a security property; a way to break it is a vulnerability.
- Anything that lets a dependency of the package reach the network at match time.

**Out of scope:**

- Vulnerabilities in third-party dependencies themselves. Report those upstream; tell us
  as well if `joinless` is affected and we will pin or bump.
- Anything requiring an attacker who already controls the machine or the Python
  environment. The package trusts the process it runs in.
- Resource exhaustion from input deliberately sized to exhaust it. Candidate generation
  degrades toward quadratic as bucket occupancy grows — documented behaviour
  ([`docs/prd.md`](docs/prd.md) FR-2), not a defect.
- Accuracy or benchmark-correctness defects. Those are ordinary bugs and belong in the
  issue tracker, where they can be discussed in public.
- Vulnerabilities in the pretrained models themselves, which are third-party artifacts
  fetched at setup and never redistributed by this project. Report those to their
  publisher.
