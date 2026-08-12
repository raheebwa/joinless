# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Semantic Versioning rule 4 permits a `0.y.z` release to break anything without notice.
This project does not take that latitude: before `1.0.0` as after it, a breaking change
bumps the MINOR version and is listed under `Changed` or `Removed`. "0.x, caveat emptor"
warns nobody.

## [Unreleased]

### Added

- Package manifest (`pyproject.toml`) with a PEP 639 MIT licence expression.
- `dev` and `neural` optional install profiles.
- `joinless` console entry point.
- The importable `joinless` package.
- `export` optional install profile (torch, transformers, `optimum[onnxruntime]`,
  onnx, huggingface_hub) for the scripted model-export tooling under `spikes/`.
- `rapidfuzz` as a runtime dependency, carrying the character-aware classical arm.
- Full line and branch coverage as an enforced floor rather than a reported number.
- `joinless.records` — `Record`, and a dual hashing scheme giving every row a stable
  identifier that does not collapse same-named rows carrying no coordinates.
- `joinless.scoring` — the `Scorer` protocol, the `ThresholdMatcher` adapter, and the
  `overlap` and `fuzzy` classical arms, selectable by name.
- `joinless.corpus` — a deterministic synthetic corpus over the eight perturbation
  families, split into three disjoint roles.
- `joinless.resolver` — `resolve`, merging two record sets that share no key.
- `joinless resolve`, `joinless compare` and `joinless doctor` on the command line.
  `doctor` reports architecture, execution provider, installed profile and offline status,
  so the Arm64, CPU-only and no-network properties are checkable rather than asserted.
