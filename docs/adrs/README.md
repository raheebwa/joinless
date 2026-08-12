# Architecture Decision Records

One file per decision. Format: Context · Decision · Consequences · Alternatives rejected.

A decision recorded here is binding until superseded by a later ADR that says so
explicitly. Superseded records are kept, not deleted — the reasoning is the point.

| # | Decision | Status |
|---|---|---|
| [0001](0001-greenfield-package-citing-prior-art.md) | Greenfield package; prior art cited, not forked | Accepted |
| [0002](0002-onnx-runtime-as-inference-runtime.md) | ONNX Runtime as the on-device inference runtime | Accepted |
| [0003](0003-classical-matcher-is-a-first-class-arm.md) | The classical matcher is a first-class arm, not a legacy path | Accepted — arm count superseded by [0008](0008-character-aware-classical-arm.md) |
| [0004](0004-synthetic-fixtures-only.md) | Synthetic fixtures only; no real-world corpus | Accepted |
| [0005](0005-mit-licence.md) | MIT licence | Accepted |
| [0006](0006-arm64-client-devices-as-target.md) | Arm64 client devices as the target platform | Accepted |
| [0007](0007-int8-dynamic-quantization-first.md) | int8 dynamic quantization as the first optimization pass | Accepted |
| [0008](0008-character-aware-classical-arm.md) | Add a character-aware classical arm | Accepted |
| [0009](0009-preparation-hoist-is-the-primary-optimization.md) | The preparation hoist is the primary optimization | Accepted |
| [0010](0010-claim-scope.md) | What this project claims, and what it does not | Accepted |
| [0011](0011-evaluation-protocol-integrity.md) | Evaluation protocol integrity | Accepted |
| [0012](0012-bring-your-own-labelled-pairs.md) | Bring-your-own labelled pairs is the only scope addition | Accepted |
| [0013](0013-fail-closed-failure-semantics.md) | Fail closed: a failure is recorded as a failure, never degraded into a number | Accepted |
| [0014](0014-optional-neural-install-profile.md) | The neural runtime is an optional install profile, not a base dependency | Accepted |
