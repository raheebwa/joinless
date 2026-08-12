# ADR-0010 — What this project claims, and what it does not

**Status:** Accepted · **Date:** 2026-08-12

## Context

A claim's scope should match the evidence behind it. The evidence this project will
produce is four matchers, measured on one synthetic benchmark, on one machine.

Two claims that sound natural are not supported by that evidence:

- **"Where the classical/neural crossover lies."** We measure two classical matchers and
  one embedding model. "Classical" spans phonetic encodings, Monge-Elkan, learned
  distance metrics and more; "neural" spans fine-tuned cross-encoders and models orders of
  magnitude larger. Measuring four points does not locate a frontier across two families.
- **"Embeddings improve record linkage."** Already established.
  [LinkTransformer](https://arxiv.org/abs/2309.00789) evaluates four linkage tasks —
  company aliases among them — and reports off-the-shelf sentence-transformer models
  outperforming Levenshtein edit-distance matching, typically by a wide margin. Asserting
  it as a finding would be false, not merely redundant.

A third is a terminology error rather than an overreach. **"Privacy-preserving record
linkage"** is a term of art for cryptographic protocols that let two parties link records
without either revealing its data to the other. Running locally is not that. Borrowing the
phrase would describe a guarantee this project does not provide.

## Decision

**Claims are bounded to the evidence: these matchers, this benchmark, this hardware.**

Supported:

- how these four matchers compare on this disclosed synthetic benchmark, reported per
  perturbation family
- what the embedding arms cost on the named reference machine — cold start, warm scoring,
  batched preparation, resident memory, artifact size
- whether the preparation hoist and quantization each change those costs, and by how much
- the conditions under which each arm is the reasonable choice **for this benchmark**

Not claimed, in any document, commit message or description:

| Not claimed | Why |
|---|---|
| "the classical/neural crossover" | four measured points do not locate a frontier across two families |
| "embeddings beat string matching" as a finding | established prior art; cite it, do not re-assert it |
| "privacy-preserving record linkage" | means cryptographic multi-party linkage; we do not do that |
| "first" / "only" benchmark of anything | unverifiable, and false as far as the record shows |
| state-of-the-art accuracy | no fine-tuning, no leaderboard comparison, explicitly a non-goal |
| pair F1 measures resolution quality | pair accuracy establishes neither correct clustering nor one-to-one assignment; what is scored is the name-similarity component, after candidate generation |
| results transfer to real corpora unchanged | fixtures are synthetic by construction (ADR-0004) |
| any platform without a run record | expected compatibility is not measurement (ADR-0006) |

What replaces the crossover framing: *given the model is more accurate, what does it cost
on an Arm64 client, and does a regime exist where the cheap matcher is still the right
call?*

## Consequences

- The headline finding is narrower and checkable. Every sentence describing a result can
  be traced to a run record or removed.
- Where a broader claim would be interesting, it becomes explicit future work with the
  measurement that would be needed — not a hedge.
- Prior art must be cited wherever the accuracy result is relied upon, since the project
  builds on it rather than establishing it.
- New documentation has a concrete list to check against, which is more useful than an
  instruction to be careful.

## Alternatives rejected

**Claim the crossover and caveat it.** Caveats do not travel — the headline gets quoted and
the qualification does not. If the claim needs a caveat to be true, the claim is wrong.

**Say nothing about scope and let readers infer it.** Readers infer the widest reading a
sentence permits. Silence about limits is a claim about their absence.
