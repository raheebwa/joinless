# ADR-0006 — Arm64 client devices as the target platform

**Status:** Accepted · **Date:** 2026-08-12

## Context

The measurement question — *when does a neural matcher earn its cost* — only has an
interesting answer where resources are constrained. On a large server the model is
affordable and nobody thinks about it. The trade becomes real on a client device: limited
memory, latency the user perceives directly, and often a battery.

Arm64 is where client computing has consolidated: phones, tablets, Apple Silicon laptops,
Windows on Arm, and Arm single-board computers.

There is a second, independent reason to stay on the client. Record linkage means holding
records about real entities, and sending those to a hosted matching service is frequently
the least acceptable step in the pipeline — legally, contractually, or ethically. A
resolver that never needs the network removes that step entirely.

## Decision

**Target Arm64 client devices. All inference runs locally. No network calls at match
time.**

Reference platform for published numbers: **Apple Silicon (M2 Max)**. The code is plain
Python plus ONNX Runtime, so other Arm64 clients — Windows on Arm, Arm64 Linux laptops,
Arm SBCs with sufficient memory — are expected to work. That expectation follows from the
dependency set and has not been measured; it is stated as expected compatibility, and a
platform earns an unqualified claim only when a run record exists for it.

## Consequences

- The constraint gives the measurement its meaning. Memory ceilings and perceptible
  latency are the whole point, not incidental limits.
- Privacy becomes a structural property rather than a policy promise: there is no code
  path that transmits a record.
- Published numbers are specific to the reference hardware. Every run records its
  hardware, OS and runtime versions (PRD MR-6), and results are always reported against
  named hardware.
- Server-scale throughput questions are out of scope. Batch behaviour on a many-core
  machine is a different project.
- The mainstream on-device inference tutorials for Arm target Android and LLM workloads.
  This project is neither, so it follows framework guidance without a template. That
  deviation is stated in the README rather than hidden.

## Alternatives rejected

**Arm64 cloud instances (Graviton / Cobalt / Axion / Ampere).** The quantization technique
is identical and the tooling is arguably better supported there. Rejected because it
removes the constraint that makes the question worth asking, forfeits the privacy
argument entirely, and makes the benchmark unreproducible for a reader without a cloud
account.

**x86 for comparison.** A cross-architecture arm would be interesting but doubles the
measurement matrix and answers a question nobody asked. Candidate for a later RFC.
