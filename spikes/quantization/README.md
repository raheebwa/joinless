# int8 quantization feasibility spike

Scripted, re-runnable tooling for [RFC-0004](../../docs/rfcs/0004-quantization-spike-protocol.md)'s
eight-step protocol. Each numbered module below implements one RFC-0004 step; running
them in order produces one record in [`benchmarks/`](../../benchmarks/).

## Why this lives outside `joinless/`

The spike's output is a record, not library code — its job is to answer a feasibility
question before anything gets built, not to ship. Nothing under `joinless/` imports
anything here, so ADR-0014's invariant (a process that imports `joinless` and runs only
classical arms never initialises ONNX Runtime) holds regardless of what this tree does.

A reader who wants to re-run the spike finds it the same way they find anything else
scripted in this repository: by path, from the root, with no package installation of its
own beyond the optional dependency groups named below.

## What this measures

One model: `sentence-transformers/all-MiniLM-L6-v2` — a 22M-parameter stock sentence
encoder, the size ADR-0009 names when it argues a model this small may already be
resident in cache and bound by something other than weight bandwidth. Its exact
revision, checksum and licence are resolved from the model host at fetch time (never
hardcoded — see `model.py`) and recorded in `step1_model.json`.

Two graphs: the fp32 ONNX export and a `quantize_dynamic` int8 conversion of it, on
CPU via ONNX Runtime (ADR-0002).

## Setup

Two optional dependency groups, on top of the base `dev` profile:

```sh
pip install -e ".[dev,export,neural]"
```

- `export` — torch, transformers, `optimum[onnxruntime]`, onnx, huggingface_hub. Needed
  for steps 1–3 (fetching, exporting, quantizing). Kept out of `neural` deliberately:
  `neural` names the inference runtime whose cost the benchmark measures, and folding
  build-time tooling into it would change what that measurement means (ADR-0015).
- `neural` — ONNX Runtime alone. Needed for steps 4–7 (loading, comparing, measuring).

A writable directory for fetched artefacts and intermediate step output, supplied by
environment variable rather than assumed to exist:

```sh
export JOINLESS_MODEL_CACHE_DIR=/path/to/a/writable/directory
```

Nothing written under it is committed — it sits outside the repository tree. Every
fetch from the model host (the raw weights in step 1, the tokenizer wherever a later
step needs one) is pointed at `$JOINLESS_MODEL_CACHE_DIR/hf`, never at a tool's own
default cache location — so the entire fetched footprint lives under the one directory
supplied, and a clean value for that variable is what makes a run reproducible on a
machine with nothing pre-cached.

## Running the protocol

Each step is `python -m` over its module, in order. Every step is independently
re-runnable: it reads whatever the prior step wrote to
`$JOINLESS_MODEL_CACHE_DIR/stepN_*.json` and writes its own fragment there in turn.

```sh
python -m spikes.quantization.model          # step 1 — select and identify the model
python -m spikes.quantization.export_fp32    # step 2 — export the fp32 ONNX graph
python -m spikes.quantization.quantize_int8  # step 3 — quantize_dynamic to int8
python -m spikes.quantization.signatures     # step 4 — compare input/output signatures
python -m spikes.quantization.operators      # step 5 — diff operator types
python -m spikes.quantization.smoke          # step 6 — fp32/int8 divergence, smoke set
python -m spikes.quantization.measure        # step 7 — cost, fresh process per arm
python -m spikes.quantization.record         # step 8 — assemble the record
```

The last command prints the path it wrote under `benchmarks/`. That file is the spike's
output; everything under `$JOINLESS_MODEL_CACHE_DIR` is working state.

## Step 13 — not here

RFC-0004's go/no-go against its four abort clauses is written from this record's real
output, after a real run. It is not part of this tooling and the record leaves the field
(`go_no_go`) `null` until someone writes it by hand against the numbers actually
produced.

## Module map

| Step | Module | Issue |
|---|---|---|
| 1 | `model.py` | #7 |
| 2 | `export_fp32.py` | #7 |
| 3 | `quantize_int8.py` | #8 |
| 4 | `signatures.py` | #9 |
| 5 | `operators.py` | #10 |
| 6 | `smoke.py` | #11 |
| 7 | `measure.py` + `workers/measure_arm.py` | #12 |
| 8 | `record.py` | #6 |
| — | `cli_common.py` | shared plumbing, no dedicated issue |

Every module separates pure, tested logic (parameter construction, diffing, arithmetic,
record assembly) from the thin, untested edge that actually touches the network, a
subprocess, or a real model. See `tests/` for what is covered and each module's
docstring for what deliberately is not, and why.
