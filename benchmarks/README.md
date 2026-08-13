# Benchmark records

One record per run. Records are evidence, not commentary: each is self-contained so a
reader can tell exactly what produced the numbers.

Every record carries:

- Hardware — CPU, core count, memory
- OS and version
- Python version, ONNX Runtime version, `rapidfuzz` version
- Model identity, revision and checksum, where applicable
- Which operators were quantized, for int8 arms
- Evaluation set identity and case mixture
- Thread count, warm-up count, repetitions, power mode
- The exact command that produced it
- Per-arm accuracy, per perturbation family, both pooled across every seed and per seed,
  each stating which question it answers, plus the seed-to-seed variation behind the
  pooled figure — never one without the other (RFC-0002 "Per-seed accuracy reporting")
- The int8 arm's per-family accuracy divergence from the fp32 arm, where both ran
  (computed from each arm's pooled figure)
- Each arm's hoisted and naive preparation cost, measured over the run's own shared
  candidate set
- The candidate-bucket occupancy distribution that set produced, and the grid cell size
  it was measured under
- The classical/neural preparation hoist speed-up, partitioned by arm family

**Every number published in the README traces to a record here.** A figure that cannot be
traced to a run in this directory does not belong in the README.

Records are never edited after the fact. A run that produced a surprising or unflattering
result stays exactly as recorded; a later run is a new record, not an overwrite.

The quantized-operator list is read from the int8 graph itself at run time, not carried
over from the spike record — and if a run's graph does not match the operator census that
graph is recorded to have, the run writes no record at all rather than one row quietly
marked unavailable while the rest reads as if nothing were wrong.
