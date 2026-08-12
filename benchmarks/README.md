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
- Per-arm results, per perturbation family

**Every number published in the README traces to a record here.** A figure that cannot be
traced to a run in this directory does not belong in the README.

Records are never edited after the fact. A run that produced a surprising or unflattering
result stays exactly as recorded; a later run is a new record, not an overwrite.
