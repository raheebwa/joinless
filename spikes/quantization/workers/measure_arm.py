# SPDX-License-Identifier: MIT
"""Fresh-process worker for RFC-0004 step 7: measure one arm (issue #12).

Run via ``python -m spikes.quantization.workers.measure_arm``, once per arm, by
:func:`spikes.quantization.measure.main` — never imported directly. That is what makes
"fresh child process per arm" true: a new interpreter, with nothing yet imported, per
invocation.

Untested by design: everything this module computes (phase arithmetic, RSS
normalization, latency summarization) is pure logic that lives in, and is tested in,
:mod:`spikes.quantization.measure`. What is left here is the part that cannot be tested
without a real model and a real subprocess — driving the tokenizer and the ONNX Runtime
session and writing down what happened, in order, once.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Sequence
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    checkpoints: dict[str, float] = {"ready": time.perf_counter()}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["fp32", "int8"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hf-cache-dir", required=True)
    parser.add_argument(
        "--batch-sizes", required=True, help="comma-separated, e.g. 1,8,32"
    )
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--intra-op-threads", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import onnxruntime
    from transformers import AutoTokenizer

    from spikes.quantization.model import MODEL_ID
    from spikes.quantization.smoke import cosine_similarity, mean_pool

    checkpoints["after_import"] = time.perf_counter()

    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = args.intra_op_threads
    session_options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        args.model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    checkpoints["after_session"] = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.hf_cache_dir)
    checkpoints["after_tokenizer"] = time.perf_counter()

    # Two invented names from the smoke set (spikes/quantization/smoke.py). Scoring a
    # pair needs both of them: RFC-0002 defines warm scoring as the cost of one
    # comparison, and a comparison is two embeddings and the similarity between them.
    # Timing one embedding and recording it as a pair understates the per-comparison
    # cost by about half, under a label claiming otherwise.
    probe_pair = ("Sunrise Agro Traders", "Sunrise Agro Trading Company")
    probe_text = probe_pair[0]

    def embed_batch(texts: list[str]) -> list[Sequence[float]]:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        feed = {
            name: encoded[name]
            for name in encoded
            if name in {"input_ids", "attention_mask", "token_type_ids"}
        }
        outputs = session.run(None, feed)
        token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"]
        return [
            mean_pool(token_embeddings[i], attention_mask[i]) for i in range(len(texts))
        ]

    def score_pair() -> float:
        left, right = embed_batch(list(probe_pair))
        return cosine_similarity(left, right)

    score_pair()  # first inference: outside every timed loop below
    checkpoints["after_first_inference"] = time.perf_counter()

    single_pair_latencies = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        score_pair()
        single_pair_latencies.append(time.perf_counter() - start)

    batched_latencies: dict[str, list[float]] = {}
    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        texts = [probe_text] * batch_size
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            embed_batch(texts)
            samples.append(time.perf_counter() - start)
        batched_latencies[str(batch_size)] = samples

    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    Path(args.out).write_text(
        json.dumps(
            {
                "arm": args.arm,
                "checkpoints": checkpoints,
                "single_pair_latencies": single_pair_latencies,
                "batched_latencies": batched_latencies,
                "peak_rss_raw": peak_rss_raw,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
