"""
bench_throughput.py — Measure decode throughput for Nemotron-Nano-9B-v2 W4A16.

Reproduces the benchmark numbers reported in the README / HF model card.
Runs on the live vLLM server (must be started separately) or directly via
the OpenAI client against any compatible endpoint.

Usage:
  # Start the server first:
  VLLM_USE_BYTECODE_HOOK=0 python serve/serve.py \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --host 0.0.0.0 --port 8000 --max-model-len 32768

  # Then run the benchmark:
  python bench/bench_throughput.py --base-url http://localhost:8000/v1 --model-name W4A16
  python bench/bench_throughput.py --base-url http://localhost:8000/v1 --model-name BF16

Output: tokens/sec p10/p50/p90 for batch=1 and batch=4, matching README table format.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

WARMUP_RUNS   = 3
MEASURE_RUNS  = 20
SHORT_PROMPT  = "What is 2 + 2?"            # decode isolation
LONG_PROMPT   = "Explain the Mamba SSM architecture in detail, covering state space models, "  \
                "the selective scan mechanism, hardware-aware parallelism, and how it differs " \
                "from attention-based transformers."
MAX_NEW_TOKENS = 128


def _chat(client, model_id: str, prompt: str, max_tokens: int):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
    )
    elapsed = time.perf_counter() - t0
    n_tokens = resp.usage.completion_tokens
    return n_tokens / elapsed if elapsed > 0 else 0.0


def _benchmark_single(client, model_id: str, prompt: str, label: str):
    print(f"\n  [{label}] warming up ({WARMUP_RUNS} runs)...", end="", flush=True)
    for _ in range(WARMUP_RUNS):
        _chat(client, model_id, prompt, MAX_NEW_TOKENS)

    print(f" measuring ({MEASURE_RUNS} runs)...", end="", flush=True)
    samples = [_chat(client, model_id, prompt, MAX_NEW_TOKENS) for _ in range(MEASURE_RUNS)]
    samples.sort()
    p10 = samples[int(len(samples) * 0.10)]
    p50 = statistics.median(samples)
    p90 = samples[int(len(samples) * 0.90)]
    print(f" done")
    return p10, p50, p90


def _benchmark_concurrent(client, model_id: str, prompt: str, batch: int, label: str):
    """Fire `batch` requests simultaneously, report per-request t/s."""
    print(f"\n  [{label} batch={batch}] warming up...", end="", flush=True)
    with ThreadPoolExecutor(max_workers=batch) as ex:
        for _ in range(WARMUP_RUNS):
            futs = [ex.submit(_chat, client, model_id, prompt, MAX_NEW_TOKENS) for _ in range(batch)]
            [f.result() for f in futs]

    print(f" measuring ({MEASURE_RUNS} rounds)...", end="", flush=True)
    all_samples = []
    with ThreadPoolExecutor(max_workers=batch) as ex:
        for _ in range(MEASURE_RUNS):
            futs = [ex.submit(_chat, client, model_id, prompt, MAX_NEW_TOKENS) for _ in range(batch)]
            all_samples.extend(f.result() for f in futs)

    all_samples.sort()
    p10 = all_samples[int(len(all_samples) * 0.10)]
    p50 = statistics.median(all_samples)
    p90 = all_samples[int(len(all_samples) * 0.90)]
    print(f" done")
    return p10, p50, p90


def run(base_url: str, model_id: str, model_name: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(base_url=base_url, api_key="none")

    print(f"\n{'=' * 60}")
    print(f"  NanoQuant Throughput Benchmark — {model_name}")
    print(f"  endpoint: {base_url}  model: {model_id}")
    print(f"  warmup={WARMUP_RUNS}  measure={MEASURE_RUNS}  max_new_tokens={MAX_NEW_TOKENS}")
    print(f"{'=' * 60}")

    results = {}

    for label, prompt in [("short", SHORT_PROMPT), ("long", LONG_PROMPT)]:
        p10, p50, p90 = _benchmark_single(client, model_id, prompt, f"batch=1 {label}")
        results[f"batch1_{label}"] = (p10, p50, p90)
        print(f"    batch=1 {label:<6}  p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}  t/s")

    for label, prompt in [("short", SHORT_PROMPT), ("long", LONG_PROMPT)]:
        p10, p50, p90 = _benchmark_concurrent(client, model_id, prompt, 4, f"batch=4 {label}")
        results[f"batch4_{label}"] = (p10, p50, p90)
        print(f"    batch=4 {label:<6}  p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}  t/s (per req)")

    print(f"\n{'=' * 60}")
    print(f"  README TABLE FORMAT ({model_name})")
    print(f"{'=' * 60}")
    b1_short = results["batch1_short"]
    b4_short = results["batch4_short"]
    print(f"  | Batch=1 decode (p50) | {b1_short[1]:.1f} t/s |")
    print(f"  | Batch=4 decode (p50) | {b4_short[1]:.1f} t/s |")
    print(f"{'=' * 60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="NanoQuant throughput benchmark")
    parser.add_argument("--base-url",   default="http://localhost:8000/v1")
    parser.add_argument("--model-id",   default=None,
                        help="Model ID as registered in vLLM (defaults to first available)")
    parser.add_argument("--model-name", default="W4A16",
                        help="Label for output (e.g. 'W4A16' or 'BF16')")
    args = parser.parse_args()

    model_id = args.model_id
    if model_id is None:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key="none")
        models = client.models.list()
        model_id = models.data[0].id
        print(f"Auto-detected model: {model_id}")

    run(args.base_url, model_id, args.model_name)


if __name__ == "__main__":
    main()
