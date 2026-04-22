"""
Quick accuracy eval for W4A16 against a running vLLM server.

Runs MATH-500 — the fastest make-or-break math benchmark. Hits the OpenAI
endpoint directly (no lm-eval framework), so results are usable in ~10 min
depending on concurrency. For the full NeMo-Skills suite use
`eval/run_benchmarks.py` instead.

Usage:
  python eval/quick_eval.py \\
      --base-url http://localhost:8000/v1 \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --workers 4
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SYSTEM_PROMPT = "/think"


def chat(base_url, model, problem, temperature=0.6, top_p=0.95, max_tokens=16000):
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_boxed(text):
    """Extract the last \\boxed{...} answer, handling nested braces."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    results = []
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())
    if results:
        return results[-1]
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def answers_equal(predicted: str, ground_truth: str) -> bool:
    """Compare answers symbolically using math-verify, falling back to string norm."""
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${predicted}$"), parse(f"${ground_truth}$")))
    except Exception:
        pass
    def norm(s):
        s = s.strip().lower()
        s = re.sub(r"[,\s\\{}()]", "", s)
        s = s.replace("left", "").replace("right", "").replace("text", "")
        try:
            return str(float(s)) if "." in s else s
        except ValueError:
            return s
    return norm(predicted) == norm(ground_truth)


def load_math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [{"problem": row["problem"], "answer": row["answer"]} for row in ds]
    print(f"Loaded {len(problems)} MATH-500 problems")
    return problems


def run_eval(base_url, model, problems, workers=4):
    correct = 0
    total = len(problems)
    results = []

    def evaluate_one(i, item):
        try:
            response = chat(base_url, model, item["problem"])
            predicted = extract_boxed(response)
            is_correct = answers_equal(predicted, item["answer"])
            return i, is_correct, predicted, item["answer"], None
        except Exception as e:
            return i, False, "", item["answer"], str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_one, i, item): i for i, item in enumerate(problems)}
        for done_idx, future in enumerate(as_completed(futures), 1):
            i, is_correct, predicted, gt, err = future.result()
            correct += is_correct
            results.append({"idx": i, "correct": is_correct, "predicted": predicted, "gt": gt, "error": err})
            if done_idx % 10 == 0 or done_idx == total:
                print(f"  {done_idx}/{total} — running accuracy: {correct/done_idx*100:.1f}%")

    return correct / total, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url",   default="http://localhost:8000/v1")
    parser.add_argument("--model",      required=True,
                        help="Model path or ID as registered in vLLM")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--output-dir", default="./eval_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}\nRunning MATH-500\n{'='*50}")
    problems = load_math500()
    t0 = time.time()
    accuracy, results = run_eval(args.base_url, args.model, problems, workers=args.workers)
    elapsed = time.time() - t0

    out_file = output_dir / "math500_results.json"
    with open(out_file, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    correct = sum(r["correct"] for r in results)
    print(f"\nMATH-500 accuracy: {accuracy*100:.1f}% ({correct}/{len(problems)}) in {elapsed/60:.1f} min")
    print(f"Results written to {out_file}")


if __name__ == "__main__":
    main()
