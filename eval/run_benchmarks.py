"""
run_benchmarks.py — Evaluate Nemotron-Nano-9B-v2 W4A16 on the same benchmark
suite used by NVIDIA's NVFP4 model card.

Benchmarks:
  AIME 2025     (aime_2025)
  MATH-500      (math_500)
  GPQA Diamond  (gpqa_diamond)
  LiveCodeBench (livecodebench)
  BFCL v3       (bfcl_v3)        — requires separate BFCL runner, see below
  IFEval        (ifeval)
  RULER 128K    (ruler_128k)

All benchmarks except BFCL use lm-evaluation-harness (EleutherAI).
BFCL requires the gorilla-llm/gorilla CLI — instructions at bottom.

Requirements:
  pip install lm-eval[math,ifeval] vllm nanoquant

Usage:
  # Against the pre-converted W4A16 checkpoint (recommended):
  python eval/run_benchmarks.py \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --output-dir ./eval_results \\
      --benchmarks aime math gpqa ifeval ruler

  # Against the base model with runtime patch:
  python eval/run_benchmarks.py \\
      --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
      --output-dir ./eval_results_bf16 \\
      --benchmarks aime math gpqa ifeval ruler

  # All benchmarks (takes ~6h on RTX 3090):
  python eval/run_benchmarks.py \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --output-dir ./eval_results \\
      --benchmarks all

BFCL v3 (separate runner):
  git clone https://github.com/ShishirPatil/gorilla
  cd gorilla/berkeley-function-call-leaderboard
  pip install -r requirements.txt
  python openfunctions_evaluation.py \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --test-category all
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# lm-eval task names
TASK_MAP = {
    "aime":    "aime_2025",
    "math":    "math_500",
    "gpqa":    "gpqa_diamond_zeroshot_cot",
    "lcb":     "livecodebench",
    "ifeval":  "ifeval",
    "ruler":   "ruler_4k,ruler_8k,ruler_16k,ruler_32k,ruler_128k",
}

REASONING_SYSTEM_PROMPT = (
    "detailed thinking on"
)


def _build_vllm_args(model_path: str, max_model_len: int, gpu_util: float) -> dict:
    """vLLM kwargs for lm-eval --model vllm."""
    from nanoquant.checkpoint import is_w4a16_checkpoint
    args = {
        "pretrained": model_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_util,
        "trust_remote_code": True,
        "compilation_config": '{"max_cudagraph_capture_size":8}',
        "enable_auto_tool_choice": False,
    }
    # Apply nanoquant patch via env flag that serve.py honours
    os.environ["NANOQUANT_MODEL_PATH"] = model_path
    return args


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    output_dir: Path,
    max_model_len: int = 32768,
    gpu_util: float = 0.92,
    num_fewshot: int = 0,
    batch_size: int = 1,
):
    """Run lm-evaluation-harness tasks and write results JSON."""
    try:
        import lm_eval
    except ImportError:
        log.error("lm-eval not installed. Run: pip install lm-eval[math,ifeval]")
        sys.exit(1)

    # Patch vLLM with nanoquant before lm-eval spins up the model
    _install_nanoquant_patch(model_path)

    task_string = ",".join(tasks)
    log.info(f"Running tasks: {task_string}")

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={model_path},max_model_len={max_model_len},"
                   f"gpu_memory_utilization={gpu_util},trust_remote_code=True,"
                   f'add_bos_token=True',
        tasks=task_string,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=True,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"results_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {out_file}")

    _print_summary(results)
    return results


def _install_nanoquant_patch(model_path: str):
    """Apply W4A16 patch to vLLM's NemotronH model loader."""
    try:
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    except ImportError:
        log.warning("vLLM not installed or NemotronH not available — patch skipped")
        return

    from nanoquant.patch import patch_nemotron_h
    from nanoquant.checkpoint import load_w4a16_checkpoint, is_w4a16_checkpoint

    pre_quantized = is_w4a16_checkpoint(model_path)
    _orig = NemotronHForCausalLM.load_weights

    def _patched(self, weights, *args, **kwargs):
        result = _orig(self, weights, *args, **kwargs)
        if pre_quantized:
            n = load_w4a16_checkpoint(self, model_path)
            log.info(f"[nanoquant] Loaded {n} pre-quantized projection layers")
        else:
            n = patch_nemotron_h(self, group_size=128, verbose=False)
            log.info(f"[nanoquant] W4A16 patch applied to {n} MambaMixer2 layers")
        return result

    NemotronHForCausalLM.load_weights = _patched
    os.environ["VLLM_USE_BYTECODE_HOOK"] = "0"


def _print_summary(results: dict):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for task, metrics in results.get("results", {}).items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {task:<40} {metric:<20} {value:.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NanoQuant benchmark evaluation")
    parser.add_argument("--model",         required=True,
                        help="Path to W4A16 checkpoint or base model ID")
    parser.add_argument("--output-dir",    default="./eval_results")
    parser.add_argument("--benchmarks",    default="all",
                        help=f"Comma-separated from: {','.join(TASK_MAP.keys())}, or 'all'")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-util",      type=float, default=0.92)
    parser.add_argument("--batch-size",    type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmarks == "all":
        keys = [k for k in TASK_MAP if k != "lcb"]   # LCB needs separate setup
        log.info("Running all benchmarks (skipping livecodebench — run separately)")
    else:
        keys = [k.strip() for k in args.benchmarks.split(",")]
        invalid = [k for k in keys if k not in TASK_MAP]
        if invalid:
            log.error(f"Unknown benchmarks: {invalid}. Valid: {list(TASK_MAP.keys())}")
            sys.exit(1)

    tasks = []
    for k in keys:
        tasks.extend(TASK_MAP[k].split(","))

    run_lm_eval(
        model_path=args.model,
        tasks=tasks,
        output_dir=output_dir,
        max_model_len=args.max_model_len,
        gpu_util=args.gpu_util,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
