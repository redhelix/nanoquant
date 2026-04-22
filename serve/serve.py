"""
serve.py — OpenAI-compatible vLLM server for Nemotron-Nano-9B-v2 with W4A16.

Two load modes (auto-detected):
  1. Pre-converted checkpoint (w4a16_manifest.json present): loads pre-quantized
     weights directly from the W4A16 shard — no re-quantization at startup.
  2. Base BF16 model: quantizes SSM layers on load (slower first start, ~2 min).

Required env:
  VLLM_USE_BYTECODE_HOOK=0   (set automatically below)

Usage:
  VLLM_USE_BYTECODE_HOOK=0 python serve/serve.py \\
      --model ./Nemotron-Nano-9B-v2-W4A16 \\
      --host 0.0.0.0 --port 8000 \\
      --max-model-len 32768 \\
      --gpu-memory-utilization 0.92 \\
      --enable-auto-tool-choice \\
      --tool-call-parser hermes

  # Or against the base model (quantizes at startup):
  VLLM_USE_BYTECODE_HOOK=0 python serve/serve.py \\
      --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
      ...same flags...
"""

import os
import sys

os.environ.setdefault("VLLM_USE_BYTECODE_HOOK", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


def _get_model_path() -> str:
    """Extract --model value from sys.argv without fully parsing args yet."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--model", "-m") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ""


def install_patch():
    """
    Patch NemotronHForCausalLM.load_weights to apply W4A16 after BF16 weights load.

    If the model path contains a W4A16 checkpoint shard, loads pre-quantized weights
    (fast path, no re-quantization). Otherwise quantizes on the fly (slow path).
    """
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from nanoquant.patch import patch_nemotron_h
    from nanoquant.checkpoint import load_w4a16_checkpoint, is_w4a16_checkpoint

    model_path = _get_model_path()
    _pre_quantized = is_w4a16_checkpoint(model_path) if model_path else False

    if _pre_quantized:
        print(f"[nanoquant] Pre-quantized checkpoint detected at {model_path!r}", flush=True)
    else:
        print("[nanoquant] No W4A16 shard found — will quantize SSM layers on load", flush=True)

    _dequant_mode = "--dequant-mode" in sys.argv

    _orig = NemotronHForCausalLM.load_weights

    def _patched(self, weights, *args, **kwargs):
        result = _orig(self, weights, *args, **kwargs)
        if _pre_quantized:
            n = load_w4a16_checkpoint(self, model_path)
            print(f"[nanoquant] Loaded {n} pre-quantized projection layers", flush=True)
        else:
            n = patch_nemotron_h(self, group_size=128, verbose=False)
            print(f"[nanoquant] W4A16 patch applied to {n} MambaMixer2 layers", flush=True)
        if _dequant_mode:
            from nanoquant.linear import W4A16Linear
            count = 0
            for module in self.modules():
                if isinstance(module, W4A16Linear):
                    module._use_dequant = True
                    count += 1
            print(f"[nanoquant] Dequant mode: {count} layers will use bf16 matmul", flush=True)
        return result

    NemotronHForCausalLM.load_weights = _patched


def main():
    install_patch()

    # Cap CUDA graph capture sizes to 4 (1,2,4,8) instead of 51 (1..512).
    # Prevents private pool OOM from per-row float32 scratch across all sizes.
    if "--compilation-config" not in sys.argv and "-cc" not in sys.argv:
        sys.argv += ["--compilation-config", '{"max_cudagraph_capture_size":8}']

    # Prevent concurrent prefill OOM: W4A16 allocates a float32 scratch buffer
    # per row per SSM layer during prefill. Concurrent prefills multiply this across
    # all requests simultaneously, exhausting the CUDA private pool and causing
    # an unrecoverable GPU error. Chunked prefill breaks long sequences into small
    # chunks so peak scratch memory stays bounded regardless of sequence length.
    # max-num-seqs caps the total requests in flight to limit worst-case concurrency.
    if "--enable-chunked-prefill" not in sys.argv:
        sys.argv += ["--enable-chunked-prefill"]
    if "--max-num-seqs" not in sys.argv:
        sys.argv += ["--max-num-seqs", "8"]
    if "--trust-remote-code" not in sys.argv:
        sys.argv += ["--trust-remote-code"]

    # Remove nanoquant-specific flags before vLLM's parser runs
    sys.argv = [a for a in sys.argv if a != "--dequant-mode"]

    from vllm.entrypoints.openai.api_server import run_server, make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import asyncio

    args = make_arg_parser(FlexibleArgumentParser()).parse_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
