"""
serve.py — OpenAI-compatible vLLM server for Nemotron-Nano-9B-v2 with W4A16 patch.

Applies the post-load quantization patch before vLLM builds CUDA graphs,
then launches the standard OpenAI-compatible API server.

Usage:
    python serve/serve.py \
        --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.92 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes

Requirements:
    pip install vllm triton torch
    VLLM_USE_BYTECODE_HOOK=0 python serve/serve.py ...
    (VLLM_USE_BYTECODE_HOOK=0 is required for the W4A16 custom_op patching)
"""

import os
import sys

# Required for W4A16 custom_op patching — bytecode hook interferes with eager body
os.environ.setdefault("VLLM_USE_BYTECODE_HOOK", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def install_patch(group_size: int = 128):
    """Monkey-patch NemotronHForCausalLM.load_weights to apply W4A16 after loading."""
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from nanoquant.patch import patch_nemotron_h

    _orig = NemotronHForCausalLM.load_weights

    def _patched(self, weights, *args, **kwargs):
        result = _orig(self, weights, *args, **kwargs)
        n = patch_nemotron_h(self, group_size=group_size, verbose=False)
        print(f"[nanoquant] W4A16 patch applied to {n} MambaMixer2 layers", flush=True)
        return result

    NemotronHForCausalLM.load_weights = _patched


if __name__ == "__main__":
    install_patch(group_size=128)

    # Inject CUDA graph capture size limit to avoid private pool OOM from
    # per-row float32 scratch allocations across 51 batch sizes (1..512).
    # Caps captures to 4 sizes (1,2,4,8) instead.
    if "--compilation-config" not in sys.argv and "-cc" not in sys.argv:
        sys.argv += ["--compilation-config", '{"max_cudagraph_capture_size":8}']

    from vllm.entrypoints.openai.api_server import run_server, make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import asyncio

    args = make_arg_parser(FlexibleArgumentParser()).parse_args()
    asyncio.run(run_server(args))
