# NanoQuant

**W4A16 quantization for NVIDIA Nemotron-Nano-9B-v2 on consumer GPUs.**

A custom Triton GEMV kernel that quantizes the SSM (Mamba-2) layers of Nemotron-Nano-9B-v2 to 4-bit weights while keeping activations in bfloat16. Developed and benchmarked on an RTX 3090.

## Results (RTX 3090, 24 GB)

| Metric | BF16 baseline | W4A16 (NanoQuant) | Gain |
|---|---|---|---|
| Batch=1 decode (p50) | 44.8 t/s | **54.1 t/s** | **1.21×** |
| Batch=4 decode (p50) | 44.8 t/s | **99.8 t/s** | **2.23×** |
| SSM layer VRAM | 16.6 GB | 8.5 GB | **−8.1 GB** |

The VRAM reduction makes the model fit comfortably on a single 24 GB GPU with a 32k context window.

## Why SSM layers?

Nemotron-Nano-9B-v2 is a **Nemotron-H** architecture: 56 layers with 27 MambaMixer2 (SSM), 25 MLP, and 4 attention layers. During decode, the SSM layers dominate at ~75% wall-clock time because their `in_proj` / `out_proj` projections are large, memory-bound GEMVs:

- `in_proj`:  [22656 × 4480] — 50.7 MB per call in BF16
- `out_proj`: [4480 × 10240] — 45.9 MB per call in BF16

W4A16 reduces these to ~4× fewer bytes read per forward pass, directly translating to throughput gains on memory-bandwidth-limited hardware.

## Architecture

```
quantize_w4()          CPU, once at load time — asymmetric int4, column-major packing
_gemv_w4a16_kernel     Triton, K-parallel 2D grid, atomicAdd into fp32 accumulator
w4a16_linear           torch.library.custom_op — opaque to Dynamo/aot_compile_fullgraph
W4A16Linear            nn.Module drop-in for vLLM's ColumnParallelLinear
patch_nemotron_h()     Replaces in_proj/out_proj in every MambaMixer2 layer
```

## Quick Start

### Option A: Runtime patch (no model conversion needed)

```bash
pip install nanoquant vllm
VLLM_USE_BYTECODE_HOOK=0 python serve/serve.py \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### Option B: Convert and save (use the pre-quantized HF checkpoint)

```bash
# Convert once, save a quantized checkpoint
python convert.py \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --output ./Nemotron-Nano-9B-v2-W4A16

# Or download the pre-converted checkpoint from HuggingFace
# huggingface-cli download redhelix/Nemotron-Nano-9B-v2-W4A16
```

### Python API

```python
import torch
from nanoquant import quantize_w4, W4A16Linear, patch_nemotron_h

# Quantize a weight tensor
W = torch.randn(22656, 4480, dtype=torch.bfloat16)
W_q, scales, zeros = quantize_w4(W, group_size=128)

# Convert a linear layer
import torch.nn as nn
linear = nn.Linear(4480, 22656, bias=False)
q_linear = W4A16Linear.from_linear(linear)

# Patch a loaded vLLM model
# patch_nemotron_h(vllm_model)
```

## How It Works

### Quantization (asymmetric int4, group_size=128)

Each group of 128 input channels gets its own scale and zero point:

```
scale = (w_max - w_min) / 15
zero  = w_min / scale
w_int = round(w / scale - zero).clamp(0, 15)
```

Weights are packed 2-per-byte in **column-major order** `[K//2, N]` to coalesce adjacent-thread memory reads during the Triton kernel.

### Triton Kernel (K-parallel 2D grid)

```
grid = (ceil(N/BLOCK_N), n_groups)
```

Each CTA handles exactly one quantization group (64 inner loop iterations for group_size=128). Groups reduce via `atomicAdd` into a float32 accumulator, then cast to bf16. This exposes `n_groups × ceil(N/BLOCK_N)` parallelism — e.g. 6195 CTAs for the `in_proj` shape.

### vLLM Integration (custom_op escape hatch)

vLLM's `aot_compile_fullgraph` compiles one Inductor artifact from a `profile_run` at batch=8192 and reuses it for all CUDA graph sizes. Any `if batch==1` fast path gets compiled away.

`torch.library.custom_op` is opaque to Dynamo — the body runs **eagerly** during each CUDA graph capture, so each captured batch size records its own Triton kernel launches. No graph breaks, no `@torch.compiler.disable` needed.

The `max_cudagraph_capture_size: 8` compilation config caps captures to 4 sizes (1,2,4,8) instead of 51, preventing private pool OOM from the per-row float32 scratch allocations.

## Comparison: W4A16 vs NVIDIA NVFP4

| | **NanoQuant W4A16** | **NVIDIA NVFP4** |
|---|---|---|
| Hardware target | RTX 3090 (Ampere SM86) | A10G / H100 / Jetson AGX Thor |
| Format | Asymmetric int4, group_size=128 | NVFP4 (4-bit float) |
| Calibration | Min-max (no calibration data) | QAD (Quantization-Aware Distillation) |
| Kernel | Custom Triton GEMV | cuDNN / CUTLASS GEMM |
| vLLM support | Runtime patch via custom_op | Native (ModelOpt) |
| Batch=1 t/s (RTX 3090) | **54.1** | N/A (A10G target) |
| VRAM (9B model) | **8.5 GB** (SSM layers) | ~5 GB full model |

NVIDIA's NVFP4 uses QAD to maintain accuracy closer to BF16. Our W4A16 uses simple min-max quantization — run the benchmarks below to compare accuracy on your tasks.

## Benchmarks

| Benchmark | BF16 | W4A16 (NanoQuant) | NVFP4 (NVIDIA) |
|---|---|---|---|
| AIME 2025 | — | — | 71.5% |
| MATH-500 | — | — | 97.2% |
| GPQA Diamond | — | — | 62.7% |
| LiveCodeBench | — | — | 67.8% |
| BFCL v3 | — | — | 65.9% |
| IFEval (Strict) | — | — | 89.3% |
| RULER 128K | — | — | 75.0% |

*W4A16 benchmark results pending — contributions welcome.*

## Files

```
nanoquant/
  kernel.py      Triton kernel + quantize_w4() + w4a16_linear custom_op
  linear.py      W4A16Linear nn.Module
  patch.py       patch_nemotron_h() for vLLM integration
convert.py       Offline quantization → HF checkpoint
serve/serve.py   vLLM API server with runtime patch
```

## Requirements

- Python 3.10+
- PyTorch 2.3+
- Triton 3.0+
- CUDA-capable GPU (Ampere or newer recommended)
- vLLM (for serving)

```bash
pip install -r requirements.txt
```

## License

Apache 2.0. Base model ([nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)) is subject to NVIDIA's license.
