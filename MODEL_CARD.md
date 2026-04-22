---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
tags:
  - mamba
  - ssm
  - nemotron
  - quantized
  - w4a16
  - triton
  - vllm
  - text-generation
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# Nemotron-Nano-9B-v2 W4A16

**W4A16-quantized Nemotron-Nano-9B-v2 — optimized for consumer GPUs (RTX 3090, 24 GB).**

This is a community quantization of [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) using a custom Triton GEMV kernel targeting the SSM (Mamba-2) layers that dominate decode throughput. Weights are 4-bit integers; activations remain bfloat16.

Quantization code and serving instructions: [github.com/redhelix/nanoquant](https://github.com/redhelix/nanoquant)

---

## Performance

### Throughput (RTX 3090, 24 GB GDDR6X)

| Metric | BF16 baseline | **W4A16 (this model)** | Speedup |
|---|---|---|---|
| Single decode token/s (p50) | 44.8 t/s | **51.0 t/s** | **1.14×** |

*Single decode: measured via OpenAI API against vLLM server (20 runs, 128 output tokens).
Batched-decode throughput under real concurrency has not yet been measured end-to-end.*

### VRAM

| | BF16 | **W4A16** | Saved |
|---|---|---|---|
| SSM layer weights | 16.6 GB | **8.5 GB** | **−8.1 GB** |

The VRAM reduction allows a 32k context window on a single 24 GB GPU. BF16 requires either a larger GPU or a reduced context.

---

## Accuracy

BF16 numbers are reproduced using NVIDIA's [NeMo-Skills eval framework](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/tutorials/posts/nemotron-nano-v2-evals.md). All results use Reasoning-On mode (thinking enabled) except RULER 128K which uses Reasoning-Off.

| Benchmark | BF16 (NeMo-Skills) | **W4A16 (this model)** | NVFP4 (NVIDIA) |
|---|---|---|---|
| AIME 2025 | 72.1% | — | 71.5% |
| MATH-500 | 97.0% | — | 97.2% |
| GPQA Diamond | 66.1% | — | 62.7% |
| LiveCodeBench | 67.4% | — | 67.8% |
| BFCL v3 | 67.0% | — | 65.9% |
| IFEval (Strict) | 90.1% | — | 89.3% |
| RULER 128K | 79.1% | — | 75.0% |

*BF16 results: pass@1 avg-of-8 (GPQA: majority@8), reproduced via NeMo-Skills. W4A16 results pending — contributions welcome via the [nanoquant repo](https://github.com/redhelix/nanoquant).*

NVIDIA's NVFP4 uses Quantization-Aware Distillation (QAD) via ModelOpt, which typically recovers most accuracy lost to quantization. This W4A16 uses simple min-max quantization without calibration data — accuracy results may vary slightly.

---

## Model Architecture

Nemotron-Nano-9B-v2 is a **Nemotron-H** hybrid SSM-transformer:

| Layer type | Count | Notes |
|---|---|---|
| MambaMixer2 (SSM) | 27 | **Quantized by NanoQuant** |
| MLP | 25 | BF16, unchanged |
| Attention | 4 | BF16, unchanged |
| **Total** | **56** | |

The SSM layers account for ~75% of decode wall-clock time due to large, memory-bound GEMVs:
- `in_proj`:  [22,656 × 4,480] — 50.7 MB per call in BF16
- `out_proj`: [4,480 × 10,240] — 45.9 MB per call in BF16

W4A16 reduces weight reads by ~4× per call, translating directly to throughput gains on memory-bandwidth-limited hardware.

---

## Quantization Method

**Format:** Asymmetric int4, group_size=128, column-major packed storage

**Per group (128 input channels):**
```
scale = (w_max - w_min) / 15
zero  = w_min / scale
w_q   = round(w / scale − zero).clamp(0, 15)   # stored as uint4
```

**Storage layout:** `[K//2, N]` column-major (2 weights packed per byte). Column-major coalesces adjacent-thread reads in the Triton kernel — the key layout decision over naive row-major packing.

**Kernel:** K-parallel 2D Triton GEMV. Grid `(ceil(N/BLOCK_N), n_groups)` — each CTA handles one quantization group (64 inner iterations). Groups reduce via `atomicAdd` into a float32 accumulator, then cast to bf16. BLOCK_N autotuned across {32, 64, 128, 256}.

**vLLM integration:** `torch.library.custom_op` makes the GEMV opaque to `aot_compile_fullgraph` — the body runs eagerly during each CUDA graph capture so per-batch-size kernel launches are correctly recorded without graph breaks.

---

## Comparison: W4A16 vs NVFP4

| | **NanoQuant W4A16** | **NVIDIA NVFP4** |
|---|---|---|
| Hardware target | RTX 3090 (Ampere, SM86) | A10G / H100 / Jetson AGX Thor |
| Quantization format | Asymmetric int4 (group_size=128) | NVFP4 (4-bit float) |
| Calibration | Min-max (no data needed) | QAD (Quantization-Aware Distillation) |
| Kernel | Custom Triton GEMV | cuDNN / CUTLASS GEMM |
| Layers quantized | SSM in_proj / out_proj (27 layers) | All layers except first/last 2 + attention |
| vLLM support | Runtime patch via custom_op | Native (ModelOpt) |
| VRAM (model only) | 8.5 GB (SSM) + ~8 GB (BF16 rest) | ~5 GB full model |

NVFP4 targets NVIDIA's professional fleet and uses hardware-native float4 GEMMs on H100/A10G. NanoQuant targets consumer Ampere GPUs (RTX 3080/3090/4090) where NVFP4 kernels are unavailable.

---

## Usage

### With vLLM (recommended)

```bash
pip install nanoquant vllm

VLLM_USE_BYTECODE_HOOK=0 python -m nanoquant.serve \
    --model redhelix/Nemotron-Nano-9B-v2-W4A16 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### Direct inference

```python
# Apply runtime patch to stock base model (no download of this checkpoint needed)
from nanoquant import patch_nemotron_h
from vllm import LLM, SamplingParams

llm = LLM(
    model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    trust_remote_code=True,
    max_model_len=32768,
    gpu_memory_utilization=0.92,
)
patch_nemotron_h(llm.llm_engine.model_executor.driver_worker.model_runner.model)

output = llm.generate("Explain the Mamba SSM architecture.", SamplingParams(max_tokens=512))
print(output[0].outputs[0].text)
```

### Tool calling

Tool calling works via `hermes` parser (Nemotron uses `<TOOLCALL>` format, closest to hermes). Use `hintonator-cascade-2-30b` or a frontier model for applications requiring reliable structured tool output.

---

## Important Notes

- `VLLM_USE_BYTECODE_HOOK=0` is **required** — the bytecode hook interferes with the W4A16 custom_op patching
- `--compilation-config '{"max_cudagraph_capture_size":8}'` is injected automatically by the serve script — limits CUDA graph captures to 4 sizes (1,2,4,8) instead of 51, preventing private pool OOM
- Tool call accuracy is approximate — hermes parser handles `tool_choice: auto` but Nemotron's `<TOOLCALL>[...]</TOOLCALL>` format differs from standard hermes
- Attention layers and MLP layers remain in BF16 — only the 27 SSM projection layers are quantized

---

## License

The quantization code (NanoQuant) is Apache 2.0.

The base model weights are subject to the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). This quantized checkpoint is a derivative work and inherits the same license terms. Commercial use is permitted subject to NVIDIA's terms.

---

## Citation

```bibtex
@misc{nanoquant2026,
  title  = {NanoQuant: W4A16 Triton GEMV for Nemotron-Nano-9B-v2 SSM Decode},
  author = {Amin Lalji},
  year   = {2026},
  url    = {https://github.com/redhelix/nanoquant},
}
```

Base model:
```bibtex
@misc{nvidia2025nemotron,
  title  = {Nemotron-Nano-9B: A Reasoning Model for Efficient Deployment},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2},
}
```
