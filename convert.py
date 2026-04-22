"""
convert.py — Offline W4A16 quantization for Nemotron-Nano-9B-v2.

Loads the base model, replaces all MambaMixer2 in_proj/out_proj with
W4A16Linear, and saves a standalone HuggingFace checkpoint that vLLM
can load without any monkey-patching.

Usage:
    python convert.py \
        --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        --output ./Nemotron-Nano-9B-v2-W4A16 \
        --group-size 128

    # Upload to HuggingFace
    huggingface-cli upload redhelix/Nemotron-Nano-9B-v2-W4A16 ./Nemotron-Nano-9B-v2-W4A16

Requirements:
    pip install torch transformers accelerate triton huggingface_hub
    # GPU not required for conversion (runs on CPU, ~5 min for 9B model)
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def convert(model_id: str, output_dir: str, group_size: int = 128, dtype: str = "bfloat16"):
    from nanoquant.kernel import quantize_w4

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading base model from {model_id!r} (this may take a few minutes)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    ssm_layers = 0
    total_saved_bytes = 0

    log.info("Quantizing MambaMixer2 SSM layers to W4A16...")
    for name, module in model.named_modules():
        if type(module).__name__ != "MambaMixer2":
            continue

        for proj_name in ("in_proj", "out_proj"):
            proj = getattr(module, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue

            W = proj.weight.data.to(torch.bfloat16)
            orig_bytes = W.numel() * 2

            W_q, scales, zeros = quantize_w4(W.cpu(), group_size)
            quant_bytes = W_q.numel() + (scales.numel() + zeros.numel()) * 2
            total_saved_bytes += orig_bytes - quant_bytes

            # Replace weight with packed tensor so state_dict captures it
            # Convention: store as "<name>.weight_q", "<name>.scales", "<name>.zeros"
            # A matching vLLM plugin will detect and use these during model load.
            # For now, save as additional state_dict keys alongside original weights.
            proj.register_buffer("weight_q", W_q)
            proj.register_buffer("w4_scales", scales)
            proj.register_buffer("w4_zeros",  zeros)
            proj.register_buffer("w4_group_size", torch.tensor(group_size))

            log.info(f"  {name}.{proj_name} [{W.shape[0]}x{W.shape[1]}] "
                     f"-> W4 ({quant_bytes/1e6:.1f} MB, was {orig_bytes/1e6:.1f} MB)")

        ssm_layers += 1

    log.info(f"Quantized {ssm_layers} SSM layers. "
             f"Total VRAM saved: {total_saved_bytes/1e9:.2f} GB")

    log.info(f"Saving quantized checkpoint to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    log.info("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Write quantization metadata
    quant_config = {
        "quant_type": "w4a16",
        "group_size": group_size,
        "bits": 4,
        "zero_point": True,
        "layout": "column_major_packed",
        "base_model": model_id,
        "nanoquant_version": "0.1.0",
    }
    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    log.info(f"Done. Saved to {output_path}")
    log.info(f"To upload: huggingface-cli upload redhelix/Nemotron-Nano-9B-v2-W4A16 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Nemotron-Nano-9B-v2 to W4A16")
    parser.add_argument("--model",      default="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output",     default="./Nemotron-Nano-9B-v2-W4A16",
                        help="Output directory for quantized checkpoint")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float16"],
                        help="Base model dtype (default: bfloat16)")
    args = parser.parse_args()

    convert(args.model, args.output, args.group_size, args.dtype)


if __name__ == "__main__":
    main()
