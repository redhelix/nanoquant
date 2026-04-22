"""
convert.py — Offline W4A16 quantization for Nemotron-Nano-9B-v2.

Loads the base BF16 model, quantizes all MambaMixer2 SSM projection layers
to W4A16, and saves a standalone HuggingFace checkpoint. The output directory
contains:

  config.json / tokenizer / BF16 weights  — unchanged base model files
  w4a16_weights.safetensors               — packed int4 weights + scales + zeros
  w4a16_manifest.json                     — maps layer paths to tensor keys
  quantization_config.json                — metadata (nanoquant_version, group_size, ...)

Serving loads the pre-quantized shard directly (no re-quantization at startup).

Usage:
    python convert.py \\
        --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
        --output ./Nemotron-Nano-9B-v2-W4A16 \\
        --group-size 128

    # Verify the checkpoint loads cleanly
    python convert.py --verify ./Nemotron-Nano-9B-v2-W4A16

    # Upload to HuggingFace
    huggingface-cli upload redhelix/Nemotron-Nano-9B-v2-W4A16 ./Nemotron-Nano-9B-v2-W4A16

Requirements:
    pip install nanoquant transformers accelerate safetensors huggingface_hub
    GPU not required — conversion runs on CPU (~8 min, ~20 GB RAM for 9B model)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def convert(model_id: str, output_dir: str, group_size: int = 128):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoquant.patch import patch_nemotron_h
    from nanoquant.checkpoint import save_w4a16_checkpoint

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading base model {model_id!r} on CPU (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()

    log.info(f"Quantizing SSM layers to W4A16 (group_size={group_size})...")
    n_patched = patch_nemotron_h(model, group_size=group_size, verbose=True)
    log.info(f"Patched {n_patched} MambaMixer2 layers")

    log.info(f"Saving base model weights to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    log.info("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    log.info("Saving W4A16 weight shard...")
    stats = save_w4a16_checkpoint(model, output_path, group_size=group_size)
    saved_gb = stats["saved_bytes"] / 1e9
    log.info(
        f"Done. {stats['projections']} projections across {stats['ssm_layers']} SSM layers. "
        f"VRAM savings: {saved_gb:.2f} GB"
    )
    log.info(f"Output: {output_path}")
    log.info(f"Upload: huggingface-cli upload redhelix/Nemotron-Nano-9B-v2-W4A16 {output_path}")


def verify(checkpoint_dir: str):
    """Smoke-test that the checkpoint loads cleanly and produces finite output."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoquant.checkpoint import load_w4a16_checkpoint, is_w4a16_checkpoint

    path = Path(checkpoint_dir)
    if not is_w4a16_checkpoint(path):
        log.error(f"{checkpoint_dir} does not appear to be a W4A16 checkpoint")
        sys.exit(1)

    log.info(f"Loading base model from {checkpoint_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()

    log.info("Loading W4A16 weight shard...")
    n = load_w4a16_checkpoint(model, path)
    log.info(f"Loaded {n} projection layers from checkpoint")

    log.info("Running forward pass smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    assert logits.isfinite().all(), "Non-finite logits — checkpoint is corrupt"
    log.info(f"Forward pass OK. Logits shape: {list(logits.shape)}, finite: True")
    log.info("Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="NanoQuant: Convert Nemotron-Nano-9B-v2 to W4A16")
    sub = parser.add_subparsers(dest="cmd")

    conv = sub.add_parser("convert", help="Quantize and save checkpoint")
    conv.add_argument("--model",      default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    conv.add_argument("--output",     default="./Nemotron-Nano-9B-v2-W4A16")
    conv.add_argument("--group-size", type=int, default=128)

    ver = sub.add_parser("verify", help="Verify a saved checkpoint loads and runs")
    ver.add_argument("checkpoint_dir")

    # Legacy: plain `python convert.py --model X --output Y` still works
    parser.add_argument("--model",      default=None)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--group-size", type=int, default=128, dest="group_size")
    parser.add_argument("--verify",     metavar="DIR", default=None,
                        help="Verify an existing checkpoint instead of converting")

    args = parser.parse_args()

    if args.cmd == "convert":
        convert(args.model, args.output, args.group_size)
    elif args.cmd == "verify":
        verify(args.checkpoint_dir)
    elif args.verify:
        verify(args.verify)
    elif args.model and args.output:
        convert(args.model, args.output, args.group_size)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
