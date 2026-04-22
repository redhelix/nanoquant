"""
convert.py — Offline W4A16 quantization for Nemotron-Nano-9B-v2.

Loads weights directly from the base model safetensors shards — no model
instantiation, no mamba-ssm dependency. Identifies SSM projection tensors by
name pattern, quantizes them to W4A16, and saves a standalone shard alongside
the original model files.

Output directory layout:
  <original model files>      — config.json, tokenizer, BF16 weight shards (symlinked)
  w4a16_weights.safetensors   — packed int4 weights + scales + zeros
  w4a16_manifest.json         — maps layer key → {W_q, scales, zeros, group_size, shape}
  quantization_config.json    — metadata

Usage:
    python convert.py convert \\
        --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
        --output ./Nemotron-Nano-9B-v2-W4A16

    python convert.py verify ./Nemotron-Nano-9B-v2-W4A16

Requirements:
    pip install torch triton safetensors huggingface_hub
    GPU not required — runs on CPU (~8 min, ~20 GB RAM)
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Tensor name suffixes that identify SSM projection weights in Nemotron-H
SSM_PROJ_PATTERNS = (".mixer.in_proj.weight", ".mixer.out_proj.weight")


def _resolve_model_dir(model_id: str) -> Path:
    """Return local path to model files, downloading from HF if needed."""
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    log.info(f"Downloading {model_id!r} from HuggingFace...")
    local = snapshot_download(model_id, ignore_patterns=["*.msgpack", "*.h5", "flax_*"])
    return Path(local)


def _iter_weight_files(model_dir: Path):
    """Yield (path, shard_tensors) for every safetensors shard in model_dir."""
    from safetensors import safe_open
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(model_dir.glob("model.safetensors"))
        shards = [s.name for s in shards]

    for shard_name in shards:
        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            yield shard_path, f


def convert(model_id: str, output_dir: str, group_size: int = 128):
    from safetensors.torch import save_file
    import torch
    from nanoquant.kernel import quantize_w4

    model_dir   = _resolve_model_dir(model_id)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Source model: {model_dir}")
    log.info(f"Output:       {output_path}")

    # Copy / symlink all non-weight files (config, tokenizer, etc.)
    for f in model_dir.iterdir():
        dest = output_path / f.name
        if dest.exists():
            continue
        if f.suffix in (".safetensors",) or f.name == "model.safetensors.index.json":
            # Symlink weight shards — we add our own shard on top
            os.symlink(f.resolve(), dest)
        else:
            shutil.copy2(f, dest)
    log.info("Copied config/tokenizer files")

    # Scan all shards, quantize SSM projection tensors
    q_tensors: dict = {}
    manifest:  dict = {}
    stats = {"ssm_projections": 0, "saved_bytes": 0}

    log.info(f"Scanning weight shards for SSM projection tensors (group_size={group_size})...")
    for shard_path, sf in _iter_weight_files(model_dir):
        keys = sf.keys()
        ssm_keys = [k for k in keys if any(k.endswith(pat) for pat in SSM_PROJ_PATTERNS)]
        if not ssm_keys:
            continue

        log.info(f"  {shard_path.name}: found {len(ssm_keys)} SSM projection(s)")
        for key in ssm_keys:
            W = sf.get_tensor(key)  # [out_features, in_features] bfloat16
            N, K = W.shape
            orig_bytes = N * K * 2

            W_q, scales, zeros = quantize_w4(W.float().cpu(), group_size)
            quant_bytes = W_q.numel() + (scales.numel() + zeros.numel()) * 2
            stats["saved_bytes"]     += orig_bytes - quant_bytes
            stats["ssm_projections"] += 1

            q_tensors[f"{key}.W_q"]    = W_q
            q_tensors[f"{key}.scales"] = scales
            q_tensors[f"{key}.zeros"]  = zeros
            manifest[key] = {
                "W_q":          f"{key}.W_q",
                "scales":       f"{key}.scales",
                "zeros":        f"{key}.zeros",
                "group_size":   group_size,
                "out_features": N,
                "in_features":  K,
            }
            log.info(f"    {key} [{N}×{K}] → W4A16 ({quant_bytes/1e6:.1f} MB, was {orig_bytes/1e6:.1f} MB)")

    if not q_tensors:
        log.error("No SSM projection tensors found — check model structure or name patterns")
        sys.exit(1)

    shard_out = output_path / "w4a16_weights.safetensors"
    save_file(q_tensors, str(shard_out))
    log.info(f"Saved W4A16 shard: {shard_out} ({len(q_tensors)} tensors)")

    with open(output_path / "w4a16_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    quant_config = {
        "quant_type":        "w4a16",
        "group_size":        group_size,
        "bits":              4,
        "zero_point":        True,
        "layout":            "column_major_packed",
        "ssm_proj_patterns": list(SSM_PROJ_PATTERNS),
        "shard_file":        "w4a16_weights.safetensors",
        "manifest_file":     "w4a16_manifest.json",
        "nanoquant_version": "0.1.0",
    }
    with open(output_path / "quantization_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    saved_gb = stats["saved_bytes"] / 1e9
    log.info(
        f"Done. {stats['ssm_projections']} projections quantized. "
        f"VRAM savings vs BF16: {saved_gb:.2f} GB"
    )
    log.info(f"Upload: hf upload redhelix/Nemotron-Nano-9B-v2-W4A16 {output_path}")


def verify(checkpoint_dir: str):
    """
    Verify the W4A16 shard loads cleanly and tensors are finite.
    Does NOT require vLLM or mamba-ssm — checks the shard file directly.
    """
    from safetensors.torch import load_file
    import torch

    path = Path(checkpoint_dir)
    quant_cfg = path / "quantization_config.json"
    manifest_f = path / "w4a16_manifest.json"
    shard_f    = path / "w4a16_weights.safetensors"

    for f in (quant_cfg, manifest_f, shard_f):
        if not f.exists():
            log.error(f"Missing: {f}")
            sys.exit(1)

    with open(quant_cfg) as f:
        cfg = json.load(f)
    log.info(f"Config: {cfg}")

    with open(manifest_f) as f:
        manifest = json.load(f)
    log.info(f"Manifest: {len(manifest)} projection layers")

    tensors = load_file(str(shard_f))
    log.info(f"Shard: {len(tensors)} tensors loaded")

    errors = 0
    for key, meta in manifest.items():
        for tkey in (meta["W_q"], meta["scales"], meta["zeros"]):
            if tkey not in tensors:
                log.error(f"  MISSING tensor: {tkey}")
                errors += 1
                continue
            t = tensors[tkey]
            if not t.isfinite().all():
                log.error(f"  NON-FINITE: {tkey}")
                errors += 1
            else:
                pass  # all good
        log.info(
            f"  {key}: W_q{list(tensors[meta['W_q']].shape)} "
            f"scales{list(tensors[meta['scales']].shape)} OK"
        )

    if errors:
        log.error(f"Verification FAILED: {errors} error(s)")
        sys.exit(1)

    # Spot-check: dequantize a small slice of one projection and confirm values are sane
    first_key = next(iter(manifest))
    meta   = manifest[first_key]
    W_q_t  = tensors[meta["W_q"]].to(torch.uint8)   # [K//2, N]
    scales = tensors[meta["scales"]].float()          # [n_g, N]
    zeros  = tensors[meta["zeros"]].float()           # [n_g, N]
    gs     = meta["group_size"]

    # Dequantize first group only (cheap, avoids large intermediate tensors)
    w_slice = W_q_t[:gs // 2, :256]                  # [gs//2, 256]
    w_lo = (w_slice & 0xF).float()
    w_hi = (w_slice >> 4).float()
    w_int = torch.stack([w_lo, w_hi], dim=1).reshape(gs, 256)  # [gs, 256]
    sc = scales[0, :256].unsqueeze(0)                # [1, 256]
    zr = zeros[0, :256].unsqueeze(0)                 # [1, 256]
    w_dq = (w_int + zr) * sc
    w_max = w_dq.abs().max().item()
    assert w_max < 1e4, f"Dequantized values look wrong: max={w_max}"

    log.info(f"Spot-check dequantization: OK (max |w_dq| ≈ {w_max:.3f})")
    log.info("Verification PASSED.")


def main():
    parser = argparse.ArgumentParser(description="NanoQuant: Convert Nemotron-Nano-9B-v2 to W4A16")
    sub = parser.add_subparsers(dest="cmd")

    conv = sub.add_parser("convert", help="Quantize SSM layers and save W4A16 checkpoint")
    conv.add_argument("--model",      default="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                      help="HF model ID or local path to base model")
    conv.add_argument("--output",     default="./Nemotron-Nano-9B-v2-W4A16")
    conv.add_argument("--group-size", type=int, default=128, dest="group_size")

    ver = sub.add_parser("verify", help="Verify W4A16 shard integrity (no vLLM/mamba-ssm needed)")
    ver.add_argument("checkpoint_dir")

    args = parser.parse_args()

    if args.cmd == "convert":
        convert(args.model, args.output, args.group_size)
    elif args.cmd == "verify":
        verify(args.checkpoint_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
