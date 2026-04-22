"""
Save / load W4A16-quantized SSM weights as a standalone checkpoint shard.

Checkpoint layout (inside the HF model directory):
  w4a16_weights.safetensors   — packed int4 weights, scales, zeros for all SSM layers
  w4a16_manifest.json         — maps layer path → {W_q, scales, zeros, group_size}
  quantization_config.json    — quant metadata (read by serve.py to decide load path)

The base model config.json / tokenizer / BF16 weights remain untouched so the
checkpoint stays compatible with stock transformers inference for the non-SSM layers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

SHARD_FILENAME    = "w4a16_weights.safetensors"
MANIFEST_FILENAME = "w4a16_manifest.json"
QUANT_CONFIG_FILE = "quantization_config.json"


def save_w4a16_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    group_size: int = 128,
) -> Dict[str, int]:
    """
    Extract W4A16-quantized SSM weights from a patched model and save them
    as a standalone shard alongside the base model checkpoint.

    The base model must already be saved to output_dir (via model.save_pretrained).
    This function adds the three W4A16 files on top.

    Returns dict with counts: {ssm_layers, projections, saved_bytes}.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    from nanoquant.linear import W4A16Linear

    output_path = Path(output_dir)
    tensors: Dict[str, torch.Tensor] = {}
    manifest: Dict[str, dict] = {}
    stats = {"ssm_layers": 0, "projections": 0, "saved_bytes": 0}

    for name, module in model.named_modules():
        if type(module).__name__ != "MambaMixer2":
            continue
        stats["ssm_layers"] += 1

        for proj_name in ("in_proj", "out_proj"):
            proj = getattr(module, proj_name, None)
            if not isinstance(proj, W4A16Linear):
                log.warning(f"{name}.{proj_name} is not W4A16Linear — skipping")
                continue

            key = f"{name}.{proj_name}"
            tensors[f"{key}.W_q"]    = proj.W_q.cpu()
            tensors[f"{key}.scales"] = proj.scales.cpu()
            tensors[f"{key}.zeros"]  = proj.zeros.cpu()

            manifest[key] = {
                "W_q":       f"{key}.W_q",
                "scales":    f"{key}.scales",
                "zeros":     f"{key}.zeros",
                "group_size": proj.group_size,
                "out_features": proj.out_features,
                "in_features":  proj.in_features,
            }

            orig_bytes = proj.out_features * proj.in_features * 2   # bf16
            quant_bytes = proj.W_q.numel() + (proj.scales.numel() + proj.zeros.numel()) * 2
            stats["saved_bytes"]  += orig_bytes - quant_bytes
            stats["projections"]  += 1

    shard_path = output_path / SHARD_FILENAME
    save_file(tensors, str(shard_path))
    log.info(f"Saved {stats['projections']} projection tensors to {shard_path}")

    with open(output_path / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2)

    quant_config = {
        "quant_type":  "w4a16",
        "group_size":   group_size,
        "bits":         4,
        "zero_point":   True,
        "layout":       "column_major_packed",
        "shard_file":   SHARD_FILENAME,
        "manifest_file": MANIFEST_FILENAME,
        "nanoquant_version": "0.1.0",
    }
    with open(output_path / QUANT_CONFIG_FILE, "w") as f:
        json.dump(quant_config, f, indent=2)

    return stats


def load_w4a16_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
) -> int:
    """
    Load pre-quantized W4A16 weights into a model's MambaMixer2 layers,
    replacing in_proj / out_proj with W4A16Linear without re-quantizing.

    This is faster than quantize_on_load and produces identical results.
    Returns the number of layers loaded.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("pip install safetensors")

    from nanoquant.linear import W4A16Linear
    from nanoquant.kernel import quantize_w4  # noqa — registers custom op

    checkpoint_path = Path(checkpoint_dir)
    manifest_path   = checkpoint_path / MANIFEST_FILENAME
    shard_path      = checkpoint_path / SHARD_FILENAME

    if not manifest_path.exists() or not shard_path.exists():
        raise FileNotFoundError(
            f"W4A16 checkpoint files not found in {checkpoint_dir}. "
            f"Expected {MANIFEST_FILENAME} and {SHARD_FILENAME}."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    tensors = load_file(str(shard_path))
    log.info(f"Loaded W4A16 shard: {shard_path} ({len(tensors)} tensors)")

    # Build a lookup: normalised module path → manifest entry.
    # Handles two manifest key formats:
    #   save_w4a16_checkpoint:  "layers.0.mixer.in_proj"          (module path)
    #   convert.py (direct):    "model.layers.0.mixer.in_proj.weight" (state dict key)
    def _find_meta(name: str, proj_name: str):
        candidates = (
            f"{name}.{proj_name}",                        # module path (save_w4a16_checkpoint)
            f"{name}.{proj_name}.weight",                 # state dict key (convert.py)
            f"model.{name}.{proj_name}.weight",           # state dict with model. prefix
            f"backbone.{name}.{proj_name}.weight",        # Nemotron-H uses backbone. prefix
        )
        for c in candidates:
            if c in manifest:
                return manifest[c]
        return None

    loaded = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "MambaMixer2":
            continue

        for proj_name in ("in_proj", "out_proj"):
            meta = _find_meta(name, proj_name)
            if meta is None:
                continue
            W_q    = tensors[meta["W_q"]]
            scales = tensors[meta["scales"]]
            zeros  = tensors[meta["zeros"]]
            gs     = meta["group_size"]
            out_f  = meta["out_features"]
            in_f   = meta["in_features"]

            proj = getattr(module, proj_name)
            device = proj.weight.device if hasattr(proj, "weight") else next(model.parameters()).device

            # Build W4A16Linear from pre-quantized tensors (no re-quantization)
            q_linear = W4A16Linear.__new__(W4A16Linear)
            nn.Module.__init__(q_linear)
            q_linear.group_size   = gs
            q_linear.out_features = out_f
            q_linear.in_features  = in_f
            q_linear.register_buffer("W_q",    W_q.to(device))
            q_linear.register_buffer("scales", scales.to(device))
            q_linear.register_buffer("zeros",  zeros.to(device))
            q_linear.bias = None

            setattr(module, proj_name, q_linear)
            loaded += 1
            log.debug(f"Loaded {key} from checkpoint")

    log.info(f"load_w4a16_checkpoint: loaded {loaded} projection layers")
    return loaded


def is_w4a16_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Return True if the directory contains a NanoQuant W4A16 checkpoint."""
    p = Path(checkpoint_dir)
    return (p / QUANT_CONFIG_FILE).exists() and (p / SHARD_FILENAME).exists()
