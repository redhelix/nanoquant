"""
Patch a loaded vLLM NemotronH model to use W4A16 GEMV for SSM layers.

Usage:
    from nanoquant.patch import patch_nemotron_h
    patch_nemotron_h(llm.llm_engine.model_executor.driver_worker.model_runner.model)
"""

import logging
import torch.nn as nn
from typing import Optional

log = logging.getLogger(__name__)


def patch_nemotron_h(
    model: nn.Module,
    group_size: int = 128,
    layers: Optional[list] = None,
    verbose: bool = True,
) -> int:
    """
    Replace in_proj and out_proj in every MambaMixer2 layer with W4A16Linear.

    Args:
        model:      Top-level vLLM model (NemotronHForCausalLM).
        group_size: Quantization group size. Must divide SSM hidden dims.
        layers:     Restrict patching to these layer indices. None = all SSM layers.
        verbose:    Log each patched layer.

    Returns:
        Number of MambaMixer2 layers patched.
    """
    from .linear import W4A16Linear

    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "MambaMixer2":
            continue

        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)

        if layers is not None and layer_idx not in layers:
            continue

        for proj_name in ("in_proj", "out_proj"):
            proj = getattr(module, proj_name, None)
            if proj is None or isinstance(proj, W4A16Linear):
                continue
            try:
                w4 = W4A16Linear.from_linear(proj, group_size=group_size)
                w4 = w4.to(proj.weight.device)
                setattr(module, proj_name, w4)
                if verbose:
                    w = proj.weight
                    log.info(f"Patched {name}.{proj_name} [{w.shape[0]},{w.shape[1]}] -> W4A16")
            except Exception as e:
                log.warning(f"Could not patch {name}.{proj_name}: {e}")

        patched += 1

    if verbose:
        log.info(f"patch_nemotron_h: patched {patched} MambaMixer2 layers")
    return patched
