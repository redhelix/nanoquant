"""W4A16 quantized nn.Module — drop-in for nn.Linear."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .kernel import quantize_w4, w4a16_linear  # noqa: F401 — registers the custom op


class W4A16Linear(nn.Module):
    """
    Quantized linear layer (int4 weights, bfloat16 activations).

    Uses torch.library.custom_op so aot_compile_fullgraph sees it as a leaf:
    the GEMV loop runs eagerly during CUDA graph capture, recording the correct
    Triton kernel launches for each captured batch size.
    """

    def __init__(
        self,
        weight: torch.Tensor,            # [out_features, in_features] bfloat16
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.group_size   = group_size
        self.out_features = weight.shape[0]
        self.in_features  = weight.shape[1]

        W_q, scales, zeros = quantize_w4(weight.cpu(), group_size)
        self.register_buffer("W_q",    W_q)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros",  zeros)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Module, group_size: int = 128) -> "W4A16Linear":
        """Convert an existing nn.Linear or vLLM parallel linear in-place."""
        weight = linear.weight.data.to(torch.bfloat16)
        bias   = linear.bias.data if getattr(linear, "bias", None) is not None else None
        return cls(weight, bias, group_size)

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        y = torch.ops.nanoquant.linear(x_2d, self.W_q, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            y = y + self.bias
        if bias is not None:
            y = y + bias
        out_shape = orig_shape[:-1] + (self.out_features,)
        return y.reshape(out_shape), None   # None = bias passthrough (vLLM convention)
