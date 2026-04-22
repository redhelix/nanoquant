"""
Correctness tests for the W4A16 GEMV kernel.

Reference: dequantize W then do BF16 matmul — these should agree
within quantization noise (atol=4.0 covers ~3-sigma for randn weights).

Run: pytest tests/test_kernel.py -v
Requires: CUDA GPU
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanoquant.kernel import quantize_w4, gemv_w4a16, w4a16_linear

DTYPE = torch.bfloat16
DEV   = "cuda"
GROUP = 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N,K", [
    (22656, 4480),   # Nemotron-Nano in_proj (actual shape)
    (4480, 10240),   # Nemotron-Nano out_proj (actual shape)
    (128,  256),     # small sanity
    (512,  512),     # square
    (1024, 640),     # group-boundary edge case (640 % 128 == 0)
])
def test_gemv_correctness(N, K):
    torch.manual_seed(42)
    x = torch.randn(1, K, dtype=DTYPE, device=DEV)
    W = torch.randn(N, K, dtype=DTYPE, device=DEV)

    W_q, scales, zeros = quantize_w4(W.cpu(), GROUP)
    W_q    = W_q.to(DEV)
    scales = scales.to(DEV)
    zeros  = zeros.to(DEV)

    y_w4 = gemv_w4a16(x, W_q, scales, zeros, GROUP)

    # Reference: dequantize then bf16 matmul
    W_f = W.float()
    W_g = W_f.reshape(N, K // GROUP, GROUP)
    mn  = W_g.amin(dim=-1, keepdim=True)
    mx  = W_g.amax(dim=-1, keepdim=True)
    sc  = ((mx - mn) / 15.0).clamp(min=1e-8)
    zr  = mn / sc
    W_int = (W_g / sc - zr).round().clamp(0, 15)
    W_dq  = ((W_int + zr) * sc).reshape(N, K).to(DTYPE)
    y_ref = torch.nn.functional.linear(x, W_dq)

    assert y_w4.shape == (1, N)
    assert y_w4.dtype == DTYPE
    assert y_w4.isfinite().all(), "non-finite output"
    torch.testing.assert_close(y_w4.float(), y_ref.float(), atol=4.0, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemv_1d_input():
    """gemv_w4a16 should accept a flat [K] input (not just [1, K])."""
    N, K = 4480, 8960
    x = torch.randn(K, dtype=DTYPE, device=DEV)
    W_q, scales, zeros = quantize_w4(torch.randn(N, K), GROUP)
    y = gemv_w4a16(x, W_q.to(DEV), scales.to(DEV), zeros.to(DEV), GROUP)
    assert y.shape == (1, N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_custom_op_batched(batch):
    """w4a16_linear custom_op should handle B > 1 (captured in CUDA graph per size)."""
    N, K = 512, 512
    torch.manual_seed(0)
    x = torch.randn(batch, K, dtype=DTYPE, device=DEV)
    W = torch.randn(N, K, dtype=DTYPE)
    W_q, scales, zeros = quantize_w4(W, GROUP)
    W_q    = W_q.to(DEV)
    scales = scales.to(DEV)
    zeros  = zeros.to(DEV)

    y = w4a16_linear(x, W_q, scales, zeros, GROUP)
    assert y.shape == (batch, N)
    assert y.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantize_w4_roundtrip():
    """Dequantized values should be close to originals (within int4 quantization error)."""
    N, K = 512, 512
    torch.manual_seed(7)
    W = torch.randn(N, K, dtype=DTYPE)
    W_q, scales, zeros = quantize_w4(W, GROUP)

    # Dequantize
    W_lo = (W_q & 0xF).float()                  # [K//2, N]
    W_hi = (W_q >> 4).float()                   # [K//2, N]
    # interleave even/odd K positions
    W_int = torch.stack([W_lo, W_hi], dim=1).reshape(K, N)  # [K, N]
    W_int = W_int.T                                          # [N, K]
    W_int_g = W_int.reshape(N, K // GROUP, GROUP)

    sc = scales.T.float().reshape(N, K // GROUP, 1)
    zr = zeros.T.float().reshape(N, K // GROUP, 1)
    W_dq = ((W_int_g + zr) * sc).reshape(N, K).to(DTYPE)

    err = (W.float() - W_dq.float()).abs()
    # max error should be < 0.5 * max_scale (half an int4 step)
    assert err.max().item() < 1.0, f"max roundtrip error too large: {err.max().item()}"
    assert err.mean().item() < 0.05, f"mean roundtrip error too large: {err.mean().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantize_group_size_variants():
    """Kernel should work with group_size = 64 and 256 as well."""
    N, K = 512, 512
    x = torch.randn(1, K, dtype=DTYPE, device=DEV)
    W = torch.randn(N, K, dtype=DTYPE)
    for gs in (64, 128, 256):
        W_q, scales, zeros = quantize_w4(W, gs)
        y = gemv_w4a16(x, W_q.to(DEV), scales.to(DEV), zeros.to(DEV), gs)
        assert y.shape == (1, N)
        assert y.isfinite().all(), f"non-finite output at group_size={gs}"
