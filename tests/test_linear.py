"""
Tests for W4A16Linear module and patch_nemotron_h().

Run: pytest tests/test_linear.py -v
Requires: CUDA GPU
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanoquant.linear import W4A16Linear
from nanoquant.patch import patch_nemotron_h

DTYPE = torch.bfloat16
DEV   = "cuda" if torch.cuda.is_available() else "cpu"
SKIP  = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Toy model that mimics NemotronH's MambaMixer2 structure ──────────────────

class _FakeLinear(nn.Module):
    def __init__(self, out_f, in_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=DTYPE))
        self.bias   = None

    def forward(self, x, bias=None):
        return nn.functional.linear(x, self.weight), None


class MambaMixer2(nn.Module):
    def __init__(self, hidden=256, d_inner=512):
        super().__init__()
        self.in_proj  = _FakeLinear(d_inner * 2, hidden)
        self.out_proj = _FakeLinear(hidden, d_inner)

    def forward(self, x):
        h, _ = self.in_proj(x)
        y, _ = self.out_proj(h[:, :self.out_proj.weight.shape[1]])
        return y


class FakeNemotronH(nn.Module):
    def __init__(self, n_ssm=4, hidden=256, d_inner=512):
        super().__init__()
        self.layers = nn.ModuleList([MambaMixer2(hidden, d_inner) for _ in range(n_ssm)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ── W4A16Linear tests ─────────────────────────────────────────────────────────

@SKIP
def test_from_linear_shapes():
    linear = nn.Linear(4480, 22656, bias=False)
    q = W4A16Linear.from_linear(linear, group_size=128).to(DEV)
    assert q.W_q.shape    == (4480 // 2, 22656)
    assert q.scales.shape == (4480 // 128, 22656)
    assert q.zeros.shape  == (4480 // 128, 22656)
    assert q.W_q.dtype    == torch.uint8
    assert q.scales.dtype == DTYPE


@SKIP
def test_forward_shape_batch1():
    linear = nn.Linear(512, 1024, bias=False)
    q = W4A16Linear.from_linear(linear, group_size=128).to(DEV)
    x = torch.randn(1, 512, dtype=DTYPE, device=DEV)
    y, bias_out = q(x)
    assert y.shape    == (1, 1024)
    assert bias_out   is None
    assert y.dtype    == DTYPE
    assert y.isfinite().all()


@SKIP
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_forward_batched(batch):
    linear = nn.Linear(512, 512, bias=False)
    q = W4A16Linear.from_linear(linear).to(DEV)
    x = torch.randn(batch, 512, dtype=DTYPE, device=DEV)
    y, _ = q(x)
    assert y.shape == (batch, 512)
    assert y.isfinite().all()


@SKIP
def test_forward_3d_input():
    """W4A16Linear should handle [seq, batch, hidden] inputs."""
    linear = nn.Linear(256, 512, bias=False)
    q = W4A16Linear.from_linear(linear).to(DEV)
    x = torch.randn(10, 2, 256, dtype=DTYPE, device=DEV)
    y, _ = q(x)
    assert y.shape == (10, 2, 512)


@SKIP
def test_forward_close_to_bf16():
    """W4A16 output should be within quantization noise of BF16 matmul."""
    torch.manual_seed(0)
    N, K = 512, 512
    linear = nn.Linear(K, N, bias=False)
    q = W4A16Linear.from_linear(linear).to(DEV)

    x = torch.randn(1, K, dtype=DTYPE, device=DEV)
    y_q, _ = q(x)
    y_ref  = nn.functional.linear(x, linear.weight.to(DEV).to(DTYPE))

    err = (y_q.float() - y_ref.float()).abs().max().item()
    assert err < 4.0, f"max error {err} exceeds quantization tolerance"


# ── patch_nemotron_h tests ────────────────────────────────────────────────────

@SKIP
def test_patch_replaces_projections():
    model = FakeNemotronH(n_ssm=4).to(DEV)
    n = patch_nemotron_h(model, group_size=64, verbose=False)
    assert n == 4
    for layer in model.layers:
        assert isinstance(layer.in_proj,  W4A16Linear)
        assert isinstance(layer.out_proj, W4A16Linear)


@SKIP
def test_patch_decode_forward():
    model = FakeNemotronH(n_ssm=2, hidden=256, d_inner=512).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)
    x = torch.randn(1, 256, dtype=DTYPE, device=DEV)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 256)
    assert y.isfinite().all()


@SKIP
def test_patch_prefill_forward():
    model = FakeNemotronH(n_ssm=2, hidden=256, d_inner=512).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)
    x = torch.randn(8, 256, dtype=DTYPE, device=DEV)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 256)
    assert y.isfinite().all()


@SKIP
def test_patch_idempotent():
    model = FakeNemotronH(n_ssm=2).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)
    n2 = patch_nemotron_h(model, group_size=64, verbose=False)
    assert n2 == 2  # already patched layers skipped but count still returned


@SKIP
def test_patch_selective_layers():
    model = FakeNemotronH(n_ssm=4).to(DEV)
    n = patch_nemotron_h(model, group_size=64, layers=[0, 2], verbose=False)
    assert n == 2
    assert isinstance(model.layers[0].in_proj, W4A16Linear)
    assert not isinstance(model.layers[1].in_proj, W4A16Linear)
    assert isinstance(model.layers[2].in_proj, W4A16Linear)
    assert not isinstance(model.layers[3].in_proj, W4A16Linear)
