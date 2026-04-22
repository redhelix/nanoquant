"""
Tests for save/load W4A16 checkpoint round-trip.

Run: pytest tests/test_checkpoint.py -v
Requires: CUDA GPU + safetensors
"""

import json
import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanoquant.linear import W4A16Linear
from nanoquant.patch import patch_nemotron_h
from nanoquant.checkpoint import (
    save_w4a16_checkpoint,
    load_w4a16_checkpoint,
    is_w4a16_checkpoint,
    SHARD_FILENAME,
    MANIFEST_FILENAME,
    QUANT_CONFIG_FILE,
)

SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DTYPE = torch.bfloat16
DEV   = "cuda" if torch.cuda.is_available() else "cpu"


class _FakeLinear(nn.Module):
    def __init__(self, out_f, in_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=DTYPE))
        self.bias = None

    def forward(self, x, bias=None):
        return nn.functional.linear(x, self.weight), None


class MambaMixer2(nn.Module):
    def __init__(self, hidden=256, d_inner=512):
        super().__init__()
        self.in_proj  = _FakeLinear(d_inner * 2, hidden)
        self.out_proj = _FakeLinear(hidden, d_inner)


class FakeNemotronH(nn.Module):
    def __init__(self, n_ssm=2):
        super().__init__()
        self.layers = nn.ModuleList([MambaMixer2() for _ in range(n_ssm)])


@SKIP
def test_save_creates_expected_files():
    pytest.importorskip("safetensors")
    model = FakeNemotronH(n_ssm=2).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_w4a16_checkpoint(model, tmpdir, group_size=64)
        assert (Path(tmpdir) / SHARD_FILENAME).exists()
        assert (Path(tmpdir) / MANIFEST_FILENAME).exists()
        assert (Path(tmpdir) / QUANT_CONFIG_FILE).exists()

        with open(Path(tmpdir) / QUANT_CONFIG_FILE) as f:
            cfg = json.load(f)
        assert cfg["quant_type"] == "w4a16"
        assert cfg["group_size"] == 64


@SKIP
def test_is_w4a16_checkpoint_positive():
    pytest.importorskip("safetensors")
    model = FakeNemotronH(n_ssm=2).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_w4a16_checkpoint(model, tmpdir, group_size=64)
        assert is_w4a16_checkpoint(tmpdir)


@SKIP
def test_is_w4a16_checkpoint_negative(tmp_path):
    assert not is_w4a16_checkpoint(tmp_path)
    assert not is_w4a16_checkpoint("/nonexistent/path")


@SKIP
def test_roundtrip_outputs_match():
    """Load from checkpoint must produce identical outputs to the patched model."""
    pytest.importorskip("safetensors")
    torch.manual_seed(42)
    model_orig = FakeNemotronH(n_ssm=2).to(DEV)
    patch_nemotron_h(model_orig, group_size=64, verbose=False)

    x = torch.randn(1, 256, dtype=DTYPE, device=DEV)
    with torch.no_grad():
        y_orig = model_orig.layers[0].in_proj(x)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_w4a16_checkpoint(model_orig, tmpdir, group_size=64)

        # Fresh model with unpatched weights — simulates loading from disk
        model_loaded = FakeNemotronH(n_ssm=2).to(DEV)
        # Set same base weights so the only difference is quantization method
        for (name_o, mod_o), (name_l, mod_l) in zip(
            model_orig.named_modules(), model_loaded.named_modules()
        ):
            pass  # weights differ — we only verify the loaded Q weights match
        load_w4a16_checkpoint(model_loaded, tmpdir)

        with torch.no_grad():
            y_loaded = model_loaded.layers[0].in_proj(x)[0]

    # Loaded checkpoint must be bitwise identical to original patched model
    torch.testing.assert_close(y_orig, y_loaded, atol=0.0, rtol=0.0)


@SKIP
def test_load_layer_count():
    pytest.importorskip("safetensors")
    model = FakeNemotronH(n_ssm=3).to(DEV)
    patch_nemotron_h(model, group_size=64, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_w4a16_checkpoint(model, tmpdir, group_size=64)
        model2 = FakeNemotronH(n_ssm=3).to(DEV)
        n = load_w4a16_checkpoint(model2, tmpdir)

    # 3 SSM layers × 2 projections = 6
    assert n == 6
