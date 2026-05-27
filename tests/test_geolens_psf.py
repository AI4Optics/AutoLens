"""Tests for GeoLens PSF computation."""

import pytest
import torch

from deeplens import GeoLens
from deeplens.config import DEPTH


class TestPSFGeometric:
    def test_psf_returns_tensor(self, cellphone_lens, device):
        """Geometric PSF should return a valid tensor."""
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        psf = cellphone_lens.psf(points, ks=32, spp=512, model="geometric")
        assert psf is not None
        assert psf.numel() > 0
        assert torch.isfinite(psf).all()

    def test_psf_shape(self, cellphone_lens, device):
        """PSF should have ks x ks spatial dimensions."""
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        ks = 32
        psf = cellphone_lens.psf(points, ks=ks, spp=512, model="geometric")
        assert psf.shape[-1] == ks
        assert psf.shape[-2] == ks

    def test_psf_non_negative(self, cellphone_lens, device):
        """PSF values should be non-negative (intensity distribution)."""
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        psf = cellphone_lens.psf(points, ks=32, spp=512, model="geometric")
        assert (psf >= 0).all()

    def test_psf_multiple_fields(self, cellphone_lens, device):
        """PSF should work with multiple field points."""
        points = torch.tensor([
            [0.0, 0.0, DEPTH],
            [0.0, 0.5, DEPTH],
        ], device=device)
        psf = cellphone_lens.psf(points, ks=16, spp=256, model="geometric")
        assert psf is not None
        assert torch.isfinite(psf).all()


class TestPSFMap:
    def test_psf_map_runs(self, cellphone_lens, device):
        """psf_map should run without errors."""
        psf_map = cellphone_lens.psf_map(ks=16, spp=256)
        assert psf_map is not None
