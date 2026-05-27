"""Tests for deeplens/phase_surface/binary2.py — Binary2Phase surface."""

import math

import pytest
import torch

from deeplens.phase_surface import Binary2Phase


class TestBinary2PhaseInit:
    def test_default_zero_coefficients(self, device):
        s = Binary2Phase(r=1.0, d=0.0, device=device)
        assert s.order2.item() == 0.0
        assert s.order12.item() == 0.0

    def test_custom_coefficients(self, device):
        s = Binary2Phase(
            r=1.0, d=0.0, order2=100.0, order4=-50.0, device=device
        )
        assert abs(s.order2.item() - 100.0) < 1e-6
        assert abs(s.order4.item() - (-50.0)) < 1e-6


class TestBinary2PhasePhase:
    def test_zero_phase_at_center(self, device):
        """Phase should be near zero at center (EPSILON in r2 gives tiny residual)."""
        s = Binary2Phase(
            r=1.0, d=0.0, order2=100.0, order4=-50.0,
            norm_radii=1.0, device=device,
        )
        x = torch.tensor([0.0], device=device)
        y = torch.tensor([0.0], device=device)
        phi = s.phi(x, y)
        assert torch.abs(phi) < 1e-4

    def test_phase_rotational_symmetry(self, device):
        """phi(r,0) should equal phi(0,r)."""
        s = Binary2Phase(
            r=2.0, d=0.0, order2=50.0, order4=-10.0,
            norm_radii=1.0, device=device,
        )
        r_val = 0.5
        phi_x = s.phi(
            torch.tensor([r_val], device=device),
            torch.tensor([0.0], device=device),
        )
        phi_y = s.phi(
            torch.tensor([0.0], device=device),
            torch.tensor([r_val], device=device),
        )
        assert torch.allclose(phi_x, phi_y, atol=1e-6)

    def test_phase_gradient_symmetry(self, device):
        """dphi/dx at (r,0) should equal dphi/dy at (0,r)."""
        s = Binary2Phase(
            r=2.0, d=0.0, order2=50.0, order4=-10.0,
            norm_radii=1.0, device=device,
        )
        r_val = 0.5
        dphidx, _ = s.dphi_dxy(
            torch.tensor([r_val], device=device),
            torch.tensor([0.0], device=device),
        )
        _, dphidy = s.dphi_dxy(
            torch.tensor([0.0], device=device),
            torch.tensor([r_val], device=device),
        )
        assert torch.allclose(dphidx, dphidy, atol=1e-6)

    def test_phase_gradient_zero_at_center(self, device):
        """Phase gradient should be near zero at center."""
        s = Binary2Phase(
            r=1.0, d=0.0, order2=100.0, norm_radii=1.0, device=device
        )
        dphidx, dphidy = s.dphi_dxy(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
        )
        assert torch.abs(dphidx) < 1e-3
        assert torch.abs(dphidy) < 1e-3


class TestBinary2PhaseLrScaling:
    def test_lr_scaling_formula(self, device):
        """lr for each order should follow lr_base / (r/norm_radii)^order."""
        s = Binary2Phase(
            r=0.5, d=0.0, norm_radii=1.0, device=device
        )
        params = s.get_optimizer_params(lrs=[0, 1e-2])
        r_norm = 0.5 / 1.0
        for i, pg in enumerate(params[1:]):  # order2..order12
            order = 2 * (i + 1)
            expected = 1e-2 / r_norm ** order
            assert abs(pg["lr"] - expected) < 1e-8

    def test_lr_scaling_monotone(self, device):
        """Higher orders should get higher lr (for r < 1, multiplier grows)."""
        s = Binary2Phase(
            r=0.5, d=0.0, norm_radii=1.0, device=device
        )
        params = s.get_optimizer_params(lrs=[0, 1e-2])
        lrs = [pg["lr"] for pg in params[1:]]  # order2..order12
        for i in range(len(lrs) - 1):
            assert lrs[i + 1] >= lrs[i] - 1e-12, "Higher orders should have >= lr"
