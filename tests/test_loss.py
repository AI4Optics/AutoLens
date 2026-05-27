"""Tests for deeplens/loss.py — PSFLoss and PSFStrehlLoss."""

import pytest
import torch

from deeplens.loss import PSFLoss, PSFStrehlLoss


class TestPSFLoss:
    def test_returns_scalar(self, device):
        psf = torch.rand(1, 3, 32, 32, device=device)
        loss_fn = PSFLoss()
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_non_negative(self, device):
        psf = torch.rand(1, 3, 32, 32, device=device)
        loss_fn = PSFLoss()
        loss = loss_fn(psf)
        assert loss.item() >= 0

    def test_delta_psf_low_loss(self, device):
        """A delta-like PSF (all energy at center) should have low concentration loss."""
        psf = torch.zeros(1, 3, 32, 32, device=device)
        psf[:, :, 16, 16] = 1.0
        loss_fn = PSFLoss(w_achromatic=0.0, w_psf_size=1.0)
        loss = loss_fn(psf)
        assert loss.item() < 0.01

    def test_achromatic_loss_zero_for_identical_channels(self, device):
        """Identical channels should have zero achromatic loss."""
        channel = torch.rand(1, 1, 16, 16, device=device)
        psf = channel.repeat(1, 3, 1, 1)
        loss_fn = PSFLoss(w_achromatic=1.0, w_psf_size=0.0)
        loss = loss_fn(psf)
        assert loss.item() < 1e-6

    def test_differentiable(self, device):
        psf = torch.rand(1, 3, 16, 16, device=device, requires_grad=True)
        loss_fn = PSFLoss()
        loss = loss_fn(psf)
        loss.backward()
        assert psf.grad is not None

    def test_handles_2d_input(self, device):
        """Should handle 2D input (single channel, no batch)."""
        psf = torch.rand(16, 16, device=device)
        loss_fn = PSFLoss()
        loss = loss_fn(psf)
        assert loss.dim() == 0

    def test_handles_3d_input(self, device):
        """Should handle 3D input (multi-channel, no batch)."""
        psf = torch.rand(3, 16, 16, device=device)
        loss_fn = PSFLoss()
        loss = loss_fn(psf)
        assert loss.dim() == 0


class TestPSFStrehlLoss:
    def test_returns_scalar(self, device):
        psf = torch.rand(1, 3, 32, 32, device=device)
        loss_fn = PSFStrehlLoss()
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_delta_psf_high_strehl(self, device):
        """A delta-like PSF should have high Strehl ratio."""
        psf = torch.zeros(2, 3, 32, 32, device=device)
        psf[:, :, 16, 16] = 1.0
        loss_fn = PSFStrehlLoss()
        strehl = loss_fn(psf)
        assert strehl.item() > 0.5

    def test_uniform_psf_low_strehl(self, device):
        """A uniform PSF should have low Strehl ratio."""
        psf = torch.ones(1, 3, 64, 64, device=device)
        loss_fn = PSFStrehlLoss()
        strehl = loss_fn(psf)
        assert strehl.item() < 0.1

    def test_differentiable(self, device):
        psf = torch.rand(1, 3, 16, 16, device=device, requires_grad=True)
        loss_fn = PSFStrehlLoss()
        loss = loss_fn(psf)
        loss.backward()
        assert psf.grad is not None
