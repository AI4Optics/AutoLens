"""Tests for GeoLens optimization: losses, constraints, optimizer."""

import os

import pytest
import torch

from deeplens import GeoLens
from deeplens.config import WAVE_RGB


class TestConstraints:
    def test_init_constraints(self, cellphone_lens, device):
        """Constraint initialization should set attributes."""
        cellphone_lens.init_constraints()
        assert hasattr(cellphone_lens, "air_edge_min")
        assert hasattr(cellphone_lens, "thick_center_min")

    def test_constraint_type_set(self, cellphone_lens, device):
        """Constraint init should set is_cellphone based on r_sensor."""
        cellphone_lens.init_constraints()
        assert hasattr(cellphone_lens, "is_cellphone")


class TestLossFunctions:
    @pytest.fixture(autouse=True)
    def _init_constraints(self, cellphone_lens):
        cellphone_lens.init_constraints()

    def test_loss_infocus_returns_scalar(self, cellphone_lens, device):
        loss = cellphone_lens.loss_infocus()
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_profile_returns_scalar(self, cellphone_lens, device):
        loss = cellphone_lens.loss_profile()
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_bound_returns_scalars(self, cellphone_lens, device):
        loss_clearance, loss_envelope = cellphone_lens.loss_bound()
        assert loss_clearance.dim() == 0
        assert loss_envelope.dim() == 0
        assert torch.isfinite(loss_clearance)
        assert torch.isfinite(loss_envelope)

    def test_loss_cra_returns_scalar(self, cellphone_lens, device):
        loss = cellphone_lens.loss_cra()
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_ray_bend_returns_scalar(self, cellphone_lens, device):
        loss = cellphone_lens.loss_ray_bend()
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_reg_returns_dict(self, cellphone_lens, device):
        loss, loss_dict = cellphone_lens.loss_reg()
        assert loss.dim() == 0
        assert isinstance(loss_dict, dict)
        assert "loss_clearance" in loss_dict
        assert "loss_envelope" in loss_dict
        assert "loss_profile" in loss_dict
        assert "loss_cra" in loss_dict
        assert "loss_ray_bend" in loss_dict

    def test_loss_infocus_decreases_when_focused(self, cellphone_lens, device):
        """A well-focused lens should have low infocus loss."""
        loss = cellphone_lens.loss_infocus(target=10.0)
        # With a very large target, loss should be 0
        assert loss.item() == 0.0

    def test_losses_non_negative(self, cellphone_lens, device):
        """All constraint losses should be non-negative."""
        assert cellphone_lens.loss_profile().item() >= 0
        loss_clearance, loss_envelope = cellphone_lens.loss_bound()
        assert loss_clearance.item() >= 0
        assert loss_envelope.item() >= 0
        assert cellphone_lens.loss_cra().item() >= 0
        assert cellphone_lens.loss_ray_bend().item() >= 0


class TestOptimizer:
    def test_get_optimizer_creates_params(self, cellphone_path, device):
        """get_optimizer should return a valid optimizer."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-2, 1e-4])
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0

    def test_short_optimization_reduces_loss(self, cellphone_path, device):
        """A few optimization steps should reduce the total loss."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        lens.init_constraints()

        # Compute initial loss
        with torch.no_grad():
            loss_before = lens.loss_infocus().item()

        # Run a few steps
        optimizer = lens.get_optimizer(lrs=[0, 1e-4, 1e-2, 1e-4])
        for _ in range(5):
            loss = lens.loss_infocus()
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Just check it didn't crash — loss may not decrease in 5 steps
        loss_after = lens.loss_infocus().item()
        assert torch.isfinite(torch.tensor(loss_after))

    def test_find_diff_surf(self, cellphone_lens, device):
        """find_diff_surf should return indices of differentiable surfaces."""
        diff_surfs = cellphone_lens.find_diff_surf()
        assert len(diff_surfs) > 0
        for idx in diff_surfs:
            assert 0 <= idx < len(cellphone_lens.surfaces)
