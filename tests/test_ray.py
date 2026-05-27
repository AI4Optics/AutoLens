"""Tests for deeplens/light/ray.py — Ray class."""

import pytest
import torch

from deeplens import Ray
from deeplens.config import DEFAULT_WAVE


class TestRayInit:
    def test_basic_creation(self, device):
        o = torch.zeros(10, 3, device=device)
        d = torch.tensor([0.0, 0.0, 1.0], device=device).expand(10, 3)
        ray = Ray(o, d, wvln=0.55, device=device)
        assert ray.o.shape == (10, 3)
        assert ray.d.shape == (10, 3)
        assert ray.is_valid.shape == (10,)

    def test_direction_normalized(self, device):
        o = torch.zeros(5, 3, device=device)
        d = torch.tensor([1.0, 1.0, 1.0], device=device).expand(5, 3)
        ray = Ray(o, d, device=device)
        norms = ray.d.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_wavelength_validation(self, device):
        o = torch.zeros(1, 3, device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        with pytest.raises(AssertionError):
            Ray(o, d, wvln=0.05, device=device)  # too small
        with pytest.raises(AssertionError):
            Ray(o, d, wvln=15.0, device=device)  # too large

    def test_batched_creation(self, device):
        o = torch.zeros(4, 8, 3, device=device)
        d = torch.zeros(4, 8, 3, device=device)
        d[..., 2] = 1.0
        ray = Ray(o, d, device=device)
        assert ray.o.shape == (4, 8, 3)
        assert ray.is_valid.shape == (4, 8)

    def test_default_validity_all_ones(self, device):
        o = torch.zeros(5, 3, device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(5, 3)
        ray = Ray(o, d, device=device)
        assert (ray.is_valid == 1.0).all()


class TestRayPropagation:
    def test_prop_to_z(self, device):
        o = torch.zeros(3, 3, device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(3, 3)
        ray = Ray(o, d, device=device)
        ray.prop_to(10.0)
        assert torch.allclose(ray.o[:, 2], torch.tensor(10.0, device=device))

    def test_prop_preserves_xy_for_axial_rays(self, device):
        o = torch.tensor([[1.0, 2.0, 0.0]], device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        ray = Ray(o, d, device=device)
        ray.prop_to(5.0)
        assert torch.allclose(ray.o[0, :2], torch.tensor([1.0, 2.0], device=device))

    def test_prop_tilted_ray(self, device):
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        d = torch.tensor([[0.0, 1.0, 1.0]], device=device)
        ray = Ray(o, d, device=device)
        ray.prop_to(5.0)
        # y should equal z since dy/dz = 1
        assert torch.allclose(ray.o[0, 1], ray.o[0, 2], atol=1e-5)


class TestRayCentroid:
    def test_centroid_symmetric(self, device):
        # Symmetric rays should have centroid at origin in x,y
        o = torch.tensor([
            [1.0, 0.0, 5.0],
            [-1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [0.0, -1.0, 5.0],
        ], device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(4, 3)
        ray = Ray(o, d, device=device)
        centroid = ray.centroid()
        assert centroid.shape[-1] == 3
        assert torch.abs(centroid[..., 0]) < 1e-5
        assert torch.abs(centroid[..., 1]) < 1e-5

    def test_centroid_respects_validity(self, device):
        o = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(2, 3)
        ray = Ray(o, d, device=device)
        ray.is_valid[0] = 0  # invalidate the (2,0,0) ray
        centroid = ray.centroid()
        assert torch.abs(centroid[..., 0]) < 1e-5  # only (0,0,0) contributes


class TestRayCoherent:
    def test_opl_tracking(self, device):
        o = torch.zeros(3, 3, device=device, dtype=torch.float64)
        d = torch.zeros(3, 3, device=device, dtype=torch.float64)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, coherent=True, device=device)
        ray.prop_to(10.0, n=1.5)
        expected_opl = 1.5 * 10.0
        assert torch.allclose(ray.opl.squeeze(-1), torch.tensor(expected_opl, device=device, dtype=torch.float64), atol=1e-10)
