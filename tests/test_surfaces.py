"""Tests for geometric surfaces: Spheric, Aspheric, Aperture."""

import math

import pytest
import torch

from deeplens.geometric_surface import Aperture, Aspheric, Spheric


class TestSpheric:
    def test_flat_surface_zero_sag(self, device):
        """A flat surface (c=0) should have zero sag everywhere."""
        s = Spheric(c=0.0, r=5.0, d=0.0, mat2="air", device=device)
        x = torch.linspace(-4, 4, 20, device=device)
        y = torch.zeros_like(x)
        sag = s.sag(x, y)
        assert torch.allclose(sag, torch.zeros_like(sag), atol=1e-8)

    def test_spheric_sag_at_center(self, device):
        """Sag at center should be zero regardless of curvature."""
        s = Spheric(c=0.5, r=5.0, d=0.0, mat2="air", device=device)
        sag = s.sag(torch.tensor([0.0], device=device), torch.tensor([0.0], device=device))
        assert torch.abs(sag) < 1e-8

    def test_spheric_sag_symmetry(self, device):
        """Sag should be rotationally symmetric."""
        s = Spheric(c=0.1, r=5.0, d=0.0, mat2="air", device=device)
        r_val = 2.0
        sag_x = s.sag(torch.tensor([r_val], device=device), torch.tensor([0.0], device=device))
        sag_y = s.sag(torch.tensor([0.0], device=device), torch.tensor([r_val], device=device))
        sag_diag = s.sag(
            torch.tensor([r_val / math.sqrt(2)], device=device),
            torch.tensor([r_val / math.sqrt(2)], device=device),
        )
        assert torch.allclose(sag_x, sag_y, atol=1e-8)
        assert torch.allclose(sag_x, sag_diag, atol=1e-8)

    def test_spheric_positive_curvature_positive_sag(self, device):
        """Positive curvature should give positive sag for off-axis points."""
        s = Spheric(c=0.2, r=5.0, d=0.0, mat2="air", device=device)
        sag = s.sag(torch.tensor([3.0], device=device), torch.tensor([0.0], device=device))
        assert sag.item() > 0

    def test_init_from_dict(self):
        d = {"c": 0.1, "r": 5.0, "d": 10.0, "mat2": "n-bk7"}
        s = Spheric.init_from_dict(d)
        assert abs(s.c.item() - 0.1) < 1e-8
        assert s.r == 5.0

    def test_init_from_dict_roc(self):
        d = {"roc": 10.0, "r": 5.0, "d": 0.0, "mat2": "air"}
        s = Spheric.init_from_dict(d)
        assert abs(s.c.item() - 0.1) < 1e-8

    def test_surf_dict_roundtrip(self, device):
        s = Spheric(c=0.15, r=3.0, d=5.0, mat2="n-bk7", device=device)
        d = s.surf_dict()
        assert abs(d["(c)"] - 0.15) < 1e-6
        assert d["r"] == 3.0

    def test_bend_penalty_is_accumulated_by_ray_reaction_not_refract(self, device):
        from deeplens import Ray

        s = Spheric(c=0.4, r=2.0, d=0.0, mat2="n-bk7", device=device)
        s.bend_angle_max = 1.0

        ray = Ray(
            o=torch.tensor([[1.5, 0.0, 0.0]], device=device),
            d=torch.tensor([[0.0, 0.0, 1.0]], device=device),
            device=device,
        )
        bend_penalty = ray.bend_penalty.clone()
        ray = s.refract(ray, eta=1.0 / 1.5)
        assert torch.allclose(ray.bend_penalty, bend_penalty)

        ray = Ray(
            o=torch.tensor([[1.5, 0.0, -1.0]], device=device),
            d=torch.tensor([[0.0, 0.0, 1.0]], device=device),
            device=device,
        )
        ray = s.ray_reaction(ray, n1=1.0, n2=1.5)
        assert ray.is_valid.item() == 1.0
        assert ray.bend_penalty.item() > 0.0

class TestAspheric:
    def test_pure_conic_zero_ai(self, device):
        """Aspheric with zero ai should reduce to a conic."""
        s = Aspheric(r=5.0, d=0.0, c=0.1, k=-1.0, ai=[0, 0, 0], mat2="air", device=device)
        x = torch.tensor([2.0], device=device)
        y = torch.tensor([0.0], device=device)
        sag = s.sag(x, y)
        # Compare with conic sag formula: c*rho^2 / (1 + sqrt(1-(1+k)*c^2*rho^2))
        rho2 = 4.0
        c = 0.1
        k = -1.0
        expected = c * rho2 / (1 + math.sqrt(1 - (1 + k) * c ** 2 * rho2))
        assert abs(sag.item() - expected) < 1e-6

    def test_aspheric_sag_symmetry(self, device):
        s = Aspheric(r=5.0, d=0.0, c=0.1, k=0.0, ai=[1e-3, -1e-5], mat2="air", device=device)
        r_val = 2.0
        sag_x = s.sag(torch.tensor([r_val], device=device), torch.tensor([0.0], device=device))
        sag_y = s.sag(torch.tensor([0.0], device=device), torch.tensor([r_val], device=device))
        assert torch.allclose(sag_x, sag_y, atol=1e-8)

    def test_aspheric_center_sag_zero(self, device):
        s = Aspheric(r=5.0, d=0.0, c=0.5, k=-1.0, ai=[1e-3, 1e-5], mat2="air", device=device)
        sag = s.sag(torch.tensor([0.0], device=device), torch.tensor([0.0], device=device))
        assert torch.abs(sag) < 1e-8

    def test_ai_contributes_to_sag(self, device):
        """Adding aspheric terms should change the sag from the base conic."""
        s_conic = Aspheric(r=5.0, d=0.0, c=0.1, k=0.0, ai=[0, 0], mat2="air", device=device)
        s_asph = Aspheric(r=5.0, d=0.0, c=0.1, k=0.0, ai=[1e-2, 0], mat2="air", device=device)
        x = torch.tensor([3.0], device=device)
        y = torch.tensor([0.0], device=device)
        sag_conic = s_conic.sag(x, y)
        sag_asph = s_asph.sag(x, y)
        assert not torch.allclose(sag_conic, sag_asph)

    def test_get_optimizer_params_count(self, device):
        """Check that optimizer param groups match expected count."""
        s = Aspheric(r=2.0, d=0.0, c=0.1, k=0.0, ai=[0.0, 0.0, 0.0], mat2="air", device=device)
        params = s.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-2, 1e-4])
        # d + c + k + 3 ai coefficients = 6 groups
        assert len(params) == 6

    def test_init_from_dict(self):
        d = {
            "r": 3.0, "d": 1.0, "c": 0.2, "k": -1.0,
            "ai": [1e-3, -1e-5], "use_ai2": False, "mat2": "air",
        }
        s = Aspheric.init_from_dict(d)
        assert s.ai_degree == 2
        assert abs(s.c.item() - 0.2) < 1e-8

    def test_surf_dict_roundtrip(self, device):
        s = Aspheric(r=3.0, d=1.0, c=0.2, k=-1.5, ai=[1e-3, -2e-5, 3e-7], mat2="n-bk7", device=device)
        d = s.surf_dict()
        assert d["type"] == "Aspheric"
        assert abs(d["k"] - (-1.5)) < 1e-6
        assert len(d["ai"]) == 3


class TestAperture:
    def test_aperture_clips_rays(self, device):
        """Rays outside the aperture should be marked invalid."""
        from deeplens import Ray

        aper = Aperture(r=1.0, d=5.0, device=device)
        # Create rays: one inside, one outside
        o = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], device=device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(2, 3)
        ray = Ray(o, d, device=device)
        ray = aper.ray_reaction(ray)
        # The first ray should be valid, the second should be clipped
        assert ray.is_valid[0].item() == 1.0
        assert ray.is_valid[1].item() == 0.0
