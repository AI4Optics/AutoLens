"""Tests for deeplens/diff_ops.py and deeplens/utils_deeplens.py."""

import pytest
import torch

from deeplens.diff_ops import interp1d


class TestInterp1d:
    def test_exact_key_points(self, device):
        """Interpolation at key points should return exact values."""
        key = torch.tensor([0.0, 1.0, 2.0], device=device).unsqueeze(-1)
        value = torch.tensor([0.0, 10.0, 20.0], device=device).unsqueeze(-1)
        query = torch.tensor([0.0, 1.0, 2.0], device=device).unsqueeze(-1)
        result = interp1d(query, key, value)
        expected = torch.tensor([0.0, 10.0, 20.0], device=device).unsqueeze(-1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_midpoint_interpolation(self, device):
        """Midpoint should give average of neighbors."""
        key = torch.tensor([0.0, 2.0], device=device).unsqueeze(-1)
        value = torch.tensor([0.0, 10.0], device=device).unsqueeze(-1)
        query = torch.tensor([1.0], device=device).unsqueeze(-1)
        result = interp1d(query, key, value)
        assert torch.allclose(result, torch.tensor([[5.0]], device=device), atol=1e-5)

    def test_monotone_output(self, device):
        """For monotone key-value pairs, interpolation should be monotone."""
        key = torch.linspace(0, 10, 11, device=device).unsqueeze(-1)
        value = key ** 2
        query = torch.linspace(0, 10, 21, device=device).unsqueeze(-1)
        result = interp1d(query, key, value)
        diffs = result[1:] - result[:-1]
        assert (diffs >= -1e-6).all()


class TestSetSeed:
    def test_reproducibility(self):
        from deeplens.utils_deeplens import set_seed

        set_seed(42)
        a = torch.rand(10)
        set_seed(42)
        b = torch.rand(10)
        assert torch.allclose(a, b)
