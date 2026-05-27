"""Tests for GeoLens evaluation: RMS, spot diagram, analysis."""

import os

import pytest
import torch

from deeplens import GeoLens
from deeplens.config import WAVE_RGB


class TestRMS:
    def test_rms_map_returns_data(self, cellphone_lens, device):
        """rms_map should return valid data."""
        result = cellphone_lens.rms_map(num_grid=5)
        # rms_map may return a tuple (rms, fields) or a tensor
        if isinstance(result, tuple):
            rms = result[0]
        else:
            rms = result
        assert rms.numel() > 0
        assert torch.isfinite(rms).all()


class TestSpotDiagram:
    def test_spot_points_returns_data(self, cellphone_lens, device):
        """spot_points should return ray positions for given fields."""
        from deeplens.config import DEPTH

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        result = cellphone_lens.spot_points(points, num_rays=256)
        assert result is not None


class TestAnalysis:
    def test_analysis_saves_files(self, cellphone_path, tmp_dir, device):
        """analysis() should create output files."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        save_name = os.path.join(tmp_dir, "test_analysis")
        lens.analysis(save_name=save_name, render=False)

        # Check that at least one output file was created
        files = os.listdir(tmp_dir)
        assert len(files) > 0

    def test_analysis_no_crash(self, cellphone_lens, device):
        """analysis() should run without errors."""
        # Run without saving
        cellphone_lens.analysis(render=False)
