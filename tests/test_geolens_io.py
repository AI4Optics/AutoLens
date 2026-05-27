"""Tests for GeoLens I/O: JSON read/write roundtrip."""

import json
import os

import pytest
import torch

from deeplens import GeoLens


class TestJsonIO:
    def test_load_cellphone(self, cellphone_lens):
        """Cellphone lens should load successfully with surfaces."""
        assert len(cellphone_lens.surfaces) > 0
        assert cellphone_lens.foclen is not None

    def test_surface_count(self, cellphone_lens):
        """Cellphone lens should have the expected number of surfaces."""
        assert len(cellphone_lens.surfaces) >= 10

    def test_json_roundtrip(self, cellphone_path, tmp_dir, device):
        """Write and re-read a lens; parameters should match."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)

        out_path = os.path.join(tmp_dir, "roundtrip.json")
        lens.write_lens_json(out_path)

        lens2 = GeoLens(filename=out_path)
        lens2.to(device)

        assert len(lens.surfaces) == len(lens2.surfaces)
        assert abs(lens.foclen - lens2.foclen) < 0.01
        assert abs(lens.fnum - lens2.fnum) < 0.01

    def test_json_roundtrip_surface_params(self, cellphone_path, tmp_dir, device):
        """Individual surface parameters should survive roundtrip."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)

        out_path = os.path.join(tmp_dir, "roundtrip2.json")
        lens.write_lens_json(out_path)

        with open(cellphone_path) as f:
            orig = json.load(f)
        with open(out_path) as f:
            saved = json.load(f)

        for i, (s_orig, s_saved) in enumerate(
            zip(orig["surfaces"], saved["surfaces"])
        ):
            assert s_orig["type"] == s_saved["type"], f"Surface {i} type mismatch"

    def test_sensor_parameters_preserved(self, cellphone_path, tmp_dir, device):
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        original_d_sensor = lens.d_sensor.item()
        original_r_sensor = lens.r_sensor

        out_path = os.path.join(tmp_dir, "sensor_test.json")
        lens.write_lens_json(out_path)
        lens2 = GeoLens(filename=out_path)
        lens2.to(device)

        assert abs(lens2.d_sensor.item() - original_d_sensor) < 1e-4
        assert abs(lens2.r_sensor - original_r_sensor) < 1e-4


class TestLensAttributes:
    def test_foclen_positive(self, cellphone_lens):
        assert cellphone_lens.foclen > 0

    def test_fnum_positive(self, cellphone_lens):
        assert cellphone_lens.fnum > 0

    def test_d_sensor_positive(self, cellphone_lens):
        assert cellphone_lens.d_sensor.item() > 0

    def test_has_aperture(self, cellphone_lens):
        """Lens should have a detected aperture index."""
        assert cellphone_lens.aper_idx is not None
