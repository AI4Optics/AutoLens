"""Tests for GeoLens ray tracing, focal length, and pupil calculation."""

import math

import pytest
import torch

from deeplens import GeoLens


class TestRayTracing:
    def test_trace_on_axis(self, cellphone_lens, device):
        """On-axis rays should trace through the lens and reach the sensor."""
        ray = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=0.0)
        ray, _ = cellphone_lens.trace(ray)
        assert ray.is_valid.sum() > 0, "No valid rays after tracing"

    def test_trace_off_axis(self, cellphone_lens, device):
        """Off-axis rays should also trace through."""
        ray = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=10.0)
        ray, _ = cellphone_lens.trace(ray)
        assert ray.is_valid.sum() > 0

    def test_trace2sensor(self, cellphone_lens, device):
        """Rays should be propagated to the sensor plane."""
        ray = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=0.0)
        ray = cellphone_lens.trace2sensor(ray)
        d_sensor = cellphone_lens.d_sensor.item()
        # Valid rays should be at sensor z
        valid = ray.is_valid > 0
        if valid.any():
            z_vals = ray.o[valid, 2]
            assert torch.allclose(
                z_vals, torch.tensor(d_sensor, device=device).expand_as(z_vals), atol=1e-3
            )

    def test_trace_preserves_wavelength(self, cellphone_lens, device):
        """Ray wavelength should not change during tracing."""
        wvln = 0.55
        ray = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=0.0, wvln=wvln)
        ray = cellphone_lens.trace2sensor(ray)
        assert abs(ray.wvln.item() - wvln) < 1e-6

    def test_vignetting_increases_with_field(self, cellphone_lens, device):
        """More rays should be vignetted at larger field angles."""
        ray_center = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=0.0, num_rays=512)
        ray_center = cellphone_lens.trace2sensor(ray_center)
        valid_center = ray_center.is_valid.sum().item()

        ray_edge = cellphone_lens.sample_from_fov(fov_x=0.0, fov_y=25.0, num_rays=512)
        ray_edge = cellphone_lens.trace2sensor(ray_edge)
        valid_edge = ray_edge.is_valid.sum().item()

        assert valid_center >= valid_edge


class TestFocalLength:
    def test_calc_foclen_returns_positive(self, cellphone_lens, device):
        efl = cellphone_lens.calc_foclen()
        assert efl > 0

    def test_calc_foclen_close_to_stored(self, cellphone_lens, device):
        """Computed focal length should be close to the stored value."""
        efl = cellphone_lens.calc_foclen()
        stored = cellphone_lens.foclen
        # Allow 10% tolerance due to aberration
        assert abs(efl - stored) / stored < 0.10

    def test_calc_foclen_custom_angle(self, cellphone_lens, device):
        """Custom test angle should still give a reasonable result."""
        efl = cellphone_lens.calc_foclen(test_fov_deg=0.5)
        assert efl > 0
        assert not math.isnan(efl)

    def test_calc_foclen_updates_attributes(self, cellphone_lens, device):
        efl = cellphone_lens.calc_foclen()
        assert cellphone_lens.efl == efl
        assert cellphone_lens.foclen == efl
        assert cellphone_lens.bfl is not None


class TestPupil:
    def test_calc_pupil(self, cellphone_lens, device):
        """Pupil calculation should set entrance pupil attributes."""
        cellphone_lens.calc_pupil()
        assert cellphone_lens.entr_pupilr > 0

    def test_entrance_pupil_finite(self, cellphone_lens, device):
        """Entrance pupil position should be finite."""
        cellphone_lens.calc_pupil()
        assert cellphone_lens.entr_pupilz is not None
        assert abs(cellphone_lens.entr_pupilz) < 1e6
