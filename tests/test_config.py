"""Tests for deeplens/config.py constants."""

import pytest
from deeplens.config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    WAVE_RGB,
    WVLN_C,
    WVLN_F,
    WVLN_d,
    SPP_PSF,
    PSF_KS,
    GEO_GRID,
)


class TestWavelengthConstants:
    def test_wave_rgb_has_three_elements(self):
        assert len(WAVE_RGB) == 3

    def test_wave_rgb_in_valid_range(self):
        for wv in WAVE_RGB:
            assert 0.1 < wv < 10.0, f"Wavelength {wv} out of range [0.1, 10] um"

    def test_default_wave_positive(self):
        assert DEFAULT_WAVE > 0

    def test_fraunhofer_lines_ordered(self):
        # C (red) > d (yellow) > F (blue)
        assert WVLN_C > WVLN_d > WVLN_F


class TestNumericalConstants:
    def test_epsilon_positive_and_tiny(self):
        assert 0 < EPSILON < 1e-6

    def test_depth_negative(self):
        # Depth represents object at -infinity
        assert DEPTH < 0

    def test_spp_psf_positive(self):
        assert SPP_PSF > 0

    def test_psf_ks_positive(self):
        assert PSF_KS > 0

    def test_geo_grid_positive(self):
        assert GEO_GRID > 0
