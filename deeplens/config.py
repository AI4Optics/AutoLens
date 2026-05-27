"""Optics configuration constants and utilities for DeepLens."""

import numpy as np

# =============================================================================
# TUNABLE — experiment knobs, safe to adjust per run/config
# =============================================================================
DEPTH = -20000.0  # approximate infinity (object distance)
DEFAULT_WAVE = 0.587  # [um] default wavelength

SPP_PSF = 2 << 13      # 16384, spp for psf calculation
SPP_COHERENT = 2 << 23  # 8388608, spp for coherent/OPL ray tracing
SPP_CALC = 1024         # spp for approximate computation, e.g., refocusing
SPP_RENDER = 32  # spp for rendering
SPP_PARAXIAL = 32  # spp for paraxial

PSF_KS = 64  # kernel size for psf calculation
GEO_GRID = 16  # grid number for PSF map


# =============================================================================
# DO NOT TOUCH — physical constants and numerical tolerances
# =============================================================================
# Numerical tolerances
DELTA = 1e-6
DELTA_PARAXIAL = 0.01
EPSILON = 1e-12  # replace 0 with EPSILON in some cases

# Standard RGB wavelengths [um]
WAVE_RGB = [0.656, 0.587, 0.486]  # R, G, B

# Fraunhofer wavelengths [um] — standard reference lines for chromatic aberration
WVLN_d = 0.5876
WVLN_F = 0.4861
WVLN_C = 0.6563

# Narrow-band spectra [um]
WAVE_RED = [0.620, 0.660, 0.700]
WAVE_GREEN = [0.500, 0.530, 0.560]
WAVE_BLUE = [0.450, 0.470, 0.490]

# Full visible spectrum and hyperspectral bands
FULL_SPECTRUM = np.arange(0.400, 0.701, 0.02)
HYPER_SPEC_RANGE = [0.42, 0.66]  # [um], reference 400nm to 700nm, 20nm step size
HYPER_SPEC_BAND = 49  # 5nm/step, per "Shift-variant color-coded diffractive spectral imaging system"
