"""Shared fixtures for AutoLens test suite."""

import os
import sys
import tempfile

import pytest
import torch

# Ensure deeplens is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def device():
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cellphone_path():
    """Path to the test lens fixture (aper_idx > 0 required for pupil calc)."""
    return os.path.join(
        os.path.dirname(__file__), "..", "datasets", "lens_zoo", "test_camera_aspheric.json"
    )


@pytest.fixture(scope="session")
def cellphone_lens(cellphone_path, device):
    """Load the cellphone lens once for the entire test session."""
    from deeplens import GeoLens

    lens = GeoLens(filename=cellphone_path)
    lens.to(device)
    return lens


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d
