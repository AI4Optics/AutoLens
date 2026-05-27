"""Tests for deeplens/material/ — Material class and catalog."""

import pytest
import torch

from deeplens.material import Material, MATERIAL_data


class TestMaterialCatalog:
    def test_catalog_not_empty(self):
        assert len(MATERIAL_data) > 0

    def test_common_glasses_present(self):
        common = ["n-bk7", "n-sf11", "n-lak14", "pmma", "coc"]
        for glass in common:
            assert glass in MATERIAL_data, f"{glass} not in catalog"

    def test_catalog_entry_has_sellmeier_coefficients(self):
        entry = MATERIAL_data["n-bk7"]
        for key in ["a_coeff", "b_coeff", "c_coeff", "d_coeff", "e_coeff", "f_coeff"]:
            assert key in entry


class TestMaterialRefractive:
    def test_air_refractive_index(self, device):
        mat = Material("air")
        mat.to(device)
        n = mat.refractive_index(torch.tensor(0.587, device=device))
        assert abs(n.item() - 1.0) < 1e-6

    def test_glass_refractive_index_greater_than_one(self, device):
        mat = Material("n-bk7")
        mat.to(device)
        n = mat.refractive_index(torch.tensor(0.587, device=device))
        assert n.item() > 1.0

    def test_refractive_index_dispersion(self, device):
        """Blue light should have higher refractive index than red (normal dispersion)."""
        mat = Material("n-bk7")
        mat.to(device)
        n_red = mat.refractive_index(torch.tensor(0.656, device=device))
        n_blue = mat.refractive_index(torch.tensor(0.486, device=device))
        assert n_blue.item() > n_red.item()

    def test_multiple_wavelengths(self, device):
        mat = Material("n-bk7")
        mat.to(device)
        wvlns = [0.486, 0.587, 0.656]
        for wv in wvlns:
            n = mat.refractive_index(torch.tensor(wv, device=device))
            assert 1.0 < n.item() < 3.0

    def test_material_name(self):
        mat = Material("n-bk7")
        assert mat.get_name() == "n-bk7"

    def test_air_material_name(self):
        mat = Material("air")
        assert mat.get_name() == "air"
