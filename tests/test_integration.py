"""Integration tests: end-to-end workflows."""

import os

import pytest
import torch

from deeplens import GeoLens
from deeplens.config import WAVE_RGB
from deeplens.geolens_pkg.optim_init import DEFAULT_COVER_GLASS_SENSOR_GAP, create_lens
from deeplens.geometric_surface import Plane


class TestCreateLens:
    def test_create_basic_lens(self, tmp_dir, device):
        """create_lens should produce a valid lens from scratch."""
        lens = create_lens(
            fov=50.0,
            fnum=2.8,
            bfl=3.0,
            foclen=5.0,
            save_dir=tmp_dir,
        )
        lens.to(device)
        assert len(lens.surfaces) > 0
        assert lens.d_sensor.item() > 0
        assert not isinstance(lens.surfaces[-1], Plane)
        assert len(lens.surfaces) - 1 in lens.find_diff_surf()

    def test_create_lens_with_cover_glass(self, tmp_dir, device):
        """create_lens should optionally add fixed sensor cover glass."""
        lens = create_lens(
            fov=50.0,
            fnum=2.8,
            bfl=3.0,
            foclen=5.0,
            save_dir=tmp_dir,
            add_cover_glass=True,
        )
        lens.to(device)
        assert isinstance(lens.surfaces[-2], Plane)
        assert lens.surfaces[-2].mat2.get_name() == "bk7"
        assert isinstance(lens.surfaces[-1], Plane)
        assert lens.surfaces[-1].mat2.get_name() == "air"
        assert lens.d_sensor.item() - lens.surfaces[-1].d.item() == pytest.approx(
            DEFAULT_COVER_GLASS_SENSOR_GAP
        )
        diff_surfs = lens.find_diff_surf()
        assert len(lens.surfaces) - 2 not in diff_surfs
        assert len(lens.surfaces) - 1 not in diff_surfs
        params = lens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-2, 1e-4])
        param_ids = set()
        for group in params:
            group_params = group["params"]
            if torch.is_tensor(group_params):
                group_params = [group_params]
            param_ids.update(id(param) for param in group_params)
        assert id(lens.surfaces[-2].d) not in param_ids
        assert id(lens.surfaces[-1].d) not in param_ids

    def test_create_lens_with_imgh(self, tmp_dir, device):
        """create_lens should accept imgh instead of foclen."""
        lens = create_lens(
            fov=60.0,
            fnum=2.0,
            bfl=2.5,
            imgh=5.0,
            save_dir=tmp_dir,
        )
        lens.to(device)
        assert len(lens.surfaces) > 0

    def test_create_lens_mutually_exclusive(self, tmp_dir):
        """Specifying both foclen and imgh should raise ValueError."""
        with pytest.raises(ValueError):
            create_lens(fov=50.0, fnum=2.0, bfl=3.0, foclen=5.0, imgh=4.0, save_dir=tmp_dir)

    def test_create_lens_traceable(self, tmp_dir, device):
        """A created lens should be traceable."""
        lens = create_lens(fov=40.0, fnum=2.8, bfl=3.0, foclen=4.0, save_dir=tmp_dir)
        lens.to(device)
        ray = lens.sample_from_fov(fov_x=0.0, fov_y=0.0)
        ray = lens.trace2sensor(ray)
        assert ray.is_valid.sum() > 0


class TestEndToEnd:
    def test_load_trace_evaluate(self, cellphone_path, device):
        """Full pipeline: load → trace → evaluate RMS."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)

        # Trace
        ray = lens.sample_from_fov(fov_x=0.0, fov_y=0.0, num_rays=512)
        ray = lens.trace2sensor(ray)
        assert ray.is_valid.sum() > 0

        # RMS
        rms = ray.rms_error()
        assert torch.isfinite(rms)
        assert rms.item() >= 0

    def test_load_compute_psf_evaluate(self, cellphone_path, device):
        """Load → PSF → check output."""
        from deeplens.config import DEPTH

        lens = GeoLens(filename=cellphone_path)
        lens.to(device)

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        psf = lens.psf(points, ks=16, spp=256, model="geometric")
        assert psf.shape[-1] == 16
        assert (psf >= 0).all()

    def test_optimization_workflow(self, cellphone_path, tmp_dir, device):
        """Load → init constraints → short optimize → save."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        lens.init_constraints()

        optimizer = lens.get_optimizer(lrs=[0, 1e-4, 1e-2, 1e-4])
        for _ in range(3):
            loss = lens.loss_infocus()
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        out_path = os.path.join(tmp_dir, "optimized.json")
        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        # Reload and verify
        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == len(lens.surfaces)


class TestDeviceTransfer:
    def test_to_device(self, cellphone_path, device):
        """Lens should transfer to device without errors."""
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        # Check a surface parameter is on the right device
        assert lens.surfaces[1].c.device.type == device.type

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_trace(self, cellphone_path):
        """Ray tracing should work on GPU."""
        device = torch.device("cuda")
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        ray = lens.sample_from_fov(fov_x=0.0, fov_y=0.0)
        ray = lens.trace2sensor(ray)
        assert ray.o.device.type == "cuda"
        assert ray.is_valid.sum() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_psf(self, cellphone_path):
        """PSF computation should work on GPU."""
        device = torch.device("cuda")
        lens = GeoLens(filename=cellphone_path)
        lens.to(device)
        from deeplens.config import DEPTH

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device)
        psf = lens.psf(points, ks=16, spp=256, model="geometric")
        assert psf.device.type == "cuda"


class TestDtype:
    def test_float64_trace(self, cellphone_path, device):
        """Lens should work in float64 precision."""
        lens = GeoLens(filename=cellphone_path)
        lens.astype(torch.float64)
        lens.to(device)
        ray = lens.sample_from_fov(fov_x=0.0, fov_y=0.0)
        ray = lens.trace2sensor(ray)
        assert ray.is_valid.sum() > 0
