"""Automated lens design from scratch using RMS spot size with curriculum learning.

Technical Paper:
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab
    initio deep learned refractive optics," Nature Communications 2024.

Usage:
    python 1_autolens.py
"""

import json
import logging
import math
import os
import random
import string
from datetime import datetime

import torch
from tqdm import tqdm

from deeplens import GeoLens
from deeplens.geolens_pkg import create_lens
from deeplens.utils_deeplens import (
    create_video_from_images,
    record_losses,
    save_loss_curve,
    set_logger,
    set_seed,
)


# ── Default lens design configuration ────────────────────────────────────────

CAMERA_CONFIG = {
    "foclen": 85.0,
    "fov": 40.0,
    "fnum": 2.8,
    "bfl": 18.0,
    "thickness": 120.0,
    "surf_list": [
        ["Spheric", "Spheric"],
        ["Spheric", "Spheric"],
        ["Spheric", "Spheric", "Spheric"],
        ["Aperture"],
        ["Spheric", "Spheric", "Spheric"],
        ["Spheric", "Aspheric"],
        ["Spheric", "Aspheric"],
    ],
    "lrs": [1e-3, 1e-3, 1e-2, 1e-4],
    "curriculum_iters": 2000,
    "refine_iters": 3000,
    "seed": 42,
}

MOBILE_CONFIG = {
    "foclen": 5.0,
    "fov": 90.0,
    "fnum": 1.8,
    "bfl": 2.0,
    "thickness": 10.0,
    "surf_list": [
        ["Aperture"],
        ["Aspheric", "Aspheric"],
        ["Aspheric", "Aspheric"],
        ["Aspheric", "Aspheric"],
        ["Aspheric", "Aspheric"],
        ["Aspheric", "Aspheric"],
        ["Aspheric", "Aspheric"],
    ],
    "lrs": [1e-4, 1e-3, 1e-2, 1e-4],
    "curriculum_iters": 2000,
    "refine_iters": 3000,
    "seed": 42,
}


# ── Curriculum design ─────────────────────────────────────────────────────────

def curriculum_design(
    self,
    lrs=[1e-3, 1e-3, 1e-2, 1e-4],
    iterations=5000,
    test_per_iter=100,
    optim_mat=False,
    shape_control=True,
    centroid=True,
    depth=None,
    wvln_list=None,
    num_ring=8,
    num_arm=1,
    spp=512,
    result_dir="./results",
):
    """Optimise the lens from scratch using curriculum aperture growth.

    Gradually increases the aperture from 25% to full size over the
    training schedule, transforming a hard global optimisation into a
    sequence of easier subproblems.

    Args:
        lrs (list, optional): Learning rates for [d, c, k, ai].
        iterations (int, optional): Total training iterations.
        test_per_iter (int, optional): Evaluate and save every N iterations.
        optim_mat (bool, optional): Optimise material parameters.
        shape_control (bool, optional): Correct surface shapes at each evaluation.
        centroid (bool, optional): Use geometric centroid as the PSF reference
            centre (True) or pinhole ideal (False). Defaults to True.
        depth (float, optional): Object distance in mm. Defaults to DEPTH.
        wvln_list (list[float], optional): Wavelengths in micrometers for
            the polychromatic RMS loss. Defaults to the lens RGB
            wavelengths.
        num_ring (int, optional): Number of radial rings in the ring-arm
            field grid. Defaults to 8.
        num_arm (int, optional): Number of azimuthal arms. Defaults to 1.
        spp (int, optional): Rays per field point per wavelength.
            Defaults to 1024.
        result_dir (str, optional): Directory to save results.
    """
    depth = self.obj_depth if depth is None else depth
    wvln_list = self.wvln_ls if wvln_list is None else wvln_list
    assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths"

    aper_start = self.surfaces[self.aper_idx].r * 0.5
    aper_final = self.surfaces[self.aper_idx].r

    os.makedirs(result_dir, exist_ok=True)
    if not logging.getLogger().hasHandlers():
        logger = logging.getLogger()
        logger.setLevel("DEBUG")
        fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel("INFO")
        fh = logging.FileHandler(f"{result_dir}/output.log")
        fh.setFormatter(fmt)
        fh.setLevel("INFO")
        logger.addHandler(sh)
        logger.addHandler(fh)
    logging.info(
        f"lr:{lrs}, iterations:{iterations}, spp:{spp}, "
        f"num_ring:{num_ring}, num_arm:{num_arm}."
    )

    optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=iterations // 4, T_mult=1
    )

    pbar = tqdm(
        total=iterations + 1, desc="Curriculum", postfix={"loss_rms": 0, "loss_reg": 0}
    )
    loss_history: dict = {}
    for i in range(iterations + 1):
        # === Evaluate ===
        if i % test_per_iter == 0:
            with torch.no_grad():
                progress = 0.5 * (1 + math.cos(math.pi * (1 - i / iterations)))
                aper_r = min(
                    aper_start + (aper_final - aper_start) * progress,
                    aper_final,
                )
                self.surfaces[self.aper_idx].update_r(aper_r)
                self.calc_pupil()

                if i > 0:
                    if shape_control:
                        self.correct_shape()

                self.write_lens_json(f"{result_dir}/iter{i}.json")
                self.analysis(f"{result_dir}/iter{i}")
                save_loss_curve(loss_history, result_dir)

                rays_backup = []
                for wv in wvln_list:
                    ray = self.sample_ring_arm_rays(
                        num_ring=num_ring,
                        num_arm=num_arm,
                        depth=depth,
                        spp=spp,
                        wvln=wv,
                        scale_pupil=1.05,
                    )
                    rays_backup.append(ray)

                pinhole_ref = -self.psf_center(
                    points_obj=ray.o[:, :, 0, :], method="pinhole"
                )

        # === Compute RMS loss ===
        loss_rms, loss_distortion = self.loss_rms_from_ray(
            rays_backup, pinhole_ref, centroid=centroid
        )

        w_reg = 0.1
        # Activate the material (n, Abbe V) constraint only while materials
        # are being optimised; with frozen materials the gradient is zero
        # and w_mat would just add noise to the logged loss.
        loss_reg, loss_dict = self.loss_reg(w_mat=1.0 if optim_mat else 0.0)

        L_total = loss_rms + w_reg * (loss_reg + loss_distortion)

        optimizer.zero_grad()
        L_total.backward()
        self._sanitize_optimizer_grads(optimizer)
        optimizer.step()
        scheduler.step()

        record_losses(loss_history, i, {
            "loss_rms": loss_rms.item(),
            "loss_reg": loss_reg.item(),
            "loss_distortion": loss_distortion.item(),
            **loss_dict,
        })
        pbar.set_postfix(
            loss_rms=loss_rms.item(),
            loss_reg=loss_reg.item(),
            loss_distortion=loss_distortion.item(),
            **loss_dict,
        )
        pbar.update(1)

    save_loss_curve(loss_history, result_dir)
    pbar.close()


GeoLens.curriculum_design = curriculum_design


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = MOBILE_CONFIG.copy()

    # Setup
    seed = cfg["seed"] if cfg["seed"] is not None else random.randint(0, 100000)
    set_seed(seed)

    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-AutoLens-{tag}"
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Save config
    with open(f"{result_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Create lens
    logging.info(
        f"==> Design target: foclen={cfg['foclen']}, FoV={cfg['fov']}deg, F/{cfg['fnum']}"
    )
    lens = create_lens(
        foclen=cfg["foclen"],
        fov=cfg["fov"],
        fnum=cfg["fnum"],
        bfl=cfg["bfl"],
        thickness=cfg["thickness"],
        surf_list=cfg["surf_list"],
        save_dir=result_dir,
    )
    lens.set_target_fov_fnum(rfov_eff=cfg["fov"] / 2 / 57.3, fnum=cfg["fnum"])

    # Stage 1: Curriculum learning (optimize material, then match to catalog)
    curriculum_iters = cfg["curriculum_iters"]
    logging.info(f"==> Stage 1: Curriculum learning ({curriculum_iters} iters)")
    lens.curriculum_design(
        lrs=[float(lr) for lr in cfg["lrs"]],
        iterations=curriculum_iters,
        test_per_iter=max(curriculum_iters // 50, 1),
        optim_mat=True,
        shape_control=True,
        centroid=True,
        result_dir=result_dir,
    )

    lens.match_materials()
    lens.set_fnum(cfg["fnum"])
    lens.write_lens_json(f"{result_dir}/curriculum_final.json")
    lens.analysis(save_name=f"{result_dir}/curriculum_final.png")

    # Stage 2: Fine-tuning with centroid PSF center
    # Centroid-based RMS only helps at wide FoV where off-axis PSFs are
    # asymmetric enough that the geometric centroid diverges from the chief
    # ray. Threshold ~30 deg half-diagonal (0.52 rad).
    use_centroid = lens.rfov_eff > 0.52
    refine_iters = cfg["refine_iters"]
    logging.info(
        f"==> Stage 2: Fine-tuning centroid={use_centroid} ({refine_iters} iters)"
    )
    lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
    lens.optimize(
        lrs=[0.5 * float(lr) for lr in cfg["lrs"]],
        iterations=refine_iters,
        test_per_iter=max(refine_iters // 50, 1),
        centroid=use_centroid,
        optim_mat=False,
        shape_control=True,
        sample_more_off_axis=True,
        result_dir=f"{result_dir}/refine",
    )

    # Final result
    lens.prune_surf(expand_factor=0.05)
    lens.post_computation()
    logging.info(f"Final: FoV={lens.rfov_eff:.4f} rad, F/{lens.fnum:.2f}, r_sensor={lens.r_sensor:.4f}")
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)
