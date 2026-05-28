# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Optimization and constraint functions for GeoLens.

Differentiable lens design has several advantages over conventional lens design:
    1. AutoDiff gradient calculation is faster and numerically more stable, which is important for complex optical systems.
    2. First-order optimization with momentum (e.g., Adam) is typically more stable than second-order optimization, and also has promising convergence speed.
    3. Efficient definition of loss functions can prevent the lens from violating constraints.

References:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.

Functions:
    - init_constraints: Initialize constraints for the lens design
    - loss_reg: An empirical regularization loss for lens design
    - loss_infocus: Sample parallel rays and compute RMS loss on the sensor plane
    - loss_profile: Penalize infeasible per-surface profile shape (sag, slope)
    - loss_bound: Single-pass geometry-bound penalty returning
      (loss_clearance, loss_envelope) — min-side (self-intersection,
      min thickness/BFL/TTL) and max-side (air gap, thickness, BFL, TTL caps)
    - loss_cra: Loss function to penalize large chief ray angle
    - loss_ray_bend: Loss function to penalize large per-surface ray bends
    - sample_ring_arm_rays: Sample rays from object space using a ring-arm pattern
    - optimize: Optimize the lens by minimizing rms errors
"""

import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import relu
from tqdm import tqdm


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then half-cosine decay to 10% of the base LR."""
    min_lr_ratio = 0.1

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

from ..config import (
    EPSILON,
    SPP_CALC,
)
from ..geometric_surface import Aperture, Aspheric, Plane, Spheric, ThinLens
from ..phase_surface import Phase
from ..utils_deeplens import record_losses, save_loss_curve

logger = logging.getLogger(__name__)


class GeoLensOptim:
    """Mixin providing differentiable optimisation for ``GeoLens``.

    Implements gradient-based lens design using PyTorch autograd:

    * **Loss functions** – RMS spot error, focus, surface regularity, gap
      constraints, material validity.
    * **Constraint initialisation** – edge-thickness and self-intersection
      guards.
    * **Optimizer helpers** – parameter groups with per-type learning rates
      and cosine annealing schedules.
    * **High-level ``optimize()``** – curriculum-learning training loop.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.

    References:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," *Nature Communications* 2024.
    """

    # ================================================================
    # Lens design constraints
    # ================================================================
    def init_constraints(self, constraint_params=None):
        """Initialize constraints for the lens design.
        
        Args:
            constraint_params (dict): Constraint parameters.
        """
        if constraint_params is None:
            constraint_params = {}
            print("Lens design constraints initialized with default values.")

        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.air_edge_min = 0.05
            self.air_edge_max = 5.0
            self.air_center_min = 0.05
            self.air_center_max = 5.0
            
            self.thick_edge_min = 0.4
            self.thick_edge_max = 2.0
            self.thick_center_min = 0.4
            self.thick_center_max = 2.0
            
            self.bfl_min = 1.0
            self.bfl_max = 5.0
            self.ttl_min = 5.0
            self.ttl_max = 15.0

            # Surface shape constraints
            self.sag2diam_max = 0.20
            self.diam2thick_max = 15.0
            self.tmax2tmin_max = 5.0
            self.surf_angle_max = 30.0  # deg
            self.sensor_cra_cap = 40.0  # deg
            self.bend_angle_max = 20.0  # deg

            # Distortion constraints
            self.distortion_max = 0.03

        else:
            self.is_cellphone = False

            self.air_edge_min = 0.1
            self.air_edge_max = 20.0
            self.air_center_min = 0.1
            self.air_center_max = 20.0

            self.thick_edge_min = 1.0
            self.thick_edge_max = 20.0
            self.thick_center_min = 2.0
            self.thick_center_max = 20.0

            self.bfl_min = 10.0
            self.bfl_max = 100.0
            self.ttl_min = 30.0
            self.ttl_max = 300.0

            # Surface shape constraints
            self.sag2diam_max = 0.50
            self.diam2thick_max = 20.0
            self.tmax2tmin_max = 10.0
            self.surf_angle_max = 45.0  # deg
            self.sensor_cra_cap = 20.0  # deg
            self.bend_angle_max = 30.0  # deg

            # Distortion constraints
            self.distortion_max = 0.03

        # Chief-ray-angle cap = min(half-FoV, sensor microlens cap)
        self.chief_ray_angle_max = min(
            float(np.degrees(self.rfov_eff)), float(self.sensor_cra_cap)
        )

        for surf in self.surfaces:
            surf.bend_angle_max = float(self.bend_angle_max)

    def loss_reg(
        self,
        w_focus=0.0,
        w_cra=1.0,
        w_ray_bend=1.0,
        w_clearance=1.0,
        w_envelope=1.0,
        w_profile=1.0,
        w_mat=0.0,
    ):
        """Compute combined regularization loss for lens design.

        Aggregates multiple constraint losses to keep the lens physically valid
        during gradient-based optimisation.

        Args:
            w_focus (float, optional): Weight for focus loss
                (:meth:`loss_infocus`). Set to 0 to skip the axial-RMS
                focus penalty entirely. Defaults to 0.0.
            w_cra (float, optional): Weight for chief ray angle loss.
                Set to 0 to skip the chief-ray-angle penalty.
                Defaults to 1.0.
            w_ray_bend (float, optional): Weight for per-surface ray bend loss.
                Set to 0 to skip the bend-angle penalty.
                Defaults to 1.0.
            w_clearance (float, optional): Weight for the clearance penalty
                (min air gap, min thickness, min BFL, min TTL). Defaults to 1.0.
            w_envelope (float, optional): Weight for the envelope penalty
                (max air gap, max thickness, max BFL, max TTL). Defaults to 1.0.
            w_profile (float, optional): Weight for per-surface profile
                feasibility (sag, slope). Defaults to 1.0.

        Returns:
            tuple: (loss_reg, loss_dict) where:
                - loss_reg (Tensor): Scalar combined regularization loss.
                - loss_dict (dict): Per-component loss values for logging.
        """
        # Only compute components whose weights are active, so callers can
        # disable expensive penalties by setting the corresponding weight to 0.
        zero = torch.tensor(0.0, device=self.device)

        if w_focus != 0:
            loss_focus = self.loss_infocus()
        else:
            loss_focus = zero

        if w_cra != 0:
            loss_cra = self.loss_cra()
        else:
            loss_cra = zero

        if w_ray_bend != 0:
            loss_ray_bend = self.loss_ray_bend()
        else:
            loss_ray_bend = zero

        if w_clearance != 0 or w_envelope != 0:
            loss_clearance, loss_envelope = self.loss_bound()
        else:
            loss_clearance = zero
            loss_envelope = zero

        if w_profile != 0:
            loss_profile = self.loss_profile()
        else:
            loss_profile = zero

        if w_mat != 0:
            loss_mat = self.loss_mat()
        else:
            loss_mat = zero

        loss_reg = (
            w_focus * loss_focus
            + w_clearance * loss_clearance
            + w_envelope * loss_envelope
            + w_profile * loss_profile
            + w_cra * loss_cra
            + w_ray_bend * loss_ray_bend
            + w_mat * loss_mat
        )

        # Return loss and loss dictionary
        loss_dict = {
            "loss_clearance": loss_clearance.item(),
            "loss_envelope": loss_envelope.item(),
            "loss_profile": loss_profile.item(),
            "loss_cra": loss_cra.item(),
            "loss_ray_bend": loss_ray_bend.item(),
            "loss_mat": loss_mat.item(),
        }
        return loss_reg, loss_dict

    def loss_infocus(self, target=0.005, wvln=None):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            target (float, optional): target of RMS loss. Defaults to 0.005 [mm].
            wvln (float, optional): Wavelength in um. Defaults to the middle
                configured RGB wavelength.
        """
        if wvln is None:
            wvln = self.wvln_ls[1]
        loss = torch.tensor(0.0, device=self.device)

        # Ray tracing and calculate RMS error
        ray = self.sample_from_fov(fov_x=0.0, fov_y=0.0, wvln=wvln, num_rays=SPP_CALC)
        ray = self.trace2sensor(ray)
        rms_error = ray.rms_error()

        # Smooth penalty: activates when rms_error exceeds target
        loss += relu(rms_error - target)

        return loss

    def loss_profile(self):
        """Penalize infeasible per-surface profile shapes.

        The "profile" is the z(r) curve of a single surface. This loss makes
        sure each surface is physically manufacturable by softly penalizing
        sampled profile points whose:
            1. Sag-to-diameter ratio approaches/exceeds ``sag2diam_max``.
            2. Surface slope angle approaches/exceeds ``surf_angle_max``.
            3. Diameter-to-thickness ratio exceeding ``diam2thick_max``
               (currently disabled).
            4. Maximum-to-minimum thickness ratio exceeding ``tmax2tmin_max``
               (currently disabled).

        Returns:
            Tensor: Scalar profile feasibility penalty.
        """
        num_points = 64

        sag2diam_max = self.sag2diam_max
        slope_max = float(np.tan(np.deg2rad(self.surf_angle_max)))
        diam2thick_max = self.diam2thick_max
        tmax2tmin_max = self.tmax2tmin_max

        loss_slope = torch.tensor(0.0, device=self.device)
        loss_diam2thick = torch.tensor(0.0, device=self.device)
        loss_tmax2tmin = torch.tensor(0.0, device=self.device)
        loss_sag2diam = torch.tensor(0.0, device=self.device)

        for i in self.find_diff_surf():
            # Sample points on the surface
            x_ls = torch.linspace(0.0, 1.0, num_points, device=self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag2Diam
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag2diam_ls = sag_ls.abs() / self.surfaces[i].r / 2
            sag2diam_violation = (sag2diam_ls - sag2diam_max) / sag2diam_max
            loss_sag2diam += relu(sag2diam_violation).mean()

            # Surface slope
            slope_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0].abs()
            slope_violation = (slope_ls - slope_max) / slope_max
            loss_slope += relu(slope_violation).mean()

            # # Diameter to thickness ratio, thick_max to thick_min ratio
            # if not self.surfaces[i].mat2.name == "air":
            #     surf2 = self.surfaces[i + 1]
            #     surf1 = self.surfaces[i]

            #     # Penalize diameter to thickness ratio
            #     diam2thick = 2 * max(surf2.r, surf1.r) / (surf2.d - surf1.d)
            #     loss_diam2thick += torch.nn.functional.relu(diam2thick - diam2thick_max)

            #     # Penalize thick_max to thick_min ratio.
            #     # Use torch.maximum/minimum for differentiable max/min.
            #     r_edge = min(surf2.r, surf1.r)
            #     thick_center = surf2.d - surf1.d
            #     thick_edge = surf2.surface_with_offset(r_edge, 0.0) - surf1.surface_with_offset(r_edge, 0.0)
            #     thick_max = torch.maximum(thick_center, thick_edge)
            #     thick_min = torch.minimum(thick_center, thick_edge).clamp(min=0.01)
            #     tmax2tmin = thick_max / thick_min

            #     loss_tmax2tmin += torch.nn.functional.relu(tmax2tmin - tmax2tmin_max)

        return loss_sag2diam + loss_slope + loss_diam2thick + loss_tmax2tmin

    def loss_bound(self):
        """Penalize geometry-bound violations in a single surface-sampling pass.

        Each surface pair is sampled once and its distances feed both the
        clearance and envelope softplus penalties for air gaps, glass
        thickness, BFL, and TTL. Sampled edge/BFL lists are penalized with a
        mean over all sampled points so gradients are dense across the profile
        rather than flowing only through the current worst point.

        Returns:
            tuple: ``(loss_clearance, loss_envelope)`` scalar tensors, so
                callers can weight them independently. Clearance penalizes
                parts that are too close / too thin, envelope penalizes the
                overall assembly growing beyond its spatial budget.
        """
        num_points = 64

        # Min bounds (clearance)
        air_center_min = self.air_center_min
        air_edge_min = self.air_edge_min
        thick_center_min = self.thick_center_min
        thick_edge_min = self.thick_edge_min
        bfl_min = self.bfl_min
        ttl_min = self.ttl_min

        # Max bounds (envelope)
        air_center_max = self.air_center_max
        air_edge_max = self.air_edge_max
        thick_center_max = self.thick_center_max
        thick_edge_max = self.thick_edge_max
        bfl_max = self.bfl_max
        ttl_max = self.ttl_max

        loss_clearance = torch.tensor(0.0, device=self.device)
        loss_envelope = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]

            # Sample surfaces once and reuse for both clearance and envelope
            r_center = torch.tensor(0.0, device=self.device)
            z_prev_center = current_surf.surface_with_offset(r_center, 0.0, valid_check=False)
            z_next_center = next_surf.surface_with_offset(r_center, 0.0, valid_check=False)

            r_edge = torch.linspace(0.5, 1.0, num_points, device=self.device) * current_surf.r
            z_prev_edge = current_surf.surface_with_offset(r_edge, 0.0, valid_check=False)
            z_next_edge = next_surf.surface_with_offset(r_edge, 0.0, valid_check=False)

            dist_center = z_next_center - z_prev_center
            dist_edges = z_next_edge - z_prev_edge

            if current_surf.mat2.name == "air":
                # Temporarily use 50.0
                loss_clearance += relu(air_center_min - dist_center)
                loss_clearance += relu(air_edge_min - dist_edges).mean()
                loss_envelope += relu(dist_center - air_center_max)
                loss_envelope += relu(dist_edges - air_edge_max).mean()
            else:
                loss_clearance += relu(thick_center_min - dist_center)
                loss_clearance += relu(thick_edge_min - dist_edges).mean()
                loss_envelope += relu(dist_center - thick_center_max)
                loss_envelope += relu(dist_edges - thick_edge_max).mean()

        # Back focal length: penalize the full sampled list.
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, num_points, device=self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        loss_clearance += relu(bfl_min - z_last_surf).mean()
        loss_envelope += relu(z_last_surf - bfl_max).mean()

        # Total track length. ttl_min may be 0 to disable the lower side;
        # only envelope is active then.
        ttl = self.d_sensor - self.surfaces[0].d
        loss_clearance += relu(ttl_min - ttl)
        loss_envelope += relu(ttl - ttl_max)

        return loss_clearance, loss_envelope

    def loss_cra(self):
        """Penalize chief ray angle violations at the sensor.

        Uses a near-paraxial pupil sample at full FoV and applies
        ``softplus(cos(CRA_max) - cos(CRA))`` scaled by the allowed cosine
        range. The loss rises smoothly as CRA approaches
        ``chief_ray_angle_max``.

        Returns:
            Tensor: Scalar chief-ray-angle penalty loss (always >= 0).
        """
        cos_cra_min = float(np.cos(np.deg2rad(self.chief_ray_angle_max)))
        cos_cra_scale = max(1.0 - cos_cra_min, EPSILON)
        ray = self.sample_ring_arm_rays(num_ring=16, num_arm=1, spp=SPP_CALC, scale_pupil=0.1)
        ray = self.trace2sensor(ray)
        cos_cra = ray.d[..., 2]
        valid = ray.is_valid > 0
        penalty_cra = relu((cos_cra_min - cos_cra) / cos_cra_scale)
        return (penalty_cra * valid).sum() / (valid.sum() + EPSILON)

    def loss_ray_bend(self):
        """Penalize accumulated per-surface ray bend violations.

        Reads ``ray.bend_penalty``, an additive sum of
        ``softplus(cos_gate - cos(bend_i))`` contributions collected during
        ``trace2sensor`` across every refraction. Each surface contributes
        independently, so large bends at one surface are not hidden by small
        bends at another.

        Returns:
            Tensor: Scalar per-surface bend penalty loss (always >= 0).
        """
        ray = self.sample_ring_arm_rays(num_ring=16, num_arm=1, spp=SPP_CALC, scale_pupil=1.0)
        ray = self.trace2sensor(ray)
        bend_penalty = ray.bend_penalty.squeeze(-1)
        valid = ray.is_valid > 0
        bend_violated = valid & (bend_penalty > EPSILON)
        return (bend_penalty * bend_violated).sum() / (
            bend_violated.sum() + EPSILON
        )

    def loss_mat(self):
        """Penalize material parameters outside manufacturable ranges.

        Constrains refractive index *n* to [1.5, 1.9] and Abbe number *V* to
        [30, 70] for each non-air surface material.

        Returns:
            Tensor: Scalar material penalty loss.
        """
        n_max = 1.9
        n_min = 1.5
        V_max = 70
        V_min = 30
        loss_mat = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.name != "air":
                if self.surfaces[i].mat2.n > n_max:
                    loss_mat += (self.surfaces[i].mat2.n - n_max) / (n_max - n_min)
                if self.surfaces[i].mat2.n < n_min:
                    loss_mat += (n_min - self.surfaces[i].mat2.n) / (n_max - n_min)
                if self.surfaces[i].mat2.V > V_max:
                    loss_mat += (self.surfaces[i].mat2.V - V_max) / (V_max - V_min)
                if self.surfaces[i].mat2.V < V_min:
                    loss_mat += (V_min - self.surfaces[i].mat2.V) / (V_max - V_min)
        
        return loss_mat

    # ================================================================
    # Loss functions
    # ================================================================
    def loss_distortion_from_ray(self, ray, pinhole_ref):
        """Compute weighted distortion loss from a traced green-channel ray.

        Args:
            ray: Traced Ray object (green channel).
            pinhole_ref: Ideal pinhole image positions, shape ``[..., 2]``.

        Returns:
            tuple: ``(loss_distortion, centroid_xy)`` where ``centroid_xy``
                has shape ``[..., 2]``.
        """
        centroid_xy = ray.centroid()[..., :2]
        ideal_height = pinhole_ref.norm(dim=-1)
        field_mask = ideal_height > EPSILON
        distortion = (centroid_xy - pinhole_ref).norm(dim=-1)
        distortion = distortion / ideal_height.clamp_min(EPSILON)
        violation = distortion - self.distortion_max
        penalty = relu(violation / self.distortion_max)
        field_rel = ideal_height / ideal_height.max().clamp_min(EPSILON)
        field_weight = field_mask.float() * (1.0 + field_rel**2)
        loss_distortion = (penalty * field_weight).sum() / (field_weight.sum() + EPSILON)
        return loss_distortion, centroid_xy

    def loss_rms_from_ray(self, rays_backup, pinhole_ref, centroid=True, use_weight_mask=True):
        """Compute RMS and distortion losses from pre-sampled ray backups.

        Green channel is traced first to anchor the centroid reference and the
        per-field weight mask; blue and red follow in that order.

        Args:
            rays_backup: List of 3 Ray objects ordered ``[B, G, R]``.
            pinhole_ref: Pinhole ideal image positions, shape ``[..., 2]``.
            centroid (bool): Use green centroid as RMS centre; else use pinhole.
            use_weight_mask (bool): Weight field points by green-channel RMS.

        Returns:
            tuple: ``(loss_rms, loss_distortion)`` scalar tensors.
        """
        loss_rms_ls = []
        loss_distortion = torch.tensor(0.0, device=self.device)
        w_mask = None
        center_ref = None
        wvln_order = [1, 0, 2]  # green first

        for wv_idx in wvln_order:
            ray = rays_backup[wv_idx].clone()
            ray = self.trace2sensor(ray)

            if center_ref is None:
                loss_distortion, centroid_xy = self.loss_distortion_from_ray(ray, pinhole_ref)
                center_ref = centroid_xy.detach().unsqueeze(-2) if centroid else pinhole_ref.unsqueeze(-2)

            ray_valid = ray.is_valid
            ray_err = ray.o[..., :2] - center_ref
            ray_err = torch.where(
                ray_valid.bool().unsqueeze(-1), ray_err, torch.zeros_like(ray_err)
            )
            mse = (ray_err**2).sum(-1).sum(-1) / (ray_valid.sum(-1) + EPSILON)

            if w_mask is None:
                if use_weight_mask:
                    w_mask = mse.sqrt().detach().clone()
                    w_mask = w_mask / (w_mask.mean() + EPSILON)
                    w_mask[0, :] = 2.0  # on-axis FoV
                else:
                    w_mask = torch.ones_like(mse)

            l_rms = torch.clamp(mse, min=EPSILON).sqrt()
            l_rms_weighted = (l_rms * w_mask).sum() / (w_mask.sum() + EPSILON)
            loss_rms_ls.append(l_rms_weighted)

        loss_rms = sum(loss_rms_ls) / len(loss_rms_ls)
        return loss_rms, loss_distortion

    # ================================================================
    # Example optimization function
    # ================================================================
    def sample_ring_arm_rays(
            self, 
            num_ring=8, 
            num_arm=2, 
            spp=2048, 
            depth=None, 
            wvln=None,
            scale_pupil=1.0, 
            sample_more_off_axis=True
    ):
        """Sample rays from object space using a ring-arm pattern.

        This method distributes sampling points (origins of ray bundles) on a polar grid in the object plane,
        defined by field of view. This is useful for capturing lens performance across the full field.
        The points include the center and `num_ring` rings with `num_arm` points on each.

        Args:
            num_ring (int): Number of rings to sample in the field of view.
            num_arm (int): Number of arms (spokes) to sample for each ring.
            spp (int): Total number of rays to be sampled, distributed among field points.
            depth (float): Depth of the object plane.
            wvln (float): Wavelength of the rays.
            scale_pupil (float): Scale factor for the pupil size.

        Returns:
            Ray: A Ray object containing the sampled rays.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln

        # Create points on rings and arms
        max_fov = self.rfov
        if sample_more_off_axis:
            beta_values = torch.linspace(0.0, 1.0, num_ring, device=self.device)
            beta_transformed = beta_values ** 0.5
            ring_fovs = max_fov * beta_transformed
        else:
            ring_fovs = max_fov * torch.linspace(0.0, 1.0, num_ring, device=self.device)

        arm_angles = torch.linspace(0.0, 2 * torch.pi, num_arm + 1, device=self.device)[:-1]
        ring_grid, arm_grid = torch.meshgrid(ring_fovs, arm_angles, indexing="ij")

        # Sample rays
        x = depth * torch.tan(ring_grid) * torch.cos(arm_grid)
        y = depth * torch.tan(ring_grid) * torch.sin(arm_grid)        
        z = torch.full_like(x, depth)
        points = torch.stack([x, y, z], dim=-1)  # shape: [num_ring, num_arm, 3]
        rays = self.sample_from_points(points=points, num_rays=spp, wvln=wvln, scale_pupil=scale_pupil)

        return rays

    def optimize(
        self,
        lrs=[1e-3, 1e-3, 1e-2, 1e-4],
        iterations=5000,
        test_per_iter=100,
        centroid=True,
        optim_mat=False,
        shape_control=True,
        sample_more_off_axis=True,
        depth=None,
        wvln_list=None,
        num_ring=8,
        num_arm=2,
        spp=1024,
        result_dir=None,
    ):
        """Optimise the lens by minimising RGB RMS spot errors.

        Runs a curriculum-learning training loop with Adam optimiser and cosine
        annealing. Periodically evaluates the lens, saves intermediate results,
        and optionally corrects surface shapes.

        Args:
            lrs (list, optional): Learning rates for [d, c, k, a] parameter groups.
                Defaults to [1e-3, 1e-3, 1e-2, 1e-4].
            iterations (int, optional): Total training iterations. Defaults to 5000.
            test_per_iter (int, optional): Evaluate and save every N iterations.
                Defaults to 100.
            centroid (bool, optional): If True, use the green-channel traced
                centroid as RMS centre reference; otherwise use pinhole model.
                Defaults to True.
            optim_mat (bool, optional): If True, include material parameters (n, V)
                in optimisation. Defaults to False.
            shape_control (bool, optional): If True, call :meth:`prune_surf`
                at each evaluation step. Defaults to True.
            depth (float, optional): Object distance in mm for ray sampling.
                Defaults to DEPTH.
            wvln_list (list[float], optional): Wavelengths in micrometers
                used to compute the RGB RMS loss. Defaults to the lens RGB
                wavelengths.
            num_ring (int, optional): Number of radial rings in the ring-arm
                field grid. Reduce for lower memory use. Defaults to 16.
            num_arm (int, optional): Number of azimuthal arms in the ring-arm
                field grid. Reduce for lower memory use. Defaults to 4.
            spp (int, optional): Rays sampled per field point per wavelength.
                Reduce for lower memory use. Defaults to 4096.
            result_dir (str, optional): Directory to save results. If None,
                auto-generates a timestamped directory. Defaults to None.

        Note:
            Debug hints:
                1. Slowly optimise with small learning rate.
                2. FoV and thickness should match well.
                3. Keep parameter ranges reasonable.
                4. Higher aspheric order is better but more sensitive.
                5. More iterations with larger ray sampling improves convergence.
        """
        depth = self.obj_depth if depth is None else depth
        wvln_list = self.wvln_ls if wvln_list is None else wvln_list
        assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths"

        # Result directory and logger
        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-DesignLens"

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
        logging.info(f"lr:{lrs}, iterations:{iterations}, num_ring:{num_ring}, num_arm:{num_arm}, rays_per_fov:{spp}.")

        # Optimizer and scheduler
        optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=iterations)

        # Training loop
        pbar = tqdm(
            total=iterations + 1,
            desc="Progress",
            postfix={"loss_rms": 0, "loss_reg": 0},
        )
        loss_history: dict = {}
        for i in range(iterations + 1):
            # ===> Evaluate the lens
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if shape_control and i > 0:
                        self.correct_shape()

                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}", full_eval=False)
                    save_loss_curve(loss_history, result_dir)

                    # Sample rays
                    self.calc_pupil()
                    rays_backup = []
                    for wv in wvln_list:
                        ray = self.sample_ring_arm_rays(
                            num_ring=num_ring,
                            num_arm=num_arm,
                            spp=spp,
                            depth=depth,
                            wvln=wv,
                            scale_pupil=1.05,
                            sample_more_off_axis=sample_more_off_axis,
                        )
                        rays_backup.append(ray)

                    # Calculate ray centers
                    pinhole_ref = -self.psf_center(
                        points_obj=ray.o[:, :, 0, :], method="pinhole"
                    )

            # ===> Optimize lens by minimizing RMS
            loss_rms, loss_distortion = self.loss_rms_from_ray(
                rays_backup, pinhole_ref, centroid=centroid
            )

            # Total loss
            w_reg = 0.1
            loss_reg, loss_dict = self.loss_reg()

            L_total = loss_rms + w_reg * (loss_reg + loss_distortion)

            # Back-propagation
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


    # ====================================================================================
    # Optimizer helpers
    # ====================================================================================
    @staticmethod
    def _sanitize_optimizer_grads(optimizer):
        """Replace non-finite gradient entries before optimizer updates.

        Non-finite gradient entries can occasionally appear for high-order
        aspheric coefficients while the forward loss is still finite.
        Letting one such entry reach Adam corrupts the corresponding surface
        parameter and turns all subsequent ray traces invalid.
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def find_diff_surf(self):
        """Get differentiable/optimizable surface indices.

        Returns a list of surface indices that can be optimized during lens design.
        Excludes the aperture surface and terminal sensor cover glass from
        optimization.

        Returns:
            list: Surface indices excluding fixed surfaces.
        """
        if self.aper_idx is None:
            diff_surf_range = list(range(len(self.surfaces)))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )

        if (
            len(self.surfaces) >= 2
            and type(self.surfaces[-2]) is Plane
            and type(self.surfaces[-1]) is Plane
        ):
            diff_surf_range = [
                idx for idx in diff_surf_range if idx < len(self.surfaces) - 2
            ]
        return diff_surf_range

    def get_optimizer_params(
        self,
        lrs=[1e-3, 1e-3, 1e-3, 1e-4],
        optim_mat=False,
        optim_surf_range=None,
    ):
        """Get optimizer parameters for different lens surfaces.

        Recommendation:
            For cellphone lens: [d, c, k, a], [1e-3, 1e-3, 1e-3, 1e-4]
            For camera lens: [d, c, 0, 0], [1e-3, 1e-3, 0, 0]

        Args:
            lrs (list): learning rate for different parameters.
            optim_mat (bool): whether to optimize material. Defaults to False.
            optim_surf_range (list): surface indices to be optimized. Defaults to None.

        Returns:
            list: optimizer parameters
        """
        # Find surfaces to be optimized
        if optim_surf_range is None:
            optim_surf_range = self.find_diff_surf()

        # Optimize lens surface parameters
        params = []
        for surf_idx in optim_surf_range:
            surf = self.surfaces[surf_idx]

            if isinstance(surf, Aperture):
                params += surf.get_optimizer_params(lrs=[lrs[0]])

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(
                    lrs=lrs[:4], optim_mat=optim_mat,
                )

            elif isinstance(surf, Phase):
                params += surf.get_optimizer_params(lrs=[lrs[0], lrs[4]])

            # elif isinstance(surf, GaussianRBF):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            # elif isinstance(surf, NURBS):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Plane):
                params += surf.get_optimizer_params(lrs=[lrs[0]], optim_mat=optim_mat)

            # elif isinstance(surf, PolyEven):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            elif isinstance(surf, ThinLens):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            else:
                raise Exception(
                    f"Surface type {surf.__class__.__name__} is not supported for optimization yet."
                )

        # Optimize sensor place
        self.d_sensor.requires_grad = True
        params += [{"params": self.d_sensor, "lr": lrs[0]}]

        return params

    def get_optimizer(
        self,
        lrs=[1e-3, 1e-3, 1e-3, 1e-4],
        optim_surf_range=None,
        optim_mat=False,
    ):
        """Get optimizers and schedulers for different lens parameters.

        Args:
            lrs (list): learning rate for different parameters [d, c, k, a].
            optim_surf_range (list): surface indices to be optimized. Defaults to None.
            optim_mat (bool): whether to optimize material. Defaults to False.

        Returns:
            list: optimizer parameters
        """
        params = self.get_optimizer_params(
            lrs=lrs, optim_surf_range=optim_surf_range, optim_mat=optim_mat,
        )
        optimizer = torch.optim.Adam(params)
        return optimizer
