# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Second-order optimisation for GeoLens.

Holds higher-order optimisers that complement the first-order ``optimize()``
loop in :mod:`optim`:

    * :meth:`GeoLensOptim2nd.optimize_lbfgs` – L-BFGS with strong-Wolfe line
      search and diagonal preconditioning matched to Adam's per-group LRs.
    * :meth:`GeoLensOptim2nd.optimize_lm` – (planned) Levenberg-Marquardt
      with per-ray Jacobian to handle the rank-deficient residual problem.

All methods reuse the loss formulation from :meth:`GeoLensOptim.optimize`:

    ``L = loss_rms + w_reg * (loss_reg + loss_distortion)``

This module is a mixin – it expects the host class to provide the methods
defined in :class:`GeoLensOptim` (``loss_rms_from_ray``, ``loss_reg``,
``sample_ring_arm_rays``, ``get_optimizer_params``, ``correct_shape``,
``write_lens_json``, ``analysis``, ``calc_pupil``, ``psf_center``).
"""

import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from ..config import EPSILON
from ..utils_deeplens import record_losses, save_loss_curve

logger = logging.getLogger(__name__)


class GeoLensOptim2nd:
    """Mixin providing second-order optimisers for ``GeoLens``.

    Designed to slot into the same MRO as :class:`GeoLensOptim`; both can be
    inherited together, and the second-order methods call ``self.loss_reg``,
    ``self.loss_rms_from_ray``, etc. defined on the first-order mixin.
    """

    # ================================================================
    # Second-order optimization: L-BFGS
    # ================================================================
    def optimize_lbfgs(
        self,
        iterations=200,
        test_per_iter=20,
        lr=1.0,
        max_iter_per_step=20,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
        precond_lrs=(1e-3, 1e-3, 1e-2, 1e-4),
        optim_mat=False,
        shape_control=True,
        centroid=True,
        sample_more_off_axis=True,
        depth=None,
        wvln_list=None,
        num_ring=8,
        num_arm=2,
        spp=1024,
        w_reg=0.1,
        resample_per_iter=1,
        grad_clip_norm=None,
        result_dir=None,
    ):
        """Optimise the lens with second-order L-BFGS.

        Uses the same loss as :meth:`optimize`:

            ``L = loss_rms + w_reg * (loss_reg + loss_distortion)``

        and the same parameter groups as :meth:`get_optimizer_params`. L-BFGS
        approximates the inverse Hessian from gradient differences, so a single
        outer iteration typically makes much more progress than one Adam step.

        Stochasticity in :meth:`loss_cra` and :meth:`loss_ray_bend` (they
        sample rings of rays each call) is fixed inside the closure by
        re-seeding the RNG with the current outer iteration index, so every
        line-search evaluation within an outer iteration sees the same random
        bundle. RMS rays are pre-sampled and frozen between resamples, exactly
        as in :meth:`optimize`.

        Diagonal preconditioning: lens parameters span ~10⁴ in magnitude
        (thickness ~1 mm vs aspheric ``a18`` ~1e-9). Plain L-BFGS overshoots
        the small-scale params on the first step. We optimise over scaled
        shadow params ``q = p / scale`` where ``scale = sqrt(lr_p)`` using the
        same per-group LRs Adam uses — so a unit step in ``q`` corresponds to
        the same physical update Adam would make at one learning step.

        Args:
            iterations (int): Number of outer L-BFGS steps.
            test_per_iter (int): Evaluate / save / resample every N outer steps.
            lr (float): Step size passed to ``torch.optim.LBFGS``. With
                ``strong_wolfe`` line search the actual step is chosen by the
                line search, so ``lr=1.0`` is the conventional choice.
            max_iter_per_step (int): Maximum closure evaluations per outer step
                (PyTorch LBFGS ``max_iter``).
            history_size (int): Number of (s, y) curvature pairs to keep.
            tolerance_grad (float): Termination tolerance on gradient infinity
                norm.
            tolerance_change (float): Termination tolerance on relative function
                value change.
            line_search_fn (str | None): ``"strong_wolfe"`` or ``None``.
            precond_lrs (tuple): Per-group Adam LRs used to derive shadow scales
                ``[d, c, k, a]``. Aspheric ``a`` LR is further divided by
                ``r^order`` inside :meth:`get_optimizer_params`.
            optim_mat (bool): Whether to include material (n, V) in the
                parameter set.
            shape_control (bool): Call :meth:`correct_shape` at each evaluation.
            centroid (bool): Use traced green-channel centroid as RMS reference.
            sample_more_off_axis (bool): Bias the ring grid toward off-axis
                fields.
            depth (float, optional): Object distance in mm. Defaults to
                ``self.obj_depth``.
            wvln_list (list[float], optional): RGB wavelengths in micrometres.
            num_ring, num_arm, spp (int): Ring-arm field grid for RMS rays.
            w_reg (float): Weight on ``(loss_reg + loss_distortion)``.
            resample_per_iter (int): Resample RMS rays every N outer steps.
                Defaults to 1 — fresh rays every step keeps the LBFGS history
                from chasing one bundle.
            grad_clip_norm (float, optional): If given, clip shadow grad norm
                to this value before each LBFGS update.
            result_dir (str, optional): Output directory.
        """
        depth = self.obj_depth if depth is None else depth
        wvln_list = self.wvln_ls if wvln_list is None else wvln_list
        assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths"

        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-LBFGS"
        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            root = logging.getLogger()
            root.setLevel("DEBUG")
            fmt = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel("INFO")
            fh = logging.FileHandler(f"{result_dir}/output.log")
            fh.setFormatter(fmt)
            fh.setLevel("INFO")
            root.addHandler(sh)
            root.addHandler(fh)
        logging.info(
            f"[LBFGS] iters={iterations}, max_iter_per_step={max_iter_per_step}, "
            f"history={history_size}, line_search={line_search_fn}, w_reg={w_reg}, "
            f"num_ring={num_ring}, num_arm={num_arm}, spp={spp}"
        )

        # Build parameter groups with the SAME per-group LRs Adam uses, then
        # turn each into a "scaled shadow" tensor q = p / scale (scale=sqrt(lr)).
        # LBFGS optimises q; we push q*scale back into the lens before every
        # forward and pull p.grad*scale into q.grad after every backward.
        param_groups = self.get_optimizer_params(
            lrs=list(precond_lrs), optim_mat=optim_mat,
        )

        raw_params = []   # original lens tensors (still leaves; receive .grad)
        scales = []       # per-tensor scaling factor (float)
        for g in param_groups:
            ps = g["params"]
            if isinstance(ps, torch.Tensor):
                ps = [ps]
            group_lr = float(g.get("lr", 1.0))
            # ``get_optimizer_params`` produces tiny LRs for high-order
            # aspheric coefficients (~1e-18 for a18). Keep the full range so
            # the diagonal preconditioner matches the per-param sensitivity.
            group_scale = max(group_lr, 1e-30) ** 0.5
            for p in ps:
                raw_params.append(p)
                scales.append(group_scale)

        shadow_params = []
        for p, s in zip(raw_params, scales):
            with torch.no_grad():
                init = p.data.detach().clone() / s
            shadow_params.append(torch.nn.Parameter(init, requires_grad=True))

        def push_to_raw():
            with torch.no_grad():
                for raw, sh, s in zip(raw_params, shadow_params, scales):
                    raw.data.copy_(sh.data * s)

        def pull_raw_to_shadow():
            """Sync shadow ← raw. Call after any in-place mutation of raw
            tensors (e.g. ``correct_shape``) so the LBFGS variable matches
            the lens state."""
            with torch.no_grad():
                for raw, sh, s in zip(raw_params, shadow_params, scales):
                    sh.data.copy_(raw.data / s)

        def pull_grad_to_shadow():
            for raw, sh, s in zip(raw_params, shadow_params, scales):
                if raw.grad is None:
                    if sh.grad is None:
                        sh.grad = torch.zeros_like(sh)
                    else:
                        sh.grad.zero_()
                else:
                    # dL/dq = dL/dp * dp/dq = dL/dp * s
                    g_q = raw.grad.detach() * s
                    if sh.grad is None:
                        sh.grad = g_q.clone()
                    else:
                        sh.grad.copy_(g_q)

        def zero_raw_grads():
            for raw in raw_params:
                if raw.grad is not None:
                    raw.grad.detach_()
                    raw.grad.zero_()

        optimizer = torch.optim.LBFGS(
            shadow_params,
            lr=lr,
            max_iter=max_iter_per_step,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )

        # Mutable holders so the closure can read the latest cached rays.
        state = {
            "rays_backup": None,
            "pinhole_ref": None,
            "iter": 0,
            "last_loss_rms": 0.0,
            "last_loss_reg": 0.0,
            "last_loss_distortion": 0.0,
            "last_loss_dict": {},
            "last_total": 0.0,
        }

        def _resample_rays():
            with torch.no_grad():
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
                pinhole_ref = -self.psf_center(
                    points_obj=ray.o[:, :, 0, :], method="pinhole"
                )
            state["rays_backup"] = rays_backup
            state["pinhole_ref"] = pinhole_ref

        def closure():
            # Re-seed inside the closure so loss_cra / loss_ray_bend use the
            # same random ring rays for every line-search evaluation within
            # this outer step. Otherwise strong_wolfe sees a noisy objective.
            torch.manual_seed(0xBEEF + state["iter"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0xBEEF + state["iter"])

            # Sync raw lens params from current shadow q, then zero raw grads.
            push_to_raw()
            zero_raw_grads()
            optimizer.zero_grad(set_to_none=False)

            loss_rms, loss_distortion = self.loss_rms_from_ray(
                state["rays_backup"], state["pinhole_ref"], centroid=centroid
            )
            loss_reg, loss_dict = self.loss_reg(
                w_mat=1.0 if optim_mat else 0.0,
            )
            L_total = loss_rms + w_reg * (loss_reg + loss_distortion)
            L_total.backward()

            # Sanitize raw grads (NaN -> 0), then transfer to shadow grads.
            for raw in raw_params:
                if raw.grad is not None:
                    raw.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            pull_grad_to_shadow()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(shadow_params, grad_clip_norm)

            state["last_loss_rms"] = loss_rms.item()
            state["last_loss_reg"] = loss_reg.item()
            state["last_loss_distortion"] = loss_distortion.item()
            state["last_loss_dict"] = loss_dict
            state["last_total"] = L_total.item()
            return L_total

        pbar = tqdm(
            total=iterations + 1,
            desc="LBFGS",
            postfix={"loss_rms": 0, "loss_reg": 0},
        )
        loss_history: dict = {}

        # Prime caches before first step.
        _resample_rays()

        for i in range(iterations + 1):
            state["iter"] = i

            # Make sure raw lens params reflect the current shadow state
            # before any evaluation, resample, or shape-correction step.
            push_to_raw()

            # === Evaluate ===
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if shape_control and i > 0:
                        self.correct_shape()
                        # correct_shape mutates raw params in place; bring
                        # the shadow vars back in sync.
                        pull_raw_to_shadow()
                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}", full_eval=False)
                    save_loss_curve(loss_history, result_dir)

            # === Resample rays for this outer step ===
            if i > 0 and (i % resample_per_iter == 0):
                _resample_rays()

            # === Outer L-BFGS step ===
            if i < iterations:
                optimizer.step(closure)
                # closure already pushed; closure's last call may not be the
                # accepted point, so explicitly resync after the step.
                push_to_raw()

                record_losses(loss_history, i, {
                    "loss_rms": state["last_loss_rms"],
                    "loss_reg": state["last_loss_reg"],
                    "loss_distortion": state["last_loss_distortion"],
                    **state["last_loss_dict"],
                })
                pbar.set_postfix(
                    loss_rms=state["last_loss_rms"],
                    loss_reg=state["last_loss_reg"],
                    loss_distortion=state["last_loss_distortion"],
                    **state["last_loss_dict"],
                )
            pbar.update(1)

        save_loss_curve(loss_history, result_dir)
        pbar.close()

    # ================================================================
    # Second-order optimization: Levenberg-Marquardt with per-ray Jacobian
    # ================================================================
    def optimize_lm(
        self,
        iterations=50,
        test_per_iter=5,
        lambda_init=1e-3,
        lambda_min=1e-12,
        lambda_max=1e8,
        up_factor=10.0,
        down_factor=2.0,
        max_inner=8,
        precond_lrs=(1e-3, 1e-3, 1e-2, 1e-4),
        optim_mat=False,
        shape_control=True,
        centroid=True,
        sample_more_off_axis=True,
        depth=None,
        wvln_list=None,
        num_ring=8,
        num_arm=2,
        spp=64,
        jac_chunk=256,
        w_reg=0.1,
        resample_per_iter=1,
        result_dir=None,
    ):
        """Optimise the lens with Levenberg-Marquardt using the per-ray Jacobian.

        Residual vector ``r`` is composed of:

            * **Per-ray xy errors** — for each (wavelength, field, ray) we
              push the rescaled difference ``(sensor_xy - center_ref)``
              into ``r``. ``len = 3 * N_fields * N_rays * 2``.
              These are the "per-ray Jacobian" components — keeping each
              ray as its own residual rather than collapsing to a per-field
              RMS keeps ``J`` full column rank (the loss-as-scalar route
              would only give a single direction in parameter space).
            * **Regularisation pseudo-residual** —
              ``sqrt(2 * w_reg * (loss_reg + loss_distortion))`` appended
              as a single scalar so ``||r||²/2`` matches the LBFGS / Adam
              composite loss.

        Each outer iteration:
            1. Build ``r`` and Jacobian ``J = ∂r/∂q`` in preconditioned
               shadow space ``q = p / sqrt(lr_p)``.
            2. Solve damped normal equations
               ``(JᵀJ + λ·diag(JᵀJ)) δq = -Jᵀ r`` (Marquardt scaling).
            3. Trial ``q + δq``; if ``||r_new||² < ||r||²`` accept and divide
               ``λ`` by ``down_factor``, else revert and multiply by
               ``up_factor`` (up to ``max_inner`` retries).

        Args:
            iterations (int): Number of outer LM steps.
            test_per_iter (int): Save / analyse every N outer steps.
            lambda_init (float): Initial damping.
            lambda_min, lambda_max (float): Bounds on damping.
            up_factor (float): Damping multiplier when a step is rejected.
            down_factor (float): Damping divisor when a step is accepted.
            max_inner (int): Maximum trial steps per outer iteration before
                bailing out (helps escape stuck states).
            precond_lrs (tuple): Per-group LRs ``[d, c, k, a]`` controlling
                the diagonal preconditioner. Same convention as
                :meth:`optimize_lbfgs`.
            optim_mat (bool): Optimise material (n, V) parameters.
            shape_control (bool): Call :meth:`correct_shape` at evaluation
                steps.
            centroid (bool): Use traced green-channel centroid as RMS ref.
            sample_more_off_axis (bool): Bias the ring grid toward off-axis
                fields.
            depth (float, optional): Object distance in mm.
            wvln_list (list[float], optional): RGB wavelengths.
            num_ring, num_arm, spp (int): Ring-arm field grid for rays.
                ``spp`` defaults to 64 — every ray contributes 2 residuals
                so total residual count is ``3·num_ring·num_arm·spp·2 + 1``.
                The Jacobian is materialised in chunks; reduce ``spp`` or
                increase ``jac_chunk`` if you OOM.
            jac_chunk (int): Number of residual rows computed per batched
                backward pass. Larger = faster but more peak GPU memory.
            w_reg (float): Weight on ``(loss_reg + loss_distortion)``.
            resample_per_iter (int): Resample rays every N outer steps.
            result_dir (str, optional): Output directory.
        """
        depth = self.obj_depth if depth is None else depth
        wvln_list = self.wvln_ls if wvln_list is None else wvln_list
        assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths"

        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-LM"
        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            root = logging.getLogger()
            root.setLevel("DEBUG")
            fmt = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel("INFO")
            fh = logging.FileHandler(f"{result_dir}/output.log")
            fh.setFormatter(fmt)
            fh.setLevel("INFO")
            root.addHandler(sh)
            root.addHandler(fh)
        logging.info(
            f"[LM] iters={iterations}, λ0={lambda_init}, up={up_factor}, "
            f"down={down_factor}, max_inner={max_inner}, w_reg={w_reg}, "
            f"num_ring={num_ring}, num_arm={num_arm}, spp={spp}"
        )

        # --- Preconditioning: same shadow-param trick as LBFGS ---
        param_groups = self.get_optimizer_params(
            lrs=list(precond_lrs), optim_mat=optim_mat,
        )
        raw_params = []
        scales_list = []
        for g in param_groups:
            ps = g["params"]
            if isinstance(ps, torch.Tensor):
                ps = [ps]
            group_lr = float(g.get("lr", 1.0))
            group_scale = max(group_lr, 1e-30) ** 0.5
            for p in ps:
                raw_params.append(p)
                scales_list.append(group_scale)

        # Per-parameter scale vector, broadcast for matmul.
        # Length matches the flat shadow vector (each raw tensor is scalar in
        # our setup, but keep it general).
        param_sizes = [p.numel() for p in raw_params]
        total_params = sum(param_sizes)
        scale_vec = torch.cat([
            torch.full((sz,), s, device=self.device, dtype=torch.float32)
            for sz, s in zip(param_sizes, scales_list)
        ])

        # Shadow params in q-space (flat vector). q_init = p_init / scale.
        with torch.no_grad():
            q = torch.cat([
                (p.detach().reshape(-1) / s)
                for p, s in zip(raw_params, scales_list)
            ]).to(self.device).clone()

        def push_q_to_raw(q_flat):
            """Copy q*scale into the raw lens params (in-place, no grad)."""
            with torch.no_grad():
                offset = 0
                for p, s, sz in zip(raw_params, scales_list, param_sizes):
                    p.data.copy_((q_flat[offset:offset + sz] * s).view_as(p))
                    offset += sz

        def pull_raw_to_q(q_flat):
            """Refresh q from current raw lens param values."""
            with torch.no_grad():
                offset = 0
                for p, s, sz in zip(raw_params, scales_list, param_sizes):
                    q_flat[offset:offset + sz].copy_(p.data.reshape(-1) / s)
                    offset += sz

        # --- Ray resampling helpers ---
        state = {
            "rays_backup": None,
            "pinhole_ref": None,
            "iter": 0,
        }

        def _resample_rays():
            with torch.no_grad():
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
                pinhole_ref = -self.psf_center(
                    points_obj=ray.o[:, :, 0, :], method="pinhole"
                )
            state["rays_backup"] = rays_backup
            state["pinhole_ref"] = pinhole_ref

        def _compute_residual_vector(create_graph=False):
            """Build the LM residual vector r with shape [N_total].

            Per-ray xy components are scaled per-field by
            ``sqrt(w_mask / N_valid)`` so ``||r_xy||²`` is the weighted mean
            of squared distances — same averaging as :meth:`loss_rms_from_ray`
            (but on squared scale rather than RMS), which gives a sum-of-
            squares objective that LM optimises naturally. The reg residual
            is appended as a single scalar so ``||r||²`` matches the LBFGS
            loss in expectation.

            Also returns scalar component losses for logging.
            """
            # Seed determinism (loss_cra / loss_ray_bend sample rays).
            torch.manual_seed(0xC0DE + state["iter"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0xC0DE + state["iter"])

            # Trace each wavelength and accumulate xy errors.
            wvln_order = [1, 0, 2]  # green first
            r_xy_components = []
            l_distortion = torch.tensor(0.0, device=self.device)
            l_rms_components_sq = []
            w_mask = None
            center_ref = None

            for wv_idx in wvln_order:
                ray = state["rays_backup"][wv_idx].clone()
                ray = self.trace2sensor(ray)
                if center_ref is None:
                    l_distortion, centroid_xy = self.loss_distortion_from_ray(
                        ray, state["pinhole_ref"]
                    )
                    center_ref = (
                        centroid_xy.detach().unsqueeze(-2) if centroid
                        else state["pinhole_ref"].unsqueeze(-2)
                    )

                ray_valid = ray.is_valid
                ray_err = ray.o[..., :2] - center_ref
                ray_err = torch.where(
                    ray_valid.bool().unsqueeze(-1),
                    ray_err,
                    torch.zeros_like(ray_err),
                )

                # Per-field normalisation: residuals should sum-of-squares
                # to the per-field weighted mse.
                n_valid = ray_valid.sum(-1, keepdim=True).clamp_min(1.0).float()
                # Weight mask anchored on green channel
                mse_field = (ray_err ** 2).sum(-1).sum(-1) / (
                    ray_valid.sum(-1) + EPSILON
                )
                if w_mask is None:
                    w_mask = mse_field.sqrt().detach().clone()
                    w_mask = w_mask / (w_mask.mean() + EPSILON)
                    w_mask[0, :] = 2.0  # on-axis weight
                # Scale each ray's residual so ||·||² equals the per-field
                # weighted MSE summed across fields.
                scale_field = (w_mask / n_valid.squeeze(-1)).sqrt()
                # Broadcast scale_field over rays and xy components.
                ray_err_scaled = ray_err * scale_field[..., None, None]
                r_xy_components.append(ray_err_scaled.reshape(-1))

                # For logging only — replicate loss_rms_from_ray exactly.
                l_rms_field = mse_field.clamp_min(EPSILON).sqrt()
                l_rms_weighted = (l_rms_field * w_mask).sum() / (
                    w_mask.sum() + EPSILON
                )
                l_rms_components_sq.append(l_rms_weighted)

            r_xy = torch.cat(r_xy_components, dim=0)

            # Reg loss as a single pseudo-residual.
            loss_reg_scalar, loss_dict = self.loss_reg(
                w_mat=1.0 if optim_mat else 0.0,
            )
            reg_combined = (loss_reg_scalar + l_distortion).clamp_min(0.0)
            r_reg = torch.sqrt(2.0 * w_reg * reg_combined + EPSILON).reshape(1)

            r = torch.cat([r_xy, r_reg], dim=0)

            # For logging (these mirror loss_rms_from_ray):
            l_rms_scalar = sum(l_rms_components_sq) / len(l_rms_components_sq)
            log = {
                "loss_rms": float(l_rms_scalar.detach()),
                "loss_reg": float(loss_reg_scalar.detach()),
                "loss_distortion": float(l_distortion.detach()),
                **loss_dict,
            }
            return r, log

        def _compute_jacobian(q_flat):
            """Build J = dr/dq via chunked batched reverse-mode AD.

            We write q*scale into the lens params (raw_params remain leaves on
            q's graph). For each chunk of ``jac_chunk`` rows we do one
            ``is_grads_batched=True`` backward pass, materialising only the
            chunk's gradients at a time. This keeps peak GPU memory bounded
            regardless of total residual count N.

            Returns:
                tuple: ``(r, J, log)`` with ``r`` of shape ``[N]``, ``J`` of
                shape ``[N, total_params]`` in q-space.
            """
            push_q_to_raw(q_flat)
            for p in raw_params:
                p.requires_grad_(True)
                if p.grad is not None:
                    p.grad = None

            r, log = _compute_residual_vector()
            N = r.numel()
            dtype = r.dtype
            device = r.device

            # Pre-allocate per-param Jacobian columns.
            J_per_param = [
                torch.zeros(N, p.numel(), device=device, dtype=dtype)
                for p in raw_params
            ]

            chunk = max(1, int(jac_chunk))
            n_chunks = (N + chunk - 1) // chunk

            for ci in range(n_chunks):
                lo = ci * chunk
                hi = min(lo + chunk, N)
                n_c = hi - lo
                is_last = (ci == n_chunks - 1)

                # Identity rows: one-hot grad_output per residual in this chunk.
                grad_outs = torch.zeros(n_c, N, device=device, dtype=dtype)
                grad_outs[torch.arange(n_c), torch.arange(lo, hi)] = 1.0

                try:
                    grads = torch.autograd.grad(
                        outputs=r,
                        inputs=raw_params,
                        grad_outputs=grad_outs,
                        retain_graph=not is_last,
                        create_graph=False,
                        is_grads_batched=True,
                        allow_unused=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    if "out of memory" in msg.lower():
                        raise RuntimeError(
                            f"[LM] OOM in Jacobian chunk (size={n_c}). "
                            f"Reduce --jac_chunk (was {chunk}) or --spp."
                        ) from e
                    # Some ops lack batched derivatives → rebuild graph + per-row.
                    logging.warning(
                        f"[LM] batched grad failed at chunk {ci} ({msg[:120]}…); "
                        "rebuilding graph and using per-row for this chunk"
                    )
                    # Need to recompute forward to get a fresh graph.
                    for p in raw_params:
                        if p.grad is not None:
                            p.grad = None
                    r2, _ = _compute_residual_vector()
                    grads = _per_row_grads_chunk(r2, raw_params, lo, hi)

                for k, g in enumerate(grads):
                    if g is not None:
                        g = g.reshape(n_c, -1)
                        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                        J_per_param[k][lo:hi].copy_(g)

            J_raw = torch.cat(J_per_param, dim=1)  # [N, total_params]
            # Convert to shadow space: dr/dq = dr/dp * dp/dq = dr/dp * scale.
            J = J_raw * scale_vec[None, :]
            return r.detach(), J.detach(), log

        def _per_row_grads_chunk(r, params, lo, hi):
            """Slow per-row fallback for one chunk."""
            n_c = hi - lo
            outs = [torch.zeros(n_c, p.numel(), device=r.device, dtype=r.dtype) for p in params]
            for offset, i in enumerate(range(lo, hi)):
                gi = torch.autograd.grad(
                    outputs=r[i],
                    inputs=params,
                    retain_graph=(offset < n_c - 1),
                    allow_unused=True,
                )
                for k, gk in enumerate(gi):
                    if gk is not None:
                        outs[k][offset].copy_(
                            torch.nan_to_num(gk.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
                        )
            return tuple(outs)

        # --- Optimisation loop ---
        pbar = tqdm(
            total=iterations + 1, desc="LM", postfix={"loss_rms": 0, "λ": lambda_init}
        )
        loss_history: dict = {}
        lambda_lm = float(lambda_init)

        # Prime ray cache.
        _resample_rays()

        # Initial loss before any step.
        push_q_to_raw(q)
        with torch.no_grad():
            r0, log0 = _compute_residual_vector()
            obj_now = 0.5 * float((r0 ** 2).sum())

        for i in range(iterations + 1):
            state["iter"] = i

            push_q_to_raw(q)

            # === Evaluate ===
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if shape_control and i > 0:
                        self.correct_shape()
                        pull_raw_to_q(q)
                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}", full_eval=False)
                    save_loss_curve(loss_history, result_dir)

            # === Resample rays ===
            if i > 0 and (i % resample_per_iter == 0):
                _resample_rays()
                # Recompute obj_now after resample (rays changed).
                push_q_to_raw(q)
                with torch.no_grad():
                    r0, log0 = _compute_residual_vector()
                    obj_now = 0.5 * float((r0 ** 2).sum())

            if i >= iterations:
                pbar.update(1)
                break

            # === LM step ===
            # Build Jacobian + residuals at current q.
            r, J, log = _compute_jacobian(q)
            JtJ = J.T @ J
            Jtr = J.T @ r
            diag_JtJ = torch.diagonal(JtJ).clone()
            # Marquardt: scale damping by diag(JtJ).
            diag_JtJ_safe = diag_JtJ.clamp_min(EPSILON)

            accepted = False
            for trial in range(max_inner):
                A = JtJ + lambda_lm * torch.diag(diag_JtJ_safe)
                try:
                    delta_q = torch.linalg.solve(A, -Jtr)
                except RuntimeError:
                    # Singular A → bump damping and retry.
                    lambda_lm = min(lambda_lm * up_factor, lambda_max)
                    continue
                delta_q = torch.nan_to_num(delta_q, nan=0.0, posinf=0.0, neginf=0.0)

                # Trial step.
                q_trial = q + delta_q
                push_q_to_raw(q_trial)
                with torch.no_grad():
                    r_new, log_new = _compute_residual_vector()
                    obj_new = 0.5 * float((r_new ** 2).sum())

                # Accept if objective decreased.
                if torch.isfinite(torch.tensor(obj_new)) and obj_new < obj_now:
                    q = q_trial
                    obj_now = obj_new
                    log = log_new
                    lambda_lm = max(lambda_lm / down_factor, lambda_min)
                    accepted = True
                    break
                else:
                    # Revert and bump damping.
                    push_q_to_raw(q)
                    lambda_lm = min(lambda_lm * up_factor, lambda_max)

            if not accepted:
                # All trials rejected; lens stays at q. Log current state.
                logging.debug(
                    f"[LM] iter {i}: no descent after {max_inner} trials, "
                    f"λ={lambda_lm:.2e}"
                )

            record_losses(loss_history, i, {**log, "lambda_lm": lambda_lm, "obj": obj_now})
            pbar.set_postfix(
                loss_rms=log["loss_rms"],
                loss_reg=log["loss_reg"],
                loss_distortion=log["loss_distortion"],
                obj=obj_now,
                lam=lambda_lm,
            )
            pbar.update(1)

        save_loss_curve(loss_history, result_dir)
        pbar.close()
