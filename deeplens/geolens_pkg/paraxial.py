# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.

"""Paraxial ABCD-matrix fast-path for :class:`GeoLens` startup."""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _tensor_to_float(value, default: float = 0.0) -> float:
    """Best-effort cast of a tensor / python scalar / None to float."""
    if value is None:
        return default
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _surface_curvature(surf) -> float:
    """Paraxial vertex curvature, absorbing any legacy 2nd-order asphere term."""
    c = _tensor_to_float(getattr(surf, "c", None), 0.0)
    a2 = _tensor_to_float(getattr(surf, "ai2", None), 0.0)
    return c + 2.0 * a2


class GeoLensParaxial:
    """Paraxial ABCD-matrix fast-path for lens initialization."""

    @torch.no_grad()
    def calc_foclen_paraxial(self, wvln: float = None) -> float:
        """Compute EFL via the paraxial ABCD matrix product."""
        wvln = self.wvln_primary if wvln is None else wvln
        if len(self.surfaces) < 2:
            efl = float("nan")
            self.efl = efl
            self.foclen = efl
            self.bfl = float("nan")
            return efl

        # Accumulator for M = [[A, B], [C, D]]; starts as identity.
        A, B, C, D = 1.0, 0.0, 0.0, 1.0

        n_prev = 1.0  # object-space medium (air)
        surfaces = list(self.surfaces)
        for i, surf in enumerate(surfaces):
            n_i = float(surf.mat2.refractive_index(wvln))
            P_i = (n_i - n_prev) * _surface_curvature(surf)

            # Refraction: left-multiply by [[1, 0], [-P_i, 1]].
            C, D = -P_i * A + C, -P_i * B + D

            # Translation to the next surface (if any).
            if i < len(surfaces) - 1:
                t = float(surfaces[i + 1].d.item() - surf.d.item())
                tr = t / n_i if n_i != 0.0 else 0.0
                # Left-multiply by [[1, tr], [0, 1]].
                A, B = A + tr * C, B + tr * D

            n_prev = n_i

        efl = -1.0 / C if C != 0.0 else float("nan")
        self.efl = efl
        self.foclen = efl
        if hasattr(self, "d_sensor"):
            self.bfl = float(self.d_sensor.item() - self.surfaces[-1].d.item())
        else:
            self.bfl = float("nan")
        return efl

    @torch.no_grad()
    def calc_fov_paraxial(self) -> None:
        """Populate FoV fields from ``foclen`` and ``sensor_size``."""
        if not hasattr(self, "foclen") or not math.isfinite(self.foclen):
            return

        self.vfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.hfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov_eff = self.dfov / 2
        self.rfov = self.rfov_eff
        self.real_dfov = self.dfov
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

    @torch.no_grad()
    def _calc_pupils_from_abcd(
        self, wvln: float = None
    ) -> tuple[float, float, float, float]:
        """Pure-ABCD entrance and exit pupil solve."""
        wvln = self.wvln_primary if wvln is None else wvln
        aper_idx = self.aper_idx
        if aper_idx is None or aper_idx >= len(self.surfaces):
            z0 = float(self.surfaces[0].d.item()) if self.surfaces else 0.0
            r0 = float(self.surfaces[0].r) if self.surfaces else 0.0
            return z0, r0, z0, r0

        aper_surf = self.surfaces[aper_idx]
        aper_z = float(aper_surf.d.item())
        aper_r = float(aper_surf.r)
        if aper_surf.is_square:
            aper_r = math.sqrt(2.0) * aper_r

        def abcd(surfaces, n_before: float) -> tuple[float, float, float, float, float]:
            A, B, C, D = 1.0, 0.0, 0.0, 1.0
            n_prev = n_before
            for i, surf in enumerate(surfaces):
                n_i = float(surf.mat2.refractive_index(wvln))
                P = (n_i - n_prev) * _surface_curvature(surf)
                C, D = -P * A + C, -P * B + D
                if i < len(surfaces) - 1:
                    t = float(surfaces[i + 1].d.item() - surf.d.item())
                    tr = t / n_i if n_i != 0.0 else 0.0
                    A, B = A + tr * C, B + tr * D
                n_prev = n_i
            return A, B, C, D, n_prev

        pre = list(self.surfaces[:aper_idx])
        if pre:
            A_pre, B_pre, C_pre, D_pre, n_after_pre = abcd(pre, n_before=1.0)
            last_pre_d = float(pre[-1].d.item())
            t_to_stop = (aper_z - last_pre_d) / (n_after_pre or 1.0)
            A_pre, B_pre = A_pre + t_to_stop * C_pre, B_pre + t_to_stop * D_pre
        else:
            A_pre, B_pre, C_pre, D_pre = 1.0, 0.0, 0.0, 1.0

        if abs(A_pre) < 1e-12:
            entr_pupilz = aper_z
            entr_pupilr = aper_r
        else:
            t_ep = -B_pre / A_pre
            entr_pupilz = float(self.surfaces[0].d.item()) - t_ep
            entr_pupilr = aper_r / abs(A_pre)

        post = list(self.surfaces[aper_idx + 1 :])
        if post:
            n_after_aper = float(aper_surf.mat2.refractive_index(wvln))
            t_stop_to_first_post = (
                float(post[0].d.item()) - aper_z
            ) / (n_after_aper or 1.0)
            A_post, B_post, C_post, D_post = 1.0, 0.0, 0.0, 1.0
            A_post, B_post = (
                A_post + t_stop_to_first_post * C_post,
                B_post + t_stop_to_first_post * D_post,
            )
            A2, B2, C2, D2, _ = abcd(post, n_before=n_after_aper)
            A_comp = A2 * A_post + B2 * C_post
            B_comp = A2 * B_post + B2 * D_post
            C_comp = C2 * A_post + D2 * C_post
            D_comp = C2 * B_post + D2 * D_post
            A_post, B_post, C_post, D_post = A_comp, B_comp, C_comp, D_comp
        else:
            A_post, B_post, C_post, D_post = 1.0, 0.0, 0.0, 1.0

        if abs(D_post) < 1e-12:
            exit_pupilz = aper_z
            exit_pupilr = aper_r
        else:
            last_z = float(self.surfaces[-1].d.item())
            exit_pupilz = last_z - B_post / D_post
            exit_pupilr = aper_r / abs(D_post)

        return entr_pupilz, entr_pupilr, exit_pupilz, exit_pupilr

    @torch.no_grad()
    def calc_pupil_paraxial(self) -> None:
        """Populate pupil fields from a pure-ABCD paraxial solve."""
        entr_z, entr_r, exit_z, exit_r = self._calc_pupils_from_abcd()

        self.entr_pupilz = entr_z
        self.entr_pupilr = entr_r
        self.entr_pupilz_parax = entr_z
        self.entr_pupilr_parax = entr_r
        self.exit_pupilz = exit_z
        self.exit_pupilr = exit_r
        self.exit_pupilz_parax = exit_z
        self.exit_pupilr_parax = exit_r

        if self.entr_pupilr > 0 and math.isfinite(self.foclen):
            self.fnum = self.foclen / (2 * self.entr_pupilr)
        else:
            self.fnum = float("nan")

    def post_computation_paraxial(self) -> None:
        """Fast drop-in for :meth:`GeoLens.post_computation`."""
        if self.aper_idx is None:
            self.find_aperture()
        self.calc_foclen_paraxial()
        self.calc_pupil_paraxial()
        self.calc_fov_paraxial()
        self.init_constraints()
