# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric surface.

The ``ai`` coefficient list starts from the 4th-order term (a4) by default.
Legacy JSON files that include a 2nd-order term (a2) are loaded via the
``use_ai2`` flag in ``init_from_dict``.  When present, ``a2`` is stored
separately and included in the sag computation but is **not** optimised
(it competes with the base curvature ``c``).

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import logging
import numpy as np
import torch

from .base import EPSILON, Surface

logger = logging.getLogger(__name__)


class Aspheric(Surface):
    r"""Even-order aspheric surface.

    The sag function is:

    .. math::

        z(\rho) = \frac{c\,\rho^2}{1 + \sqrt{1-(1+k)c^2\rho^2}}
                 + \sum_{i=2}^{n} a_{2i}\,\rho^{2i},
        \quad \rho^2 = x^2 + y^2

    The polynomial starts at the 4th-order term (a4) because the 2nd-order
    term competes with the base curvature ``c``.

    All coefficients ``c``, ``k``, and ``ai`` are differentiable torch
    tensors so they can be optimised with gradient descent.

    Attributes:
        c (torch.Tensor): Base curvature [1/mm].
        k (torch.Tensor): Conic constant.
        ai2 (torch.Tensor or None): 2nd-order aspheric coefficient (legacy).
        ai (torch.Tensor): Even-order aspheric coefficients
            ``[a4, a6, a8, ...]``.
    """

    def __init__(
        self,
        r,
        d,
        c,
        k,
        ai,
        mat2,
        ai2=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize an aspheric surface.

        Args:
            r (float): Aperture radius [mm].
            d (float): Axial vertex position [mm].
            c (float): Base curvature ``1/R`` [1/mm].
            k (float): Conic constant (``0`` = sphere, ``-1`` = paraboloid).
            ai (list[float] or None): Even-order aspheric coefficients
                starting from the 4th-order term: ``[a4, a6, a8, ...]``.
                Pass ``None`` or an empty list for a pure conic.
            mat2 (str or Material): Material on the transmission side.
            ai2 (float or None, optional): 2nd-order aspheric coefficient
                from legacy data.  Included in sag but not optimised.
                Defaults to ``None``.
            pos_xy (list[float], optional): Lateral offset ``[x, y]`` [mm].
                Defaults to ``[0.0, 0.0]``.
            vec_local (list[float], optional): Local normal direction.
                Defaults to ``[0.0, 0.0, 1.0]``.
            is_square (bool, optional): Square aperture flag.
                Defaults to ``False``.
            device (str, optional): Compute device. Defaults to ``"cpu"``.
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        self.c = torch.tensor(float(c))
        self.k = torch.tensor(float(k))

        # 2nd-order coefficient (legacy, not optimised)
        if ai2 is not None:
            self.ai2 = torch.tensor(float(ai2))
        else:
            self.ai2 = None

        self.ai_degree = 0
        self._init_ai(ai)

        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        ai = surf_dict.get("ai", [])
        ai2_val = None

        # Backward compatibility: old format includes a2 as first element.
        # New files written by this code set use_ai2 explicitly.
        if surf_dict.get("use_ai2", True) and len(ai) > 0:
            if "use_ai2" not in surf_dict:
                logger.warning(
                    "Surface dict lacks 'use_ai2'; assuming ai[0]=%.4g is the "
                    "2nd-order coefficient (legacy format).", ai[0]
                )
            ai2_val = ai[0]  # Extract the a2 coefficient
            ai = ai[1:]      # Remaining: [a4, a6, a8, ...]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            # ``k`` is optional in some converted surface payloads; default
            # to 0 so missing conic data behaves like a pure sphere.
            k=surf_dict.get("k", 0.0),
            ai=ai,
            ai2=ai2_val,
            mat2=surf_dict["mat2"],
        )

    def _init_ai(self, values):
        """Store aspheric coefficients as plain tensors ``_ai4``, ``_ai6``, ..."""
        old_degree = getattr(self, "ai_degree", 0)
        for j in range(old_degree):
            attr = f"_ai{2 * (j + 2)}"
            if hasattr(self, attr):
                delattr(self, attr)

        if values is None or len(values) == 0:
            self.ai_degree = 0
            return

        if torch.is_tensor(values):
            coeffs = values.detach().reshape(-1).tolist()
        else:
            coeffs = [float(v) for v in values]

        for j, coeff in enumerate(coeffs):
            setattr(self, f"_ai{2 * (j + 2)}", torch.tensor(float(coeff)))
        self.ai_degree = len(coeffs)

    def update_r(self, r):
        """Update surface radius."""
        super().update_r(r)

    # =======================================
    # Surface math (sag, derivatives)
    # =======================================

    def _get_curvature_params(self):
        """Get physical curvature params, adding tolerancing error if active."""
        c = self.c
        k = self.k
        if self.tolerancing:
            c = c + self.c_error
            k = k + self.k_error
        return c, k

    def _sag(self, x, y):
        """Compute surface sag (height) z = sag(x, y).

        The aspheric surface is defined as:
            z = r²c / (1 + sqrt(1 - (1+k)r²c²)) + [a2*r²] + Σ a_{2i} * r^{2i}

        where r² = x² + y², c is curvature, k is conic constant, and ai are
        the aspheric coefficients (ai4, ai6, ai8, ...).
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        # Clamp the conic radicand to stay strictly positive: when
        # (1+k)*c^2*r^2 >= 1 the analytic surface is undefined and the bare
        # sqrt would emit NaN that propagates through every gradient. Clamping
        # keeps sag finite for out-of-range rays so the optimizer can recover.
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        total_surface = r2 * c / (1 + torch.sqrt(sf_arg))

        # Legacy a2 term: a2 * r²
        if self.ai2 is not None:
            total_surface = total_surface + self.ai2 * r2

        # Aspheric polynomial: ai4*r⁴ + ai6*r⁶ + ai8*r⁸ + ...
        r_pow = r2 * r2  # starts at r^4
        for i in range(self.ai_degree):
            total_surface = total_surface + getattr(self, f"_ai{2 * (i + 2)}") * r_pow
            r_pow = r_pow * r2

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order height derivatives df/dx and df/dy.

        For the aspheric polynomial Σ a_{2i} * r^{2i} (i >= 2), the derivative
        w.r.t. r² is Σ i * a_{2i} * r^{2(i-1)}, i.e.: 2*a4*r² + 3*a6*r⁴ + ...
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        sf = torch.sqrt(sf_arg)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        # d(a2*r²)/dr² = a2
        if self.ai2 is not None:
            dsdr2 = dsdr2 + self.ai2

        # Derivative of aspheric polynomial w.r.t. r²: 2*ai4*r² + 3*ai6*r⁴ + ...
        r_pow = r2
        for i in range(self.ai_degree):
            order = i + 2  # 2, 3, 4, ...
            dsdr2 = dsdr2 + order * getattr(self, f"_ai{2 * (i + 2)}") * r_pow
            r_pow = r_pow * r2

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined.

        Fully tensorized (no Python branch on the tensor value of ``k``) so
        the function is safe to trace through ``torch.compile``. When
        ``k <= -1`` the conic has no real boundary, so every point is
        treated as valid.
        """
        c, k = self._get_curvature_params()
        one_plus_k = 1 + k
        # Avoid division by zero / negative when computing the limit; the
        # bogus value is masked out by the where below.
        safe = torch.where(
            one_plus_k > 0, one_plus_k, torch.ones_like(one_plus_k)
        )
        limit_sq = 1.0 / (c * c * safe)
        inside = (x * x + y * y) < limit_sq
        return torch.where(one_plus_k > 0, inside, torch.ones_like(inside))

    def max_height(self):
        """Maximum valid height."""
        c, k = self._get_curvature_params()
        if k > -1:
            c_sq = (c**2).clamp(min=EPSILON)
            return torch.sqrt(1 / (k + 1) / c_sq).item() - 0.001
        return 10e3

    # =======================================
    # Optimization
    # =======================================

    def get_optimizer_params(
        self,
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        optim_mat=False,
    ):
        """Get optimizer parameter groups for the asphere.

        Args:
            lrs (list, optional): learning rates for ``[d, c, k, ai]``.
            optim_mat (bool, optional): whether to optimize material.
                Defaults to False.
        """
        params = []

        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        if self.ai_degree > 0:
            lr_base = lrs[3] if len(lrs) > 3 else 1e-4
            r_norm = max(self.r, 1e-6)
            for i in range(self.ai_degree):
                order = 2 * (i + 2)
                ai_t = getattr(self, f"_ai{order}")
                ai_t.requires_grad_(True)
                lr_ai = lr_base / (r_norm ** order)
                params.append({"params": [ai_t], "lr": lr_ai})

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # Tolerancing
    # =======================================

    @torch.no_grad()
    def init_tolerance(self, tolerance_params=None):
        """Perturb the surface with some tolerance.

        Args:
            tolerance_params (dict): Tolerance for surface parameters.

        References:
            [1] https://www.edmundoptics.com/capabilities/precision-optics/capabilities/aspheric-lenses/
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/all-about-aspheric-lenses/?srsltid=AfmBOoon8AUXVALojol2s5K20gQk7W1qUisc6cE4WzZp3ATFY5T1pK8q
        """
        super().init_tolerance(tolerance_params)
        if tolerance_params is None:
            tolerance_params = {}
        self.c_tole = tolerance_params.get("c_tole", 0.001)
        self.k_tole = tolerance_params.get("k_tole", 0.001)
        self.c_error = 0.0
        self.k_error = 0.0

    def sample_tolerance(self):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)

    def zero_tolerance(self):
        """Zero tolerance."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0

    def sensitivity_score(self):
        """Tolerance squared sum for d, c, k."""
        score_dict = super().sensitivity_score()
        idx = getattr(self, "surf_idx", id(self))

        if self.c.grad is not None:
            c_grad = self.c.grad
            score_dict[f"surf{idx}_c_grad"] = round(c_grad.item(), 6)
            score_dict[f"surf{idx}_c_score"] = round(
                (self.c_tole**2 * c_grad**2).item(), 6
            )
        if self.k.grad is not None:
            k_grad = self.k.grad
            score_dict[f"surf{idx}_k_grad"] = round(k_grad.item(), 6)
            score_dict[f"surf{idx}_k_score"] = round(
                (self.k_tole**2 * k_grad**2).item(), 6
            )
        return score_dict

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        c, k = self._get_curvature_params()
        c_val = c.item()
        k_val = k.item()

        has_ai2 = self.ai2 is not None
        surf_dict = {
            "type": "Aspheric",
            "r": round(self.r, 4),
            "(c)": round(c_val, 4),
            "roc": round(1 / c_val, 4) if c_val != 0.0 else float("inf"),
            "d": round(self.d.item(), 4),
            "k": round(k_val, 4),
            "ai": [],
            "use_ai2": has_ai2,
            "mat2": self.mat2.get_name(),
        }

        # Prepend a2 to ai list if present (ai2 key is informational;
        # deserialization reads ai[0] when use_ai2=True)
        if has_ai2:
            surf_dict["ai2"] = float(format(self.ai2.item(), ".6e"))
            surf_dict["ai"].append(float(format(self.ai2.item(), ".6e")))

        for i in range(self.ai_degree):
            order = i + 2
            coeff = getattr(self, f"_ai{2 * (i + 2)}")
            surf_dict[f"(ai{2 * order})"] = float(format(coeff.item(), ".6e"))
            surf_dict["ai"].append(float(format(coeff.item(), ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        c, k = self._get_curvature_params()
        c_val = c.item()

        assert c_val != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai_degree > 0 or k.item() != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )

        # Collect absolute ai values, PARM 1 = a2, PARM 2+ = a4, a6, ...
        abs_ai = [self.ai2.item() if self.ai2 is not None else 0.0]
        for i in range(self.ai_degree):
            abs_ai.append(getattr(self, f"_ai{2 * (i + 2)}").item())

        # Pad with zeros for Zemax PARM format (needs 8 PARMs: a2..a16)
        while len(abs_ai) < 8:
            abs_ai.append(0.0)

        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {c_val}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {c_val}
    DISZ {d_next.item()}
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        return zmx_str
