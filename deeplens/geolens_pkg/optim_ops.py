# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Surface operations mixin for GeoLens.

Provides methods for managing optical surface geometry:
    - Aspheric surface conversion and order management
    - Surface pruning (clear aperture sizing)
    - Lens shape correction
"""

import logging

import numpy as np
import torch

from ..config import SPP_CALC
from ..geometric_surface import Aperture, Aspheric, Spheric


class GeoLensSurfOps:
    """Mixin providing surface geometry operations for GeoLens.

    Methods:
        - add_aspheric: Convert a spherical surface to aspheric.
        - increase_aspheric_order: Add higher-order polynomial terms.
        - prune_surf: Size clear apertures by ray tracing.
        - correct_shape: Fix lens geometry during optimisation.
        - insert_element: Insert a flat lens element into an air gap.
        - split_element: Split a glass element into two with an air gap.
        - split_to_doublet: Convert one element into a cemented doublet.
    """

    # ====================================================================================
    # Aspheric surface management
    # ====================================================================================
    @torch.no_grad()
    def add_aspheric(self, surf_idx=None, ai_degree=4):
        """Convert a spherical surface to aspheric for improved aberration correction.

        If ``surf_idx`` is given, converts that specific surface. Otherwise,
        automatically selects the best candidate following established optical
        design principles:

        1. First asphere: placed near the aperture stop (corrects spherical
           aberration).
        2. Subsequent aspheres: placed far from the stop (corrects field-dependent
           aberrations like coma, astigmatism, distortion).
        3. Prefer air-glass interfaces over cemented surfaces.
        4. Among candidates at similar stop-distances, prefer larger semi-diameter
           (higher marginal ray height → more SA contribution).

        The new surface starts with ``k=0`` and all polynomial coefficients at
        zero, so it is initially identical to the original spherical surface.

        Note:
            After calling this method, any existing optimizer is stale.
            Call ``get_optimizer()`` again to include the new parameters.

        Args:
            surf_idx (int or None): Surface index to convert. If ``None``,
                auto-selects the best candidate.
            ai_degree (int): Number of even-order aspheric coefficients
                ``[a2, a4, a6, ...]``. Defaults to 4.

        Returns:
            int: Index of the converted surface.

        Raises:
            IndexError: If ``surf_idx`` is out of range.
            ValueError: If ``surf_idx`` points to a non-Spheric surface, or no
                eligible candidate exists for auto-selection.

        References:
            Design principles from ``research/aspheric_design_principles.md``.
        """
        if surf_idx is not None:
            if surf_idx < 0 or surf_idx >= len(self.surfaces):
                raise IndexError(
                    f"surf_idx={surf_idx} out of range [0, {len(self.surfaces) - 1}]."
                )
            if not isinstance(self.surfaces[surf_idx], Spheric):
                raise ValueError(
                    f"Surface {surf_idx} is {type(self.surfaces[surf_idx]).__name__}, "
                    f"expected Spheric. To add higher-order terms to an existing "
                    f"Aspheric surface, use increase_aspheric_order(surf_idx={surf_idx})."
                )
            self._spheric_to_aspheric(surf_idx, ai_degree)
            logging.info(
                f"Converted surface {surf_idx} from Spheric to Aspheric "
                f"(ai_degree={ai_degree})."
            )
            return surf_idx

        # Auto-select best candidate
        surf_idx = self._find_best_asphere_candidate()
        self._spheric_to_aspheric(surf_idx, ai_degree)
        logging.info(
            f"Auto-selected surface {surf_idx} as best asphere candidate. "
            f"Converted to Aspheric (ai_degree={ai_degree})."
        )
        return surf_idx

    def _find_best_asphere_candidate(self):
        """Select the best Spheric surface to convert to Aspheric.

        Strategy based on classical aspheric placement theory:

        * **No existing aspheres** → nearest to aperture stop (maximises
          spherical aberration correction, analogous to Schmidt corrector).
        * **Asphere(s) already near stop** → farthest from stop (corrects
          field-dependent aberrations), but **excluding outermost surfaces**
          (first/last refractive surfaces are typically large protective
          elements that are impractical and expensive to aspherize).
        * Ties broken by larger semi-diameter (proxy for marginal ray height).
        * Only air-glass interfaces are considered (cemented surfaces excluded).

        Returns:
            int: Surface index of the best candidate.

        Raises:
            ValueError: If no eligible Spheric surfaces exist.
        """
        # Ensure aperture index is known
        if not hasattr(self, "aper_idx") or self.aper_idx is None:
            self.calc_pupil()
        aper_idx = self.aper_idx

        if aper_idx is not None:
            aper_z = self.surfaces[aper_idx].d.item()
        else:
            # No explicit aperture; approximate with system midpoint
            aper_z = (self.surfaces[0].d.item() + self.surfaces[-1].d.item()) / 2.0

        # Identify surfaces belonging to the first element and the last
        # refractive surface.  These are excluded from subsequent asphere
        # selection because:
        #   - The first element is typically a large protective meniscus
        #     (both front and back surfaces are impractical to aspherize).
        #   - The last refractive surface is at the system boundary.
        refractive_indices = [
            i for i, s in enumerate(self.surfaces) if not isinstance(s, Aperture)
        ]
        excluded = set()
        if len(refractive_indices) >= 2:
            # First element = first two refractive surfaces
            excluded.add(refractive_indices[0])
            excluded.add(refractive_indices[1])
            # Last refractive surface
            excluded.add(refractive_indices[-1])

        # Collect candidates: Spheric surfaces at air-glass boundaries
        candidates = []
        for i, surf in enumerate(self.surfaces):
            if not isinstance(surf, Spheric):
                continue
            if not self._is_air_glass_interface(i):
                continue
            dist_from_stop = abs(surf.d.item() - aper_z)
            candidates.append((i, dist_from_stop, surf.r))

        if not candidates:
            raise ValueError(
                "No eligible Spheric surfaces found for aspherization. "
                "All surfaces are either already aspheric, apertures, or cemented."
            )

        # Count existing aspheric surfaces
        num_existing = sum(
            1 for s in self.surfaces if isinstance(s, Aspheric)
        )

        if num_existing == 0:
            # First asphere → nearest to stop, break ties by larger radius
            candidates.sort(key=lambda x: (x[1], -x[2]))
        else:
            # Subsequent → farthest from stop, but exclude outermost surfaces
            # (front/back elements are impractical asphere candidates in
            # camera lens design).  Fall back to full list only if excluding
            # outermost surfaces leaves no candidates.
            inner = [c for c in candidates if c[0] not in excluded]
            if inner:
                candidates = inner
            else:
                logging.warning(
                    "All remaining candidates are outermost surfaces; "
                    "falling back to full candidate list."
                )
            candidates.sort(key=lambda x: (-x[1], -x[2]))

        best_idx = candidates[0][0]
        logging.info(
            f"Asphere candidates (idx, dist_from_stop, radius): "
            f"{[(c[0], round(c[1], 2), round(c[2], 2)) for c in candidates]}. "
            f"Selected surface {best_idx}."
        )
        return best_idx

    def _is_air_glass_interface(self, surf_idx):
        """Check whether a surface sits at an air-glass boundary.

        Looks past adjacent Aperture surfaces when determining the medium
        on the incident side.

        Args:
            surf_idx (int): Surface index.

        Returns:
            bool: ``True`` if exactly one side is air and the other is glass.
        """
        # Material before: walk backwards past aperture surfaces
        mat_before = "air"
        for j in range(surf_idx - 1, -1, -1):
            if not isinstance(self.surfaces[j], Aperture):
                mat_before = self.surfaces[j].mat2.get_name()
                break

        mat_after = self.surfaces[surf_idx].mat2.get_name()

        before_is_air = mat_before == "air"
        after_is_air = mat_after == "air"
        return before_is_air != after_is_air

    def _spheric_to_aspheric(self, surf_idx, ai_degree=4):
        """Replace a Spheric surface with an equivalent Aspheric in-place.

        The new surface has ``k=0`` and all ``ai`` coefficients set to zero,
        preserving the original sag profile exactly.

        Args:
            surf_idx (int): Index of the surface to convert.
            ai_degree (int): Number of even-order polynomial terms.

        Raises:
            ValueError: If the surface is not Spheric.
        """
        surf = self.surfaces[surf_idx]
        if not isinstance(surf, Spheric):
            raise ValueError(
                f"Surface {surf_idx} is {type(surf).__name__}, not Spheric."
            )

        new_surf = Aspheric(
            r=surf.r,
            d=surf.d.item(),
            c=surf.c.item(),
            k=0.0,
            ai=[0.0] * ai_degree,
            mat2=surf.mat2.get_name(),
            pos_xy=[surf.pos_x.item(), surf.pos_y.item()],
            vec_local=surf.vec_local.tolist(),
            is_square=surf.is_square,
            device=surf.d.device,
        )
        self.surfaces[surf_idx] = new_surf

    @torch.no_grad()
    def increase_aspheric_order(self, surf_idx=None, increment=1):
        """Add higher-order polynomial terms to existing Aspheric surfaces.

        Appends ``increment`` additional even-order coefficients (initialised
        to zero). For example, degree 4 ``[a4, a6, a8, a10]`` becomes degree 5
        ``[a4, a6, a8, a10, a12]`` after ``increment=1``.

        Follows the principle of *start low, add incrementally*: increase
        order only when residual higher-order aberrations persist after
        optimisation at the current order.

        Note:
            After calling this method, any existing optimizer is stale.
            Call ``get_optimizer()`` again to include the new parameters.

        Args:
            surf_idx (int or None): Surface index. If ``None``, auto-selects
                the best candidate (see ``_find_best_order_increase_candidate``).
            increment (int): Number of additional coefficients to add.
                Defaults to 1.

        Returns:
            int: Index of the surface whose order was increased.

        Raises:
            IndexError: If ``surf_idx`` is out of range.
            ValueError: If ``surf_idx`` is given but is not Aspheric, if
                no Aspheric surfaces exist when ``surf_idx`` is ``None``,
                or if ``increment`` < 1.
        """
        if increment < 1:
            raise ValueError(f"increment must be >= 1, got {increment}.")
        if surf_idx is not None:
            if surf_idx < 0 or surf_idx >= len(self.surfaces):
                raise IndexError(
                    f"surf_idx={surf_idx} out of range [0, {len(self.surfaces) - 1}]."
                )
        else:
            surf_idx = self._find_best_order_increase_candidate()

        surf = self.surfaces[surf_idx]
        if not isinstance(surf, Aspheric):
            raise ValueError(
                f"Surface {surf_idx} is {type(surf).__name__}, expected Aspheric."
            )
        old_degree = surf.ai_degree
        self._increase_surface_order(surf, increment)
        logging.info(
            f"Surface {surf_idx}: aspheric order {old_degree} -> {surf.ai_degree}."
        )

        return surf_idx

    def _find_best_order_increase_candidate(self):
        """Select the best Aspheric surface to increase polynomial order.

        Follows Principle 5 (*one surface, one term at a time*) from
        aspheric design theory.  Ranking criteria (in priority order):

        1. **Lowest current ``ai_degree``** — the surface with fewest
           polynomial terms benefits most from an additional term.
        2. **Largest semi-diameter ``r``** — proxy for marginal ray height;
           higher-order terms have more leverage on surfaces where the
           beam is widest (Principle 1).
        3. **Highest refractive-index contrast ``Δn``** — larger Δn at
           the interface amplifies the aspheric correction (Principle 2).

        Returns:
            int: Surface index of the best candidate.

        Raises:
            ValueError: If no Aspheric surfaces exist.
        """
        candidates = []
        for i, surf in enumerate(self.surfaces):
            if not isinstance(surf, Aspheric):
                continue

            # Compute Δn at this interface using mat2.n from surface objects
            n_before = 1.0  # default: air
            for j in range(i - 1, -1, -1):
                if not isinstance(self.surfaces[j], Aperture):
                    n_before = self.surfaces[j].mat2.n
                    break
            n_after = surf.mat2.n
            delta_n = abs(n_after - n_before)

            candidates.append((i, surf.ai_degree, float(surf.r), float(delta_n)))

        if not candidates:
            raise ValueError("No Aspheric surfaces found to increase order.")

        # Sort: lowest ai_degree first, then largest r, then highest Δn
        candidates.sort(key=lambda x: (x[1], -x[2], -x[3]))

        best_idx = candidates[0][0]
        logging.info(
            f"Order-increase candidates (idx, degree, r, Δn): "
            f"{[(c[0], c[1], round(c[2], 2), round(c[3], 3)) for c in candidates]}. "
            f"Selected surface {best_idx}."
        )
        return best_idx

    def _increase_surface_order(self, surf, increment=1):
        """Append zero-initialised higher-order coefficients to a surface.

        Updates ``ai_degree``, individual ``ai{2*(j+2)}`` attributes, and the
        ``ai`` tensor consistently.  The ai list starts from the 4th-order
        term (a4): ``[a4, a6, a8, ...]``.

        Args:
            surf (Aspheric): Surface to modify.
            increment (int): Number of additional coefficients.
        """
        device = surf.d.device
        ai_list = [] if surf.ai is None else surf.ai.detach().cpu().tolist()
        ai_list.extend([0.0] * increment)
        surf.ai = torch.tensor(ai_list, device=device)

    # ====================================================================================
    # Surface pruning and shape correction
    # ====================================================================================
    @torch.no_grad()
    def correct_spacing(self, margin=0.01):
        """Separate adjacent surfaces that violate center clearance.

        This is safe inside an active optimization loop: surface positions are
        shifted in-place, preserving optimizer references.  Downstream surfaces
        and the sensor are shifted together so already-corrected upstream gaps
        are not disturbed.
        """
        num_surfs = len(self.surfaces)
        if num_surfs < 2:
            return

        for i in range(num_surfs - 1):
            a = self.surfaces[i]
            b = self.surfaces[i + 1]

            if a.mat2.name == "air":
                center_min = self.air_center_min
            else:
                center_min = self.thick_center_min

            r0 = torch.tensor([0.0], device=self.device)
            center_gap = (
                b.surface_with_offset(r0, 0.0, valid_check=False)
                - a.surface_with_offset(r0, 0.0, valid_check=False)
            ).min()

            center_violation = float((center_min - center_gap).item())
            shift = max(center_violation, 0.0)
            if shift <= 0.0:
                continue

            shift += float(margin)

            for surf in self.surfaces[i + 1:]:
                surf.d.add_(shift)
            if hasattr(self.d_sensor, "add_"):
                self.d_sensor.add_(shift)
            else:
                self.d_sensor += shift

    @torch.no_grad()
    def prune_surf(self, expand_factor=0.05):
        """Prune surfaces to allow all valid rays to go through.

        Determines the clear aperture for each surface by ray tracing, then
        applies a fractional margin and enforces manufacturability constraints
        (edge thickness and air-gap clearance).

        Args:
            expand_factor (float, optional): Fractional expansion applied to
                the ray-traced clear aperture radius.  Auto-selected if None:
                5 % for all lenses.
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # ------------------------------------------------------------------
        # 1. Temporarily remove radius limits so the trace is unclipped
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. Trace rays at full FoV to find maximum ray height per surface
        # ------------------------------------------------------------------
        assert self.rfov is not None, "prune_surf() requires self.rfov."
        fov_deg = self.rfov * 180 / torch.pi
        num_fov_samples = 16
        fov_y = torch.linspace(0.0, fov_deg, num_fov_samples, device=self.device)
        ray = self.sample_from_fov(
            fov_x=[0.0], fov_y=fov_y, num_rays=SPP_CALC, scale_pupil=1.0
        )
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(
            ray_o_record, nan=0.0, posinf=0.0, neginf=0.0
        )
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]
        

        # ------------------------------------------------------------------
        # 3. Propose new radii (not yet committed to surfaces).
        # ------------------------------------------------------------------
        proposed_r = [float(self.surfaces[i].r) for i in range(num_surfs)]
        for i in surface_range:
            if surf_r_max[i] > 0:
                base = float(surf_r_max[i].item())
            else:
                base = float(self.surfaces[i].r)

            r_expand = max(min(base * expand_factor, 2.0), 0.1)
            proposed_r[i] = base + r_expand

        # ------------------------------------------------------------------
        # 4. Edge-clearance pass — proactively cap adjacent pairs so the
        #    committed radii never produce self-intersection at the edge.
        #    Thresholds match loss_bound. The cap uses the common
        #    clear-aperture overlap between adjacent surfaces so one surface is
        #    not pruned against regions where the neighbour has already been
        #    apertured away. Aperture surfaces are skipped; the stop size is an
        #    optical specification and should not be changed by pruning. The cap
        #    is computed via a single vectorized grid search rather than a
        #    serial binary loop.
        #
        #    Each pruned surface is checked against both neighbours. The
        #    previous implementation only capped surface i against i + 1,
        #    which allowed surface i to expand into i - 1 and later crash
        #    tracing/optimization.
        # ------------------------------------------------------------------
        min_radius_floor = 0.1  # mm — guard against update_r(0) killing a surface
        n_cand = 64
        n_edge = 64
        r_frac = torch.linspace(0.5, 1.0, n_edge, device=self.device)
        cand_frac = torch.linspace(1.0 / n_cand, 1.0, n_cand, device=self.device)

        def cap_radius_against_pair(cap_idx, prev_idx, next_idx):
            prev_surf = self.surfaces[prev_idx]
            next_surf = self.surfaces[next_idx]
            if isinstance(prev_surf, Aperture) or isinstance(next_surf, Aperture):
                return
            if isinstance(self.surfaces[cap_idx], Aperture):
                return

            edge_min = 0.1 # mm
            r_check = proposed_r[cap_idx]
            if r_check <= 0:
                return

            other_idx = next_idx if cap_idx == prev_idx else prev_idx
            other_r = proposed_r[other_idx]
            overlap_r = min(r_check, other_r)
            if overlap_r <= 0:
                return

            required_r = max(
                float(surf_r_max[cap_idx].item()),
                min_radius_floor,
            )

            # Cheap 1D probe over the shared aperture — most pairs clear and
            # we short-circuit. Do not evaluate either surface beyond the
            # other surface's physical radius.
            r_pts = r_frac * overlap_r
            z_prev = prev_surf.surface_with_offset(r_pts, 0.0, valid_check=False)
            z_next = next_surf.surface_with_offset(r_pts, 0.0, valid_check=False)
            gap = float((z_next - z_prev).min().item())
            if gap >= edge_min:
                return

            # Vectorized cap: evaluate gap for 64 candidate radii in one pass.
            cand_r = cand_frac * r_check
            cand_overlap_r = torch.minimum(
                cand_r, torch.tensor(other_r, device=self.device)
            )
            r_grid = cand_overlap_r.unsqueeze(1) * r_frac.unsqueeze(0)
            z_prev_grid = prev_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            z_next_grid = next_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            per_cand_gap = (z_next_grid - z_prev_grid).min(dim=-1).values
            valid_mask = per_cand_gap >= edge_min
            if not bool(valid_mask.any()):
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): no candidate "
                    f"radius satisfies edge_min {edge_min:.3f} mm at "
                    f"r_check {r_check:.3f} mm (possible sag crossing near "
                    f"axis). Reducing surface {cap_idx} to the ray-required radius "
                    f"{required_r:.3f} mm, but edge clearance may remain "
                    f"violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = float((cand_frac[valid_mask].max() * r_check).item())
            if r_safe < required_r:
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): ray-required "
                    f"radius {required_r:.3f} mm exceeds edge-clearance-safe "
                    f"radius {r_safe:.3f} mm for edge_min {edge_min:.3f} mm. "
                    f"Reducing surface {cap_idx} to the ray-required radius; edge "
                    f"clearance may remain violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = max(r_safe, min_radius_floor)
            if proposed_r[cap_idx] > r_safe:
                proposed_r[cap_idx] = r_safe

        for i in surface_range:
            if i > 0:
                cap_radius_against_pair(i, i - 1, i)
            if i < num_surfs - 1:
                cap_radius_against_pair(i, i, i + 1)

        # ------------------------------------------------------------------
        # 4b. Commit the capped proposed radii to the surfaces.
        # ------------------------------------------------------------------
        for i in surface_range:
            if proposed_r[i] > 0:
                self.surfaces[i].update_r(proposed_r[i])

    @torch.no_grad()
    def anchor_first_surface(self):
        """Shift the whole lens so the first surface sits at z = 0.

        Modifies ``d`` in-place so optimizer references are preserved.
        Safe to call inside a training loop.
        """
        d0 = self.surfaces[0].d.item()
        if abs(d0) < 1e-12:
            return
        for surf in self.surfaces:
            surf.d.add_(-d0)
        self.d_sensor -= d0

    @torch.no_grad()
    def set_front_aperture_gap(self, d0=0.1):
        """Set the air gap after a front aperture stop.

        If the aperture is the first surface, keep the aperture fixed and shift
        every downstream surface plus the sensor so surface 1 sits ``d0`` mm
        after the aperture.  This preserves the rest of the lens group's
        internal axial spacings.
        """
        d0 = float(d0)
        if self.aper_idx != 0 or len(self.surfaces) < 2:
            return
        if not isinstance(self.surfaces[0], Aperture):
            return
        if d0 < 0.0:
            raise ValueError(f"front aperture gap must be non-negative, got {d0}")

        current_gap = (self.surfaces[1].d - self.surfaces[0].d).item()
        shift = d0 - current_gap
        if abs(shift) < 1e-12:
            return

        for surf in self.surfaces[1:]:
            surf.d.add_(shift)
        self.d_sensor += shift

    @torch.no_grad()
    def correct_shape(self, expand_factor=0.05, front_aperture_gap=0.1):
        """Finalize lens geometry after optimization.

        Anchors the first surface to ``z = 0`` and prunes clear apertures.
        This uses in-place updates for axial shifts, so optimizer references are
        preserved.

        Args:
            expand_factor (float, optional): Height expansion factor for surface pruning.
                If None, uses the default from :meth:`prune_surf`.
            front_aperture_gap (float, optional): If the aperture is the first
                surface, set the gap from the aperture to the next surface to
                this value [mm]. Defaults to 0.1.
        """
        # Rule 1: move the first surface to z = 0.
        self.anchor_first_surface()

        # Rule 2: keep a minimum front-stop air gap for mobile-style lenses.
        self.set_front_aperture_gap(d0=front_aperture_gap)

        # Rule 3: repair axial spacing before pruning apertures.
        # self.correct_spacing()

        # Rule 4: prune surfaces to fit valid rays, with margins for manufacturability.
        self.prune_surf(expand_factor=expand_factor)

    # ====================================================================================
    # Element insertion
    # ====================================================================================
    @torch.no_grad()
    def insert_element(self, after_surf_idx, glass="bk7", thickness=None):
        """Insert a new flat lens element after a given surface.

        The inserted element is a plane-parallel plate made of two flat
        spherical surfaces (``c=0``). Air spacing and thickness are derived
        from the local gap if ``thickness`` is not provided.

        Args:
            after_surf_idx (int): Insert the new element immediately after this
                surface index. The referenced surface must exit into air.
            glass (str, optional): Glass name for the new element.
                Defaults to ``"bk7"``.
            thickness (float | None, optional): Center thickness [mm]. If
                ``None``, uses the median thickness of existing glass elements,
                clamped to ``[0.3, 5.0]`` mm.

        Returns:
            tuple[int, int]: ``(front_idx, back_idx)`` of the newly inserted
            front and back surfaces.

        Raises:
            IndexError: If ``after_surf_idx`` is outside the surface list.
            ValueError: If the insertion point is not an air gap.
        """
        num_surfs = len(self.surfaces)
        if after_surf_idx < 0 or after_surf_idx >= num_surfs:
            raise IndexError(
                f"after_surf_idx={after_surf_idx} out of range [0, {num_surfs - 1}]."
            )

        ref_surf = self.surfaces[after_surf_idx]
        if isinstance(ref_surf, Aperture):
            mat_name = "air"
        else:
            mat_name = ref_surf.mat2.get_name()
        if mat_name != "air":
            raise ValueError(
                f"Surface {after_surf_idx} exits into {mat_name!r}, not air. "
                f"Can only insert an element in an air gap."
            )

        if thickness is None:
            glass_thicknesses = []
            for i in range(num_surfs - 1):
                surf = self.surfaces[i]
                if isinstance(surf, Aperture):
                    continue
                if surf.mat2.get_name() != "air":
                    d_this = surf.d.item()
                    d_next = self.surfaces[i + 1].d.item()
                    t = d_next - d_this
                    if t > 0:
                        glass_thicknesses.append(t)

            if glass_thicknesses:
                thickness = float(np.median(glass_thicknesses))
            else:
                thickness = 1.0
            thickness = max(0.3, min(thickness, 5.0))

        next_idx = after_surf_idx + 1
        d_left = ref_surf.d.item()
        if next_idx < num_surfs:
            d_right = self.surfaces[next_idx].d.item()
        else:
            d_right = (
                self.d_sensor.item()
                if hasattr(self.d_sensor, "item")
                else float(self.d_sensor)
            )

        original_gap = d_right - d_left
        min_air = 0.1
        available_for_air = original_gap - thickness
        if available_for_air < 2 * min_air:
            air_before = min_air
            air_after = min_air
            needed_span = thickness + 2 * min_air
            overflow = needed_span - original_gap
        else:
            air_before = available_for_air / 2
            air_after = available_for_air / 2
            overflow = 0.0

        d_front = d_left + air_before
        d_back = d_front + thickness

        if overflow > 0:
            for surf in self.surfaces[next_idx:]:
                surf.d = surf.d.item() + overflow
            if hasattr(self.d_sensor, "data"):
                self.d_sensor.data.fill_(self.d_sensor.item() + overflow)
            else:
                self.d_sensor = self.d_sensor + overflow

        r_new = ref_surf.r
        if next_idx < num_surfs:
            r_new = min(r_new, self.surfaces[next_idx].r)
        r_new *= 0.9

        device = ref_surf.d.device
        front_surf = Spheric(c=0.0, r=r_new, d=d_front, mat2=glass, device=device)
        back_surf = Spheric(c=0.0, r=r_new, d=d_back, mat2="air", device=device)

        insert_pos = after_surf_idx + 1
        self.surfaces.insert(insert_pos, front_surf)
        self.surfaces.insert(insert_pos + 1, back_surf)

        if self.aper_idx is not None and self.aper_idx >= insert_pos:
            self.aper_idx += 2

        front_idx = insert_pos
        back_idx = insert_pos + 1
        logging.info(
            f"Inserted element after surface {after_surf_idx}: "
            f"front={front_idx} (d={d_front:.3f}), "
            f"back={back_idx} (d={d_back:.3f}), "
            f"glass={glass!r}, thickness={thickness:.3f} mm, "
            f"air_before={air_before:.3f} mm, air_after={air_after:.3f} mm."
        )
        return front_idx, back_idx

    @torch.no_grad()
    def split_element(self, surf_idx, air_gap=0.1):
        """Split a glass element into two thinner elements with an air gap.

        Inserts two flat surfaces inside an existing glass element so the
        original glass volume becomes ``glass | air | glass`` while keeping
        the total track length unchanged.

        Args:
            surf_idx (int): Front surface of the glass element to split.
            air_gap (float, optional): Width of the new air gap [mm].
                Defaults to ``0.1``.

        Returns:
            tuple[int, int]: Indices of the inserted back/front split surfaces.

        Raises:
            IndexError: If ``surf_idx`` is out of range or has no successor.
            ValueError: If the target is not the front of a glass element, or
                if the element is too thin for the requested gap.
        """
        num_surfs = len(self.surfaces)
        if surf_idx < 0 or surf_idx >= num_surfs - 1:
            raise IndexError(
                f"surf_idx={surf_idx} must be in [0, {num_surfs - 2}] "
                f"so a back surface exists."
            )

        front_surf = self.surfaces[surf_idx]
        if isinstance(front_surf, Aperture):
            raise ValueError(
                f"Surface {surf_idx} is an Aperture; not a glass element."
            )
        mat_name = front_surf.mat2.get_name()
        if mat_name == "air":
            raise ValueError(
                f"Surface {surf_idx} exits into air, not glass. "
                f"split_element must be called on the front of a glass element."
            )

        back_surf = self.surfaces[surf_idx + 1]
        if isinstance(back_surf, Aperture):
            raise ValueError(
                f"Surface {surf_idx + 1} is an Aperture; element has no "
                f"clear back surface."
            )

        d_front = front_surf.d.item()
        d_back = back_surf.d.item()
        thickness = d_back - d_front
        if thickness <= 2 * air_gap:
            raise ValueError(
                f"Element thickness {thickness:.3f} mm is too thin to split "
                f"with air_gap={air_gap:.3f} mm. Need thickness > 2 * air_gap."
            )

        d_mid_back = d_front + (thickness - air_gap) / 2
        d_mid_front = d_mid_back + air_gap

        r_ref = min(front_surf.r, back_surf.r)
        device = front_surf.d.device

        mid_back = Spheric(c=0.0, r=r_ref, d=d_mid_back, mat2="air", device=device)
        mid_front = Spheric(c=0.0, r=r_ref, d=d_mid_front, mat2=mat_name, device=device)

        insert_pos = surf_idx + 1
        self.surfaces.insert(insert_pos, mid_back)
        self.surfaces.insert(insert_pos + 1, mid_front)

        if self.aper_idx is not None and self.aper_idx >= insert_pos:
            self.aper_idx += 2

        mid_back_idx = insert_pos
        mid_front_idx = insert_pos + 1
        logging.info(
            f"Split element at surface {surf_idx}: "
            f"mid_back={mid_back_idx} (d={d_mid_back:.3f}), "
            f"mid_front={mid_front_idx} (d={d_mid_front:.3f}), "
            f"air_gap={air_gap:.3f} mm."
        )
        return mid_back_idx, mid_front_idx

    @torch.no_grad()
    def split_to_doublet(self, surf_idx, second_glass="sf2"):
        """Convert a single glass element into a cemented doublet.

        Inserts one flat cemented interface at the midpoint of the target
        element so the original glass becomes two touching sub-elements with
        different materials and no air gap.

        Args:
            surf_idx (int): Front surface of the glass element to convert.
            second_glass (str, optional): Glass name for the rear sub-element.
                Defaults to ``"sf2"``.

        Returns:
            int: Index of the newly inserted cemented interface.

        Raises:
            IndexError: If ``surf_idx`` is out of range or has no successor.
            ValueError: If the target is not the front of a glass element, or
                if the element has non-positive thickness.
        """
        num_surfs = len(self.surfaces)
        if surf_idx < 0 or surf_idx >= num_surfs - 1:
            raise IndexError(
                f"surf_idx={surf_idx} must be in [0, {num_surfs - 2}] "
                f"so a back surface exists."
            )

        front_surf = self.surfaces[surf_idx]
        if isinstance(front_surf, Aperture):
            raise ValueError(
                f"Surface {surf_idx} is an Aperture; not a glass element."
            )
        first_glass = front_surf.mat2.get_name()
        if first_glass == "air":
            raise ValueError(
                f"Surface {surf_idx} exits into air, not glass. "
                f"split_to_doublet must be called on the front of a glass element."
            )

        back_surf = self.surfaces[surf_idx + 1]
        if isinstance(back_surf, Aperture):
            raise ValueError(
                f"Surface {surf_idx + 1} is an Aperture; element has no "
                f"clear back surface."
            )

        d_front = front_surf.d.item()
        d_back = back_surf.d.item()
        thickness = d_back - d_front
        if thickness <= 0:
            raise ValueError(
                f"Element thickness {thickness:.3f} mm is non-positive; "
                f"cannot split into a doublet."
            )

        d_mid = d_front + thickness / 2
        r_ref = min(front_surf.r, back_surf.r)
        device = front_surf.d.device

        cement_surf = Spheric(
            c=0.0, r=r_ref, d=d_mid, mat2=second_glass, device=device
        )

        insert_pos = surf_idx + 1
        self.surfaces.insert(insert_pos, cement_surf)

        if self.aper_idx is not None and self.aper_idx >= insert_pos:
            self.aper_idx += 1

        cement_idx = insert_pos
        logging.info(
            f"Split element at surface {surf_idx} into doublet: "
            f"cement interface={cement_idx} (d={d_mid:.3f}), "
            f"first={first_glass!r}, second={second_glass!r}."
        )
        return cement_idx

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match lens materials to a glass catalog.

        Args:
            mat_table (str, optional): Glass catalog name. Common options include
                'CDGM', 'SCHOTT', 'OHARA'. Defaults to 'CDGM'.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)
