# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Classical optical performance evaluation for geometric lens systems.

This module provides a mixin class ``GeoLensEval`` that adds Zemax-equivalent
optical evaluation capabilities to ``GeoLens``.  Every metric is computed via
geometric ray tracing: rays are sampled from object space, propagated through
all lens surfaces (refraction + clipping), and analyzed at the sensor plane.

Coordinate convention (shared with the rest of DeepLens):
    - **z-axis**: optical axis, light travels in +z direction.
    - **y-axis**: meridional (tangential) plane.
    - **x-axis**: sagittal plane.
    - Sensor plane is at ``z = self.d_sensor``.

Key dependencies consumed from the parent ``GeoLens`` instance:
    - ``self.sample_radial_rays()``, ``self.sample_grid_rays()``: ray sampling.
    - ``self.trace(ray)``, ``self.trace2sensor(ray)``: sequential ray tracing.
    - ``self.psf()``, ``self.psf_rgb()``: point-spread-function computation.
    - ``self.render()``: image-plane rendering via ray tracing or PSF convolution.
    - ``self.d_sensor``, ``self.sensor_size``, ``self.pixel_size``, ``self.rfov_eff``,
      ``self.foclen``, ``self.device``: lens geometry attributes.

Functions:
    Spot Diagram:
        spot_points: Core ray-tracing function — samples rays from physical
            object points, traces through the lens, returns sensor positions.
            Shared by draw_spot_radial, draw_spot_map, and rms_map.
        draw_spot_radial: Spot diagrams at evenly-spaced field angles along a
            chosen direction (meridional/sagittal/diagonal), with RGB overlay.
        draw_spot_map: 2-D grid of spot diagrams across the full field of view.

    RMS Spot Error:
        rms_map_rgb: Per-pixel RMS spot radius for R/G/B, referenced to the
            green-channel centroid (chromatic shift included).
        rms_map: Per-pixel RMS spot radius for a single wavelength, referenced
            to its own centroid.

    Distortion:
        calc_distortion_map: 2-D grid of actual-vs-ideal image positions.
        draw_distortion_map: Scatter plot of the distortion grid.
        distortion_center: Normalized centroid positions for arbitrary object
            points (used for warp/unwarp).

    MTF (Modulation Transfer Function):
        mtf: Geometric MTF (tangential + sagittal) at a single field position
            via PSF → FFT.
        psf2mtf: Static utility converting a 2-D PSF array to tangential and
            sagittal MTF curves.
        draw_mtf: Grid of tangential MTF curves for multiple depths, FOVs, and
            RGB wavelengths.

    Field Curvature:
        field_curvature: Placeholder for field-curvature computation.
        draw_field_curvature: Best-focus defocus vs field angle for RGB, found
            by a vectorized defocus sweep with parabolic interpolation.

    Vignetting:
        vignetting: Fractional ray-throughput map across the field.
        draw_vignetting: Grayscale image of relative illumination.

    Wavefront & Aberration:
        wavefront_error: OPD at exit pupil for a given field position (RMS, PV, Strehl).
        rms_wavefront_error: Scalar RMS wavefront error convenience wrapper.
        draw_wavefront_error: OPD maps at multiple field positions.

    Comprehensive Analysis:
        analysis_spot: RMS and geometric spot radii averaged over RGB at
            multiple field positions.
        analysis_rendering: Render a test image through the lens and report
            PSNR / SSIM.
        analysis: One-call entry point that chains layout drawing, spot
            analysis, and (optionally) full evaluation + rendering.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from ..config import (
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
    SPP_RENDER,
)
from ..imgsim import assign_points_to_pixels
from ..ray import Ray

# RGB color definitions for wavelength visualization
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    """Mixin that adds classical optical evaluation methods to ``GeoLens``.

    This class is **never instantiated on its own**.  It is mixed into
    ``GeoLens`` via multiple inheritance, so every method can access lens
    geometry (``self.d_sensor``, ``self.rfov_eff``, …) and ray-tracing routines
    (``self.trace()``, ``self.trace2sensor()``, …) directly through ``self``.

    All evaluation functions follow the same pattern:
        1. Sample rays from object space (parallel / grid / radial).
        2. Trace rays through the lens (``self.trace`` or ``self.trace2sensor``).
        3. Analyze ray positions / directions at the sensor plane.
        4. Optionally produce a matplotlib figure saved to disk.

    Results are accuracy-aligned with Zemax OpticStudio for the same lens
    prescriptions and ray-sampling densities.

    Attributes consumed from ``GeoLens`` (via ``self``):
        d_sensor (float): Axial position of the sensor plane (mm).
        sensor_size (tuple[float, float]): Sensor (width, height) in mm.
        pixel_size (float): Pixel pitch in mm.
        sensor_res (tuple[int, int]): Sensor resolution (H, W) in pixels.
        rfov_eff (float): Effective half field-of-view in **radians** (pinhole model).
        rfov (float): Half field-of-view in **radians** (ray-traced).
        foclen (float): Equivalent focal length in mm.
        fnum (float): F-number.
        aper_idx (int): Index of the aperture stop surface.
        device (torch.device): Compute device (CPU / CUDA).
    """

    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def spot_points(self, points, num_rays=SPP_PSF, wvln=None):
        """Trace rays from object points to sensor and return the traced Ray.

        Samples rays from each physical object point toward the entrance pupil,
        traces through all lens surfaces (refraction + clipping), and returns
        the resulting Ray object on the sensor plane.

        This is the shared computational core for spot diagrams
        (``draw_spot_radial``, ``draw_spot_map``) and RMS error maps
        (``rms_map``, ``rms_map_rgb``).

        Algorithm:
            1. ``self.sample_from_points(points, num_rays, wvln)`` generates a
               fan of ``num_rays`` rays per object point, aimed at the entrance
               pupil.
            2. ``self.trace2sensor()`` propagates through all surfaces and
               clips vignetted rays.

        Args:
            points (torch.Tensor): Physical 3D object-space coordinates with
                shape ``[..., 3]`` (mm).  Supported layouts:
                - ``[3]`` — single point.
                - ``[N, 3]`` — N points (e.g. radial field positions).
                - ``[H, W, 3]`` — 2-D field grid.
                Generated by ``self.point_source_grid(normalized=False)`` for
                grid sampling, or ``self.point_source_radial(normalized=False)``
                for radial sampling.
            num_rays (int): Number of rays sampled per object point.
                Defaults to ``SPP_PSF``.
            wvln (float): Wavelength in micrometers.
                Defaults to ``DEFAULT_WAVE``.

        Returns:
            Ray: Traced ray on the sensor plane, with shape
                ``[..., num_rays, 3]`` for positions and ``[..., num_rays]``
                for validity mask. Use ``ray.o[..., :2]`` for transverse
                positions and ``ray.is_valid`` for the validity mask.
                ``ray.centroid()`` gives the weighted centroid.
        """
        wvln = self.wvln_primary if wvln is None else wvln
        ray = self.sample_from_points(points=points, num_rays=num_rays, wvln=wvln)
        return self.trace2sensor(ray)

    @torch.no_grad()
    def draw_spot_radial(
        self,
        save_name="./lens_spot_radial.png",
        num_fov=6,
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        direction="-y",
        show=False,
    ):
        """Draw spot diagrams at evenly-spaced field angles along a chosen direction.

        A *spot diagram* visualizes the transverse ray-intercept distribution on
        the sensor plane for a point source at a given field angle and depth.
        It reveals the combined effect of all aberrations (spherical, coma,
        astigmatism, field curvature, chromatic, …).

        Algorithm:
            For each wavelength in ``wvln_list``:
                1. ``self.point_source_radial(direction, normalized=False)``
                   generates physical object-space points along the chosen
                   direction.
                2. ``self.spot_points()`` samples rays and traces to sensor.
                3. Valid ray (x, y) positions are scatter-plotted per subplot.
            All wavelengths are overlaid in a single figure with RGB coloring.

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_spot_radial.png'``.
            num_fov (int): Number of field positions sampled uniformly from
                on-axis (0) to full-field. Defaults to 6.
            depth (float): Object distance in mm (negative = real object).
                Defaults to ``DEPTH``.
            num_rays (int): Rays per field position per wavelength.
                Defaults to ``SPP_PSF``.
            wvln_list (list[float]): Wavelengths in micrometers.
                Defaults to ``WAVE_RGB`` (red, green, blue).
            direction (str): Sampling direction —
                ``"-y"`` (meridional, default), ``"y"``, ``"-x"``, ``"x"``,
                ``"-diagonal"``, or ``"diagonal"`` (45°).
            show (bool): If ``True``, display the figure interactively instead
                of saving to disk. Defaults to ``False``.
        """
        depth = self.obj_depth if depth is None else depth
        if wvln_list is None:
            wvln_list = self.wvln_ls
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if depth == float("inf"):
            depth = self.obj_depth

        # Parse direction sign ("-y" / "-x" / "-diagonal" samples the opposite side)
        sign = -1.0 if direction.startswith("-") else 1.0
        direction = direction.lstrip("-+")

        # Generate physical object-space points using ray-traced FoV to cover
        # the full sensor, even for lenses with significant distortion.
        max_obj_height = abs(depth) * float(np.tan(self.rfov)) * 0.98
        r = torch.linspace(0, 1.0, num_fov, device=self.device) * sign

        if direction == "y":
            px, py = torch.zeros_like(r), r * max_obj_height
        elif direction == "x":
            px, py = r * max_obj_height, torch.zeros_like(r)
        else:  # diagonal
            px, py = r * max_obj_height, r * max_obj_height

        z = torch.full_like(px, depth)
        points = torch.stack([px, py, z], dim=-1)

        # Prepare figure
        cols = min(3, max(1, num_fov))
        rows = (num_fov + cols - 1) // cols
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3.5, rows * 3), squeeze=False
        )
        axs = np.asarray(axs).reshape(-1)
        fov_fractions = (
            np.linspace(0.0, 1.0, num_fov) if num_fov > 1 else np.array([0.0])
        )
        rfov_deg = float(self.rfov) * 180.0 / float(np.pi)

        # Trace and draw each wavelength separately, overlaying results.
        field_x = [[] for _ in range(num_fov)]
        field_y = [[] for _ in range(num_fov)]
        for wvln_idx, wvln in enumerate(wvln_list):
            ray = self.spot_points(points, num_rays=num_rays, wvln=wvln)
            ray_o = ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()
            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            for i in range(num_fov):
                mask = ray_valid_np[i, :] > 0
                xi = ray_o[i, mask, 0]
                yi = ray_o[i, mask, 1]
                ax = axs[i]
                ax.scatter(xi, yi, 2, color=color, alpha=0.5)
                ax.set_aspect("equal", adjustable="datalim")
                ax.tick_params(axis="both", which="major", labelsize=7)
                if xi.size:
                    field_x[i].append(xi)
                    field_y[i].append(yi)

        for i in range(num_fov):
            fov_deg = round(float(fov_fractions[i]) * rfov_deg, 1)
            if field_x[i]:
                xs = np.concatenate(field_x[i])
                ys = np.concatenate(field_y[i])
                xc, yc = xs.mean(), ys.mean()
                d = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)
                rms_um = float(np.sqrt(np.mean(d * d)) * 1000.0)
                geo_um = float(d.max() * 1000.0)
                title = f"FoV={fov_deg}°  RMS={rms_um:.1f}µm  GEO={geo_um:.1f}µm"
            else:
                title = f"FoV={fov_deg}°  (vignetted)"
            axs[i].set_title(title, fontsize=10)

        for ax in axs[num_fov:]:
            ax.axis("off")

        plt.tight_layout()
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def draw_spot_map(
        self,
        save_name="./lens_spot_map.png",
        num_grid=5,
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        show=False,
    ):
        """Draw a 2-D grid of spot diagrams across the full field of view.

        Unlike ``draw_spot_radial`` (which samples only a radial slice),
        this method samples a ``num_grid × num_grid`` grid of field positions
        covering both the x (sagittal) and y (meridional) axes, revealing
        off-axis aberrations that are invisible in a 1-D radial scan.

        Algorithm:
            For each wavelength in ``wvln_list``:
                1. ``self.point_source_grid(normalized=False)`` creates physical
                   object-space grid points, shape ``[grid_h, grid_w, 3]``.
                2. ``self.spot_points()`` samples rays and traces to sensor.
                3. Valid (x, y) positions are scatter-plotted in the
                   corresponding subplot of the ``num_grid × num_grid`` figure.
            All wavelengths are overlaid with RGB coloring.

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_spot_map.png'``.
            num_grid (int | tuple[int, int]): Number of grid points along each
                axis. Total subplots = ``grid_w * grid_h``. Defaults to 5.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            num_rays (int): Rays per grid cell per wavelength.
                Defaults to ``SPP_PSF``.
            wvln_list (list[float]): Wavelengths in micrometers.
                Defaults to ``WAVE_RGB``.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        depth = self.obj_depth if depth is None else depth
        if wvln_list is None:
            wvln_list = self.wvln_ls
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Generate physical object-space grid points, shape [grid_h, grid_w, 3]
        points = self.point_source_grid(depth=depth, grid=num_grid, normalized=False)

        grid_w, grid_h = num_grid
        fig, axs = plt.subplots(
            grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3)
        )
        axs = np.atleast_2d(axs)

        # Loop wavelengths and overlay scatters
        for wvln_idx, wvln in enumerate(wvln_list):
            ray = self.spot_points(points, num_rays=num_rays, wvln=wvln)

            # Convert to numpy, shape [grid_h, grid_w, num_rays, 2]
            ray_o = -ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Draw per grid cell
            for i in range(grid_h):
                for j in range(grid_w):
                    valid = ray_valid_np[i, j, :]
                    xi, yi = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                    # Filter valid rays
                    mask = valid > 0
                    x_valid, y_valid = xi[mask], yi[mask]

                    # Plot points for this wavelength
                    axs[i, j].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                    axs[i, j].set_aspect("equal", adjustable="datalim")
                    axs[i, j].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # RMS map
    # ================================================================
    @torch.no_grad()
    def rms_map(self, num_grid=32, depth=None, wvln=None, center=None):
        """Compute per-field-position RMS spot radius for a single wavelength.

        Traces ``SPP_PSF`` rays per grid cell and computes the root-mean-square
        distance of valid ray hits from a reference centroid.  When ``center``
        is ``None``, each cell uses its own centroid (monochromatic blur).
        When an external ``center`` is provided (e.g. the green-channel
        centroid), the RMS includes the chromatic shift from that reference.

        Algorithm:
            1. ``self.point_source_grid(normalized=False)`` generates physical
               object points on a ``[num_grid, num_grid]`` field grid.
            2. ``self.spot_points()`` samples ``SPP_PSF`` rays per point and
               traces to sensor.
            3. If ``center`` is ``None``, compute per-cell centroid
               ``c = mean(valid ray_xy)``; otherwise use the provided ``center``.
            4. ``RMS = sqrt( mean( ||ray_xy - c||^2 ) )``.

        Args:
            num_grid (int | tuple[int, int]): Spatial resolution of the field
                sampling grid. Defaults to 32.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            wvln (float): Wavelength in micrometers. Defaults to ``DEFAULT_WAVE``.
            center (torch.Tensor | None): External reference centroid with shape
                ``[grid_h, grid_w, 2]``.  If ``None``, each cell's own
                centroid is used. Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - **rms**: RMS spot error map, shape ``[grid_h, grid_w]``,
                  in mm.
                - **centroid**: Per-cell centroid used as reference, shape
                  ``[grid_h, grid_w, 2]``.  Useful for passing as
                  ``center`` to subsequent calls (e.g. in ``rms_map_rgb``).
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Generate physical grid points and trace rays to sensor
        points = self.point_source_grid(depth=depth, grid=num_grid, normalized=False)
        ray = self.spot_points(points, num_rays=SPP_PSF, wvln=wvln)

        # Reuse Ray.centroid() — shape [grid_h, grid_w, 3], slice to [grid_h, grid_w, 2]
        centroid = ray.centroid()[..., :2]

        # Use external center if provided, otherwise own centroid
        ref = center if center is not None else centroid

        # RMS relative to reference, shape [grid_h, grid_w]
        ray_xy = ray.o[..., :2]
        ray_valid = ray.is_valid
        rms = torch.sqrt(
            (((ray_xy - ref.unsqueeze(-2)) ** 2).sum(-1) * ray_valid).sum(-1)
            / (ray_valid.sum(-1) + EPSILON)
        )

        return rms, centroid

    @torch.no_grad()
    def rms_map_rgb(self, num_grid=32, depth=None, wvln_list=None):
        """Compute per-field-position RMS spot radius for R, G, B wavelengths.

        The RMS spot radius is a standard measure of geometrical image quality.
        For each field position in a ``num_grid × num_grid`` grid, this method
        traces ``SPP_PSF`` rays per wavelength and computes the root-mean-square
        distance of valid ray hits from a **common** reference centroid.

        The reference centroid is the centroid of the middle wavelength
        (``wvln_list[1]``, nominally green).  Using a common reference means
        the returned RMS values include *lateral chromatic aberration* (the
        shift between R/G/B centroids), making the map useful as a
        polychromatic image-quality metric.

        Algorithm:
            1. Call ``rms_map(wvln=wvln_list[1])`` to get the middle-wavelength
               RMS map **and** its centroid.
            2. Call ``rms_map(wvln=wvln_list[0], center=mid_centroid)`` and
               ``rms_map(wvln=wvln_list[2], center=mid_centroid)`` to measure
               short/long wavelength blur relative to the middle reference.
            3. Stack as ``[wvln_list[0], wvln_list[1], wvln_list[2]]``.

        Args:
            num_grid (int): Spatial resolution of the field sampling grid.
                Defaults to 32.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            wvln_list (list[float]): Exactly three wavelengths in micrometers,
                ordered ``[short, middle, long]`` (e.g. ``[R, G, B]``). The
                middle wavelength is used as the reference centroid. Defaults
                to ``WAVE_RGB``.

        Returns:
            torch.Tensor: RMS spot error map with shape ``[3, num_grid, num_grid]``
                (channels ordered as ``wvln_list``). Units are mm (same as
                sensor coordinates).
        """
        depth = self.obj_depth if depth is None else depth
        if wvln_list is None:
            wvln_list = self.wvln_ls
        assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths (R, G, B)"

        # Green (middle wavelength) first to obtain the shared reference centroid
        rms_g, green_centroid = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=wvln_list[1]
        )

        # Red and blue relative to the green centroid
        rms_r, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=wvln_list[0], center=green_centroid
        )
        rms_b, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=wvln_list[2], center=green_centroid
        )

        return torch.stack([rms_r, rms_g, rms_b], dim=0)

    # ================================================================
    # Distortion
    # ================================================================
    @torch.no_grad()
    def calc_distortion_map(self, num_grid=16, depth=None, wvln=None, num_rays=SPP_CALC):
        """Compute a 2-D distortion grid mapping ideal to actual image positions.

        For each cell in a ``num_grid × num_grid`` field grid, rays are traced
        to the sensor and their centroid is computed.  The centroid is then
        normalized to ``[-1, 1]`` sensor coordinates, producing a map that
        shows how each ideal image point is displaced by lens distortion.

        This map can be used with ``torch.nn.functional.grid_sample`` to warp
        or unwarp rendered images.

        Args:
            num_grid (int): Grid resolution along each axis. Defaults to 16.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            wvln (float): Wavelength in micrometers. Defaults to ``DEFAULT_WAVE``.
            num_rays (int): Rays per grid cell. Only the centroid is kept, so
                PSF-scale sampling is wasteful here; default ``SPP_CALC`` is
                plenty and keeps ``num_grid=128`` unwarps from running out of
                GPU memory.

        Returns:
            torch.Tensor: Distortion grid with shape ``[num_grid, num_grid, 2]``.
                Each entry ``(dx, dy)`` is in normalized sensor coordinates
                ``[-1, 1]``, representing the actual centroid position for the
                corresponding ideal grid position.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Sample and trace rays, shape (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln, uniform_fov=False
        )
        ray = self.trace2sensor(ray)

        # Normalize to the same display convention used by
        # ``distortion_center`` and the spot/PSF visualizers.
        ray_center = -ray.centroid()[..., :2]
        sensor_w, sensor_h = self.sensor_size
        x_dist = ray_center[..., 0] / (sensor_w / 2)
        y_dist = ray_center[..., 1] / (sensor_h / 2)
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    def distortion_center(self, points):
        """Compute the distorted image centroid for arbitrary normalized object points.

        Given object points in normalized coordinates, this method converts them
        to physical object-space positions, traces rays from each point through
        the lens, and returns the ray centroid on the sensor in normalized
        ``[-1, 1]`` coordinates.  This is the inverse mapping needed for
        distortion correction (unwarping).

        Algorithm:
            1. Convert normalized ``(x, y)`` ∈ [-1, 1] to physical object-space
               positions using ``self.calc_scale(depth)`` and ``self.sensor_size``.
            2. ``self.sample_from_points()`` generates rays from each point.
            3. ``self.trace2sensor()`` propagates rays.
            4. Compute centroid and normalize back to ``[-1, 1]``.

        Args:
            points (torch.Tensor): Normalized point source positions with shape
                ``[N, 3]`` or ``[..., 3]``.  ``x, y`` ∈ [-1, 1] encode the
                field position; ``z`` ∈ (-∞, 0] is the object depth in mm.

        Returns:
            torch.Tensor: Normalized distortion centroid positions with shape
                ``[N, 2]`` or ``[..., 2]``.  ``x, y`` ∈ [-1, 1].
        """
        sensor_w, sensor_h = self.sensor_size

        # Convert normalized points to object space coordinates
        depth = points[..., 2]
        scale = self.calc_scale(depth)
        points_obj_x = points[..., 0] * scale * sensor_w / 2
        points_obj_y = points[..., 1] * scale * sensor_h / 2
        points_obj = torch.stack([points_obj_x, points_obj_y, depth], dim=-1)

        # Sample rays and trace to sensor
        ray = self.sample_from_points(points=points_obj)
        ray = self.trace2sensor(ray)

        # Calculate centroid and normalize to [-1, 1]
        ray_center = -ray.centroid()  # shape [..., 3]
        distortion_center_x = ray_center[..., 0] / (sensor_w / 2)
        distortion_center_y = ray_center[..., 1] / (sensor_h / 2)
        distortion_center = torch.stack((distortion_center_x, distortion_center_y), dim=-1)
        return distortion_center

    @torch.no_grad()
    def draw_distortion_map(
        self, save_name=None, num_grid=16, depth=None, wvln=None, show=False
    ):
        """Draw an ideal sensor grid with distorted sample points overlaid.

        The axes box follows the physical sensor aspect ratio so non-square
        sensors render as a rectangle instead of a forced square.

        Args:
            save_name (str | None): File path for the output PNG.  If ``None``,
                auto-generates ``'./distortion_{depth}.png'``.
            num_grid (int): Grid resolution per axis. Defaults to 16.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            wvln (float): Wavelength in micrometers. Defaults to ``DEFAULT_WAVE``.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Ray tracing to calculate distortion map
        distortion_grid = self.calc_distortion_map(num_grid=num_grid, depth=depth, wvln=wvln)
        arr = (
            distortion_grid.detach().cpu().numpy()
            if hasattr(distortion_grid, "detach")
            else np.asarray(distortion_grid)
        )
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError("distortion_grid must have shape [grid_h, grid_w, 2]")

        grid_h, grid_w = arr.shape[:2]
        gx = np.linspace(-1.0, 1.0, grid_w)
        gy = np.linspace(-1.0, 1.0, grid_h)
        ideal_x, ideal_y = np.meshgrid(gx, gy)
        actual_x, actual_y = arr[..., 0], arr[..., 1]
        sensor_w = max(float(self.sensor_size[0]), EPSILON)
        sensor_h = max(float(self.sensor_size[1]), EPSILON)
        box_aspect = sensor_h / sensor_w

        fig, ax = plt.subplots(figsize=(6, 6))
        for row in range(ideal_x.shape[0]):
            ax.plot(ideal_x[row, :], ideal_y[row, :], color="0.8", linewidth=0.8, zorder=1)
        for col in range(ideal_x.shape[1]):
            ax.plot(ideal_x[:, col], ideal_y[:, col], color="0.8", linewidth=0.8, zorder=1)
        ax.scatter(actual_x, actual_y, s=24, color="tab:blue", edgecolors="none", zorder=2)

        all_x = np.concatenate((ideal_x.ravel(), actual_x.ravel()))
        all_y = np.concatenate((ideal_y.ravel(), actual_y.ravel()))
        pad_x = max(0.05, 0.04 * max(float(all_x.max() - all_x.min()), EPSILON))
        pad_y = max(0.05, 0.04 * max(float(all_y.max() - all_y.min()), EPSILON))
        ax.set_xlim(float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ax.set_ylim(float(all_y.min() - pad_y), float(all_y.max() + pad_y))
        ax.set_aspect(box_aspect, adjustable="box")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if show:
            plt.show()
        else:
            depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
            if save_name is None:
                save_name = f"./distortion_{depth_str}.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # MTF
    # ================================================================
    def mtf(self, fov, depth=None, wvln=None):
        """Compute the geometric MTF at a single field position.

        The *Modulation Transfer Function* describes how well the lens preserves
        contrast as a function of spatial frequency.  MTF = 1 at low frequencies
        (perfect contrast) and falls toward 0 near the diffraction limit or the
        Nyquist frequency of the sensor.

        This implementation uses the *geometric* (ray-based) approach:
            1. Compute the PSF at the given field position via ``self.psf()``.
            2. Convert PSF → MTF via ``psf2mtf()`` (project onto tangential and
               sagittal axes, then take the magnitude of the 1-D FFT).

        Tangential MTF captures resolution in the meridional (radial) direction;
        sagittal MTF captures resolution perpendicular to it.  The difference
        between the two indicates astigmatism.

        Args:
            fov (float): Field position as a **fraction** of the ray-traced
                ``self.rfov`` (0 = on-axis, 1 = full field). Internally mapped
                to the normalized meridional object point used by ``self.psf()``.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            wvln (float): Wavelength in micrometers. Defaults to ``DEFAULT_WAVE``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - **freq**: Spatial frequency axis in cycles/mm (positive
                  frequencies only, excluding DC).
                - **mtf_tan**: Tangential (meridional) MTF values, normalized
                  so that MTF → 1 at low frequency.
                - **mtf_sag**: Sagittal MTF values, same normalization.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        theta = float(fov) * float(self.rfov)
        tan_half_field_y = self.sensor_size[1] / (2 * self.foclen)
        point_y = -float(np.tan(theta)) / max(float(tan_half_field_y), EPSILON)
        point = [0, point_y, depth]
        psf = self.psf(points=point, recenter=True, wvln=wvln)
        freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size=self.pixel_size)
        return freq, mtf_tan, mtf_sag

    @staticmethod
    def psf2mtf(psf, pixel_size):
        """Convert a 2-D point-spread function to tangential and sagittal MTF curves.

        The MTF is the magnitude of the optical transfer function (OTF), which
        is the Fourier transform of the PSF.  For separable 1-D analysis:
            1. Integrate the PSF along the x-axis → *tangential* line-spread
               function (LSF_tan).
            2. Integrate the PSF along the y-axis → *sagittal* LSF_sag.
            3. Take ``|FFT(LSF)|`` and normalize by the DC component so that
               MTF(0) = 1.

        Only positive frequencies (excluding DC) are returned, following the
        convention used in Zemax MTF plots.

        Args:
            psf (torch.Tensor | np.ndarray): 2-D PSF with shape ``[H, W]``.
                The array's y-axis (rows) corresponds to the **tangential**
                (meridional) direction; x-axis (columns) to the **sagittal**
                direction.
            pixel_size (float): Pixel pitch in mm.  Determines the frequency
                axis scaling: ``Nyquist = 0.5 / pixel_size`` cycles/mm.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - **freq**: Spatial frequency in cycles/mm (positive, excluding
                  DC).  Length is roughly ``H // 2``.
                - **mtf_tan**: Tangential MTF, normalized to 1 at DC.
                - **mtf_sag**: Sagittal MTF, normalized to 1 at DC.

        References:
            - https://en.wikipedia.org/wiki/Optical_transfer_function
            - Edmund Optics: Introduction to Modulation Transfer Function.
        """
        # Convert to numpy (supports torch tensors and numpy arrays)
        try:
            psf_np = psf.detach().cpu().numpy()
        except AttributeError:
            try:
                psf_np = psf.cpu().numpy()
            except AttributeError:
                psf_np = np.asarray(psf)

        # Compute line spread functions (integrate PSF over orthogonal axes)
        # y-axis corresponds to tangential; x-axis corresponds to sagittal
        lsf_sagittal = psf_np.sum(axis=0)  # function of x
        lsf_tangential = psf_np.sum(axis=1)  # function of y

        # One-sided spectra (for real inputs)
        mtf_sag = np.abs(np.fft.rfft(lsf_sagittal))
        mtf_tan = np.abs(np.fft.rfft(lsf_tangential))

        # Normalize by DC to ensure MTF(0) == 1
        dc_sag = mtf_sag[0] if mtf_sag.size > 0 else 1.0
        dc_tan = mtf_tan[0] if mtf_tan.size > 0 else 1.0
        if dc_sag != 0:
            mtf_sag = mtf_sag / dc_sag
        if dc_tan != 0:
            mtf_tan = mtf_tan / dc_tan

        # Frequency axis in cycles/mm (one-sided)
        fx = np.fft.rfftfreq(lsf_sagittal.size, d=pixel_size)
        freq = fx
        positive_freq_idx = freq > 0

        return (
            freq[positive_freq_idx],
            mtf_tan[positive_freq_idx],
            mtf_sag[positive_freq_idx],
        )

    @torch.no_grad()
    def draw_mtf(
        self,
        save_name="./lens_mtf.png",
        relative_fov_list=[0.0, 0.7, 1.0],
        depth_list=None,
        wvln_list=None,
        psf_ks=128,
        freq_cutoff=None,
        show=False,
    ):
        """Draw a grid of tangential MTF curves for multiple depths and field positions.

        Produces a ``len(depth_list) × len(relative_fov_list)`` subplot grid.
        Each subplot shows the tangential MTF for R, G, B wavelengths plus a
        vertical line at the sensor Nyquist frequency
        (``0.5 / pixel_size`` cycles/mm).

        Algorithm per subplot:
            1. Compute the RGB PSF via ``self.psf_rgb()`` at the specified
               ``(depth, relative_fov)`` with kernel size ``psf_ks``.
            2. For each wavelength channel, call ``psf2mtf()`` to obtain the
               tangential MTF curve.
            3. Plot frequency vs MTF with RGB coloring.

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_mtf.png'``.
            relative_fov_list (list[float]): Relative field positions in
                ``[0, 1]``, where 0 = on-axis and 1 = full field.
                Defaults to ``[0.0, 0.7, 1.0]``.
            depth_list (list[float]): Object distances in mm.
                ``float('inf')`` is automatically replaced by ``DEPTH``.
                Defaults to ``[DEPTH]``.
            wvln_list (list[float]): Wavelengths in micrometers to plot.
                Defaults to ``WAVE_RGB``.
            psf_ks (int): PSF kernel size in pixels (controls frequency
                resolution of the resulting MTF). Defaults to 128.
            freq_cutoff (float | None): Upper limit of the spatial-frequency
                axis in cycles/mm. If ``None`` (default), the sensor Nyquist
                frequency ``0.5 / pixel_size`` is used.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        if depth_list is None:
            depth_list = [self.obj_depth]
        if wvln_list is None:
            wvln_list = self.wvln_ls
        pixel_size = self.pixel_size
        nyquist_freq = 0.5 / pixel_size
        # Default cutoff frequency = sensor Nyquist
        if freq_cutoff is None:
            freq_cutoff = nyquist_freq
        num_fovs = len(relative_fov_list)
        if float("inf") in depth_list:
            depth_list = [self.obj_depth if x == float("inf") else x for x in depth_list]
        num_depths = len(depth_list)

        # Create figure and subplots (num_depths * num_fovs subplots)
        fig, axs = plt.subplots(
            num_depths, num_fovs, figsize=(num_fovs * 3, num_depths * 3), squeeze=False
        )

        tan_half_field_y = self.sensor_size[1] / (2 * self.foclen)

        # Iterate over depth and field of view
        for depth_idx, depth in enumerate(depth_list):
            for fov_idx, fov_relative in enumerate(relative_fov_list):
                theta = float(fov_relative) * float(self.rfov)
                point_y = -float(np.tan(theta)) / max(float(tan_half_field_y), EPSILON)
                point = [0, point_y, depth]

                # Calculate MTF curves for each wavelength
                for wvln_idx, wvln in enumerate(wvln_list):
                    psf = self.psf(points=point, ks=psf_ks, wvln=wvln)
                    freq, mtf_tan, _ = self.psf2mtf(psf, pixel_size)

                    # Plot MTF curves
                    ax = axs[depth_idx, fov_idx]
                    color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]
                    wvln_label = RGB_LABELS[wvln_idx % len(RGB_LABELS)]
                    wvln_nm = int(wvln * 1000)
                    ax.plot(
                        freq,
                        mtf_tan,
                        color=color,
                        label=f"{wvln_label}({wvln_nm}nm)-Tan",
                    )

                # Draw Nyquist frequency
                ax.axvline(
                    x=nyquist_freq,
                    color="k",
                    linestyle=":",
                    linewidth=1.2,
                    label="Nyquist",
                )

                # Set title and label for subplot
                fov_deg = round(fov_relative * self.rfov * 180 / np.pi, 1)
                depth_str = "inf" if depth == float("inf") else f"{depth}"
                ax.set_title(f"FoV={fov_deg}°, Depth={depth_str}mm", fontsize=10)
                ax.set_xlabel("Spatial Frequency [cycles/mm]", fontsize=8)
                ax.set_ylabel("MTF", fontsize=8)
                ax.legend(fontsize=7)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0, freq_cutoff)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Field Curvature
    # ================================================================
    @torch.no_grad()
    def draw_field_curvature(
        self,
        save_name=None,
        num_points=64,
        z_span=1.0,
        z_steps=201,
        wvln_list=None,
        spp=256,
        show=False,
    ):
        """Draw field curvature: best-focus defocus (Δz) vs field angle for RGB.

        *Field curvature* (Petzval curvature) causes off-axis image points to
        focus on a curved surface rather than the flat sensor.  This method
        finds the axial position of minimum RMS spot size at each field angle
        and plots the deviation from the nominal sensor plane.

        Algorithm (fully vectorized per wavelength):
            1. Construct a meridional ray fan at ``num_points`` field angles,
               each with ``spp`` rays spanning the entrance pupil.
            2. Trace all rays through the lens in a single batched call.
            3. For each of ``z_steps`` defocus planes within ``±z_span`` mm of
               ``self.d_sensor``, propagate rays analytically (linear
               extension) and compute the variance of the y-coordinate.
            4. The defocus with minimum variance is the best-focus plane.
               Parabolic interpolation on the three-point neighborhood gives
               sub-grid-step precision.
            5. Repeat for each wavelength; overlay R/G/B curves on a single plot.

        Args:
            save_name (str | None): File path for the output PNG.  If ``None``,
                defaults to ``'./field_curvature.png'``.
            num_points (int): Number of field-angle samples from 0 to
                ``self.rfov_eff``. Defaults to 64.
            z_span (float): Half-range of the defocus sweep in mm.  If the
                best-focus hits the boundary, a warning is printed.
                Defaults to 1.0.
            z_steps (int): Number of uniformly-spaced defocus planes within
                ``±z_span``. Higher values give finer axial resolution.
                Defaults to 201.
            wvln_list (list[float]): Wavelengths in micrometers.
                Defaults to ``WAVE_RGB``.
            spp (int): Rays per field point (sampled uniformly across the
                entrance pupil in the meridional plane). Defaults to 256.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        if wvln_list is None:
            wvln_list = self.wvln_ls
        device = self.device
        rfov_deg = float(self.rfov) * 180.0 / np.pi

        # Sample field angles [0, rfov_deg], shape [F]
        rfov_samples = torch.linspace(0.0, rfov_deg, num_points, device=device)

        # Entrance pupil (computed once)
        pupilz, pupilr = self.get_entrance_pupil()

        # Defocus sweep grid, shape [Z]
        d_sensor = self.d_sensor
        z_grid = d_sensor + torch.linspace(-z_span, z_span, z_steps, device=device)

        delta_z_tan = []

        for wvln in wvln_list:
            # --- Batch ray construction for all field angles ---
            # Pupil positions: shape [spp]
            pupil_y = torch.linspace(-pupilr, pupilr, spp, device=device) * 0.99

            # Ray origins: shape [F, spp, 3] (meridional plane: x=0)
            ray_o = torch.zeros(num_points, spp, 3, device=device)
            ray_o[..., 1] = pupil_y.unsqueeze(0)  # y = pupil sample
            ray_o[..., 2] = pupilz  # z = entrance pupil z

            # Ray directions: shape [F, spp, 3] (meridional: dx=0)
            fov_rad = rfov_samples * (np.pi / 180.0)  # [F]
            sin_fov = torch.sin(fov_rad)  # [F]
            cos_fov = torch.cos(fov_rad)  # [F]
            ray_d = torch.zeros(num_points, spp, 3, device=device)
            ray_d[..., 1] = sin_fov.unsqueeze(-1)  # [F, 1] -> [F, spp]
            ray_d[..., 2] = cos_fov.unsqueeze(-1)

            # Create batched ray and trace all field angles at once
            ray = Ray(ray_o, ray_d, wvln=wvln, device=device)
            ray, _ = self.trace(ray)

            # --- Vectorized best-focus for all field angles ---
            # ray.o: [F, spp, 3], ray.d: [F, spp, 3]
            oz = ray.o[..., 2:3]  # [F, spp, 1]
            dz = ray.d[..., 2:3]  # [F, spp, 1]
            t = (z_grid.view(1, 1, -1) - oz) / (dz + EPSILON)  # [F, spp, Z]

            oa = ray.o[..., 1:2]  # y-axis (tangential)
            da = ray.d[..., 1:2]
            pos_y = oa + da * t  # [F, spp, Z]

            w = ray.is_valid.unsqueeze(-1).float()  # [F, spp, 1]
            pos_y = pos_y * w  # mask invalid rays
            w_sum = w.sum(dim=1)  # [F, 1]

            centroid = pos_y.sum(dim=1) / (w_sum + EPSILON)  # [F, Z]
            ms = (((pos_y - centroid.unsqueeze(1)) ** 2) * w).sum(dim=1) / (
                w_sum + EPSILON
            )  # [F, Z]

            best_idx = torch.argmin(ms, dim=1)  # [F]

            # Warn if best focus hits z_span boundary
            boundary_hit = (best_idx == 0) | (best_idx == z_steps - 1)
            if boundary_hit.any():
                n_boundary = boundary_hit.sum().item()
                print(
                    f"Warning: {n_boundary}/{num_points} field angles hit z_span "
                    f"boundary. Consider increasing z_span (currently {z_span} mm)."
                )

            # Parabolic interpolation for sub-grid precision
            idx_c = best_idx.clamp(1, z_steps - 2)  # avoid boundary
            f_range = torch.arange(num_points, device=device)
            y_l = ms[f_range, idx_c - 1]
            y_c = ms[f_range, idx_c]
            y_r = ms[f_range, idx_c + 1]
            denom = 2.0 * (y_l - 2.0 * y_c + y_r)
            shift = (y_l - y_r) / (denom + EPSILON)  # fractional index offset
            shift = shift.clamp(-0.5, 0.5)  # safety clamp

            z_step_size = (2.0 * z_span) / (z_steps - 1)
            best_z = z_grid[idx_c] + shift * z_step_size  # [F]
            dz_tan = (best_z - d_sensor).cpu().numpy()

            # Mark fully-vignetted field angles as NaN (gaps in plot)
            valid_count = w.sum(dim=1).squeeze(-1)  # [F]
            fully_vignetted = (valid_count < 2).cpu().numpy()
            dz_tan[fully_vignetted] = np.nan

            delta_z_tan.append(dz_tan)

        # Plot
        fov_np = rfov_samples.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Field Curvature (Δz vs Field Angle)")

        all_vals = np.abs(np.concatenate(delta_z_tan)) if len(delta_z_tan) > 0 else np.array([0.0])
        x_range = float(max(0.2, all_vals.max() * 1.2)) if all_vals.size > 0 else 0.2

        for w_idx in range(len(wvln_list)):
            color = RGB_COLORS[w_idx % len(RGB_COLORS)]
            lbl = RGB_LABELS[w_idx % len(RGB_LABELS)]
            ax.plot(
                delta_z_tan[w_idx],
                fov_np,
                color=color,
                linestyle="-",
                label=f"{lbl}-Tan",
            )

        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1.0)
        ax.set_xlabel("Defocus Δz (mm) relative to sensor plane")
        ax.set_ylabel("Field Angle (deg)")
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(0, rfov_deg)
        ax.legend(fontsize=8)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = "./field_curvature.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Vignetting
    # ================================================================
    @torch.no_grad()
    def vignetting(self, depth=None, num_grid=32, num_rays=512, wvln=None):
        """Compute the relative-illumination (vignetting) map across the field.

        Vignetting measures how much light is lost at each field position due to
        rays being clipped by lens apertures or barrel edges.  It is computed as
        the fraction of traced rays that remain valid (not vignetted) at each
        grid cell, normalized by the total number of launched rays.

        A value of 1.0 means all rays reach the sensor (no vignetting); 0.0
        means complete light blockage.  Real lenses typically show 1.0 on-axis
        and fall off toward the field edges due to mechanical vignetting and the
        cos⁴ illumination law.

        Algorithm:
            1. ``self.sample_grid_rays()`` with ``uniform_fov=False`` (uniform
               image-space sampling) to ensure correct sensor-plane mapping.
            2. ``self.trace2sensor()`` propagates rays and marks clipped ones as
               invalid.
            3. Per-cell throughput = ``count(valid) / num_rays``.

        Args:
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            num_grid (int): Grid resolution per axis. Defaults to 32.
            num_rays (int): Rays launched per grid cell.  Higher values reduce
                Monte-Carlo noise. Defaults to 512.

        Returns:
            torch.Tensor: Vignetting map with shape ``[num_grid, num_grid]``,
                values in ``[0, 1]``.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Sample rays in uniform image space (not FOV angles) for correct sensor mapping
        # shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln, uniform_fov=False
        )

        # Trace rays to sensor
        ray = self.trace2sensor(ray)

        # Calculate vignetting map
        vignetting = ray.is_valid.sum(-1) / (ray.is_valid.shape[-1])
        return vignetting

    @torch.no_grad()
    def draw_vignetting(
        self,
        filename=None,
        depth=None,
        resolution=512,
        show=False,
        num_grid=32,
        num_rays=512,
        wvln=None,
    ):
        """Draw the vignetting map as a grayscale image with a colorbar.

        Computes the vignetting map via ``self.vignetting()``, bilinearly
        upsamples it to ``resolution × resolution``, and displays it as a
        grayscale image where white = no vignetting and black = fully vignetted.

        Args:
            filename (str | None): File path for the output PNG.  If ``None``,
                auto-generates ``'./vignetting_{depth}.png'``.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            resolution (int): Output image size in pixels (square).
                Defaults to 512.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
            num_grid (int): Grid resolution per axis used before upsampling.
            num_rays (int): Rays launched per grid cell during throughput sampling.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Calculate vignetting map
        vignetting = self.vignetting(
            depth=depth,
            num_grid=num_grid,
            num_rays=num_rays,
            wvln=wvln,
        )

        # Interpolate vignetting map to desired resolution
        vignetting = F.interpolate(
            vignetting.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        fig, ax = plt.subplots()
        ax.imshow(vignetting.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show:
            plt.show()
        else:
            if filename is None:
                filename = f"./vignetting_{depth}.png"
            plt.savefig(filename, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Wavefront error
    # ================================================================
    @torch.no_grad()
    def wavefront_error(
        self,
        relative_fov=0.0,
        depth=None,
        wvln=None,
        num_rays=SPP_COHERENT,
        ks=256,
    ):
        """Compute wavefront error (OPD) at the exit pupil for a given field position.

        The wavefront error is the optical path difference between the actual
        wavefront and the ideal spherical reference wavefront. The reference sphere
        is centered at the ideal image point (chief ray intersection with the sensor)
        and passes through the exit pupil center.

        By Fermat's principle, a perfect lens has equal total optical path (object →
        lens → image) for all rays. The deviation from this equal-path condition is
        the wavefront error:

            ``OPD(x,y) = [OPL(x,y) + r(x,y)] - mean_over_pupil``

        where ``OPL(x,y)`` is the accumulated optical path from the object through
        the lens to the exit pupil, and ``r(x,y)`` is the geometric distance from
        the exit pupil point to the ideal image point. Piston (mean) is removed.

        Uses the same coherent ray-tracing infrastructure as :meth:`pupil_field`.

        Args:
            relative_fov (float): Relative field of view in ``[-1, 1]`` along the
                meridional (y) direction. ``0`` = on-axis, ``1`` = full field.
            depth (float): Object distance [mm]. Use ``DEPTH`` for practical infinity.
            wvln (float): Wavelength [µm].
            num_rays (int): Number of rays to sample through the pupil.
            ks (int): Grid resolution for the OPD map at the exit pupil.

        Returns:
            dict:
                - ``opd_map`` (Tensor): OPD map on exit pupil grid, shape ``[ks, ks]``,
                  in waves. Invalid (vignetted) regions are zero.
                - ``rms`` (float): RMS wavefront error in waves (piston removed).
                - ``pv`` (float): Peak-to-valley wavefront error in waves.
                - ``valid_mask`` (Tensor): Boolean mask of valid pupil pixels ``[ks, ks]``.
                - ``strehl`` (float): Maréchal approximation Strehl ratio.

        Note:
            This function sets the default dtype to ``torch.float64`` for phase
            accuracy (consistent with :meth:`pupil_field`).

        References:
            [1] V. N. Mahajan, "Optical Imaging and Aberrations, Part II", Ch. 1.
            [2] Zemax OpticStudio, "Wavefront Error Analysis".
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Float64 required for accurate OPL accumulation
        self.astype(torch.float64)
        device = self.device
        sensor_w, sensor_h = self.sensor_size
        wvln_mm = wvln * 1e-3

        # Build normalized point: positive relative_fov -> negative y (convention)
        point_norm = torch.tensor(
            [0.0, -relative_fov, depth], dtype=torch.float64, device=device
        )
        points = point_norm.unsqueeze(0)  # [1, 3]

        # Convert to physical object coordinates
        scale = self.calc_scale(points[:, 2].item())
        point_obj_x = points[:, 0] * scale * sensor_w / 2
        point_obj_y = points[:, 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[:, 2]], dim=-1)

        # Find ideal image point via chief ray
        # psf_center returns negated centroid, so negate back to get actual image position
        chief_pointc = self.psf_center(point_obj, method="chief_ray")  # [1, 2]
        img_x = -chief_pointc[0, 0]
        img_y = -chief_pointc[0, 1]
        img_z = float(self.d_sensor)

        # Sample rays and trace coherently to exit pupil
        ray = self.sample_from_points(
            points=point_obj, num_rays=num_rays, wvln=wvln
        )
        ray.coherent = True
        ray = self.trace2exit_pupil(ray)

        # Get exit pupil parameters
        pupilz, pupilr = self.get_exit_pupil()
        pupilr = float(pupilr)
        pupilz = float(pupilz)

        # Extract valid rays (squeeze batch dim since single point)
        valid = ray.is_valid.squeeze(0) > 0  # [num_rays]
        ray_x = ray.o[0, :, 0]  # [num_rays]
        ray_y = ray.o[0, :, 1]
        opl = ray.opl[0, :, 0]  # [num_rays]

        if valid.sum() == 0:
            raise RuntimeError(
                f"No valid rays at relative_fov={relative_fov}. "
                "The field may be fully vignetted."
            )

        # Distance from each ray's exit pupil position to ideal image point
        dist_to_img = torch.sqrt(
            (ray_x - img_x) ** 2
            + (ray_y - img_y) ** 2
            + (pupilz - img_z) ** 2
        )

        # Total optical path = OPL through lens to exit pupil + free-space to image
        total_path = opl + dist_to_img  # [num_rays]

        # Remove piston (mean over valid rays) to get wavefront error
        total_path_valid = total_path[valid]
        mean_path = total_path_valid.mean()
        opd_mm = total_path - mean_path  # OPD in [mm]
        opd_waves = opd_mm / wvln_mm  # OPD in [waves]

        # Compute RMS and PV from per-ray values (more accurate than from grid)
        opd_valid = opd_waves[valid]
        rms_waves = torch.sqrt(torch.mean(opd_valid**2)).item()
        pv_waves = (opd_valid.max() - opd_valid.min()).item()

        # Maréchal approximation: Strehl ≈ exp(-(2π·σ)²)
        strehl = math.exp(-(2 * math.pi * rms_waves) ** 2)

        # Bin OPD values onto exit pupil grid using assign_points_to_pixels
        # Grid covers [-pupilr, pupilr] x [-pupilr, pupilr]
        pupil_range = [-pupilr, pupilr]
        pupil_points = torch.stack([ray_x[valid], ray_y[valid]], dim=-1)  # [N, 2]
        pupil_mask = torch.ones(pupil_points.shape[0], device=device)

        # Sum of weighted OPD values
        opd_sum = assign_points_to_pixels(
            points=pupil_points,
            mask=pupil_mask,
            ks=ks,
            x_range=pupil_range,
            y_range=pupil_range,
            value=opd_valid,
        )
        # Sum of weights (count)
        count = assign_points_to_pixels(
            points=pupil_points,
            mask=pupil_mask,
            ks=ks,
            x_range=pupil_range,
            y_range=pupil_range,
            value=torch.ones_like(opd_valid),
        )
        valid_mask = count > 0
        opd_map = torch.where(valid_mask, opd_sum / count, torch.zeros_like(opd_sum))

        return {
            "opd_map": opd_map,
            "rms": rms_waves,
            "pv": pv_waves,
            "valid_mask": valid_mask,
            "strehl": strehl,
        }

    @torch.no_grad()
    def rms_wavefront_error(
        self,
        relative_fov=0.0,
        depth=None,
        wvln=None,
        num_rays=SPP_COHERENT,
    ):
        """Compute scalar RMS wavefront error at a given field position.

        Convenience wrapper around :meth:`wavefront_error`.

        Args:
            relative_fov (float): Relative field of view in ``[-1, 1]``.
            depth (float): Object distance [mm].
            wvln (float): Wavelength [µm].
            num_rays (int): Number of rays to sample.

        Returns:
            float: RMS wavefront error in waves.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        result = self.wavefront_error(
            relative_fov=relative_fov,
            depth=depth,
            wvln=wvln,
            num_rays=num_rays,
        )
        return result["rms"]

    @torch.no_grad()
    def draw_wavefront_error(
        self,
        save_name="./wavefront_error.png",
        num_fov=6,
        depth=None,
        wvln=None,
        num_rays=SPP_COHERENT,
        ks=256,
        show=False,
    ):
        """Draw wavefront error (OPD) maps at multiple field positions.

        Evaluates the wavefront error along the meridional (y) direction from
        on-axis to full field, and displays each OPD map with RMS and PV
        annotations.

        Args:
            save_name (str): Filename to save the figure.
            num_fov (int): Number of field positions to evaluate.
            depth (float): Object distance [mm].
            wvln (float): Wavelength [µm].
            num_rays (int): Number of rays to sample per field position.
            ks (int): Grid resolution for each OPD map.
            show (bool): If True, display the figure interactively.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        fov_list = torch.linspace(0, 1, num_fov).tolist()
        rfov_deg = float(self.rfov_eff) * 180.0 / math.pi

        cols = min(3, max(1, num_fov))
        rows = (num_fov + cols - 1) // cols
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3.5, rows * 3.5), squeeze=False
        )
        axs = np.asarray(axs).reshape(-1)

        # Collect all OPD ranges to use a shared color scale
        results = []
        vmax = 0.0
        for fov in fov_list:
            try:
                result = self.wavefront_error(
                    relative_fov=fov,
                    depth=depth,
                    wvln=wvln,
                    num_rays=num_rays,
                    ks=ks,
                )
                results.append(result)
                opd_valid = result["opd_map"][result["valid_mask"]]
                if len(opd_valid) > 0:
                    vmax = max(vmax, opd_valid.abs().max().item())
            except RuntimeError:
                results.append(None)

        if vmax == 0:
            vmax = 1.0  # fallback

        for i, (fov, result) in enumerate(zip(fov_list, results)):
            fov_deg = fov * rfov_deg
            if result is None:
                axs[i].set_title(f"FoV={fov_deg:.2f}°  (vignetted)", fontsize=10)
                axs[i].axis("off")
                continue

            opd = result["opd_map"].cpu().numpy()
            mask = result["valid_mask"].cpu().numpy()
            rms = result["rms"]
            pv = result["pv"]

            # Mask invalid regions with NaN for visualization
            opd_vis = np.where(mask, opd, np.nan)

            im = axs[i].imshow(
                opd_vis,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="bilinear",
            )
            axs[i].set_title(
                f"FoV={fov_deg:.2f}°  RMS={rms:.4f}λ  PV={pv:.3f}λ",
                fontsize=10,
            )
            axs[i].axis("off")
            fig.colorbar(
                im,
                ax=axs[i],
                fraction=0.046,
                pad=0.04,
                label="OPD [waves]",
            )

        for j in range(len(fov_list), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    def field_curvature(self):
        """Compute field curvature data (best-focus defocus vs field angle).

        Field curvature is the axial shift of the best-focus surface away from
        the flat sensor plane as a function of field angle.  It is caused by
        the Petzval sum of lens surface curvatures and refractive indices.

        Not yet implemented.  See ``draw_field_curvature()`` for a plotting
        version that already performs the underlying computation.
        """
        pass

    # ====================================================================================
    # Spot, rendering, and comprehensive analysis
    # ====================================================================================
    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        save_name=None,
        depth=None,
        spp=SPP_RENDER,
        unwarp=False,
        method="ray_tracing",
        show=False,
    ):
        """Render a test image through the lens and report PSNR / SSIM.

        Simulates what the sensor would capture if the given image were placed
        at the specified object distance.  The rendering accounts for all
        geometric aberrations (blur, distortion, vignetting, chromatic effects).
        Optionally applies an inverse distortion warp (``unwarp``) and reports
        quality metrics for both the raw and unwarped renderings.

        Algorithm:
            1. Convert ``img_org`` to a ``[1, 3, H, W]`` float tensor and
               temporarily set the sensor resolution to match.
            2. Call ``self.render()`` with the chosen method (ray tracing or PSF
               convolution).
            3. Compute PSNR and SSIM between the original and rendered images.
            4. If ``unwarp=True``, apply ``self.unwarp()`` to correct geometric
               distortion and report metrics again.
            5. Restore the original sensor resolution.

        Args:
            img_org (np.ndarray | torch.Tensor): Source image with shape
                ``[H, W, 3]``, either uint8 ``[0, 255]`` or float ``[0, 1]``.
            save_name (str | None): Path prefix for saved PNGs.  If not
                ``None``, saves ``'{save_name}.png'`` and (if unwarped)
                ``'{save_name}_unwarped.png'``. Defaults to ``None``.
            depth (float): Object distance in mm. Defaults to ``DEPTH``.
            spp (int): Samples (rays) per pixel for rendering.
                Defaults to ``SPP_RENDER``.
            unwarp (bool): If ``True``, apply distortion correction after
                rendering. Defaults to ``False``.
            method (str): Rendering backend — ``'ray_tracing'`` or
                ``'psf_conv'``. Defaults to ``'ray_tracing'``.
            show (bool): If ``True``, display the result with matplotlib.
                Defaults to ``False``.

        Returns:
            torch.Tensor: Rendered (and optionally unwarped) image with shape
                ``[1, 3, H, W]``, float values in ``[0, 1]``.
        """
        depth = self.obj_depth if depth is None else depth
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from torchvision.utils import save_image

        # Change sensor resolution to match the image
        sensor_res_original = self.sensor_res
        if isinstance(img_org, np.ndarray):
            img = torch.from_numpy(img_org).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif torch.is_tensor(img_org):
            img = img_org.permute(2, 0, 1).unsqueeze(0).float()
            if img.max() > 1.0:
                img = img / 255.0
        img = img.to(self.device)
        self.set_sensor_res(sensor_res=img.shape[-2:])

        # Image rendering
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # Compute PSNR and SSIM
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
        render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
        render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}")

        # Save image
        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # Compute PSNR and SSIM
            render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
            render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
            render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
            print(
                f"Rendered image (unwarped): PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}"
            )

            if save_name is not None:
                save_image(img_render, f"{save_name}_unwarped.png")

        # Change the sensor resolution back
        self.set_sensor_res(sensor_res=sensor_res_original)

        # Show image
        if show:
            plt.imshow(img_render.cpu().squeeze(0).permute(1, 2, 0).numpy())
            plt.title("Rendered image")
            plt.axis("off")
            plt.show()
            plt.close()

        return img_render

    @torch.no_grad()
    def analysis_spot(
        self,
        num_field=3,
        depth=None,
        wvln_list=None,
        num_rays=SPP_PSF,
    ):
        """Compute polychromatic RMS and geometric spot radii at multiple field positions.

        Traces rays at ``num_field`` evenly-spaced field positions along the
        meridional direction for every wavelength in ``wvln_list``, and computes
        polychromatic RMS and geometric spot radii referenced to the
        **combined centroid across all wavelengths** (matching Zemax's
        default "RMS Spot Radius w.r.t. Centroid").

        This provides a quick polychromatic spot-size summary used for design
        comparisons and printed to stdout during ``analysis()``.

        Algorithm (per field point):
            1. Trace rays for every wavelength in ``wvln_list`` to the sensor.
            2. Pool all valid ray intercepts across wavelengths and compute
               one combined centroid ``c``.
            3. RMS = sqrt(mean(||xy - c||²)) over all pooled rays — a single
               polychromatic RMS that includes lateral chromatic aberration.
            4. radius = max(||xy - c||) over all pooled rays.
            5. Convert from mm to μm (× 1000).

        Args:
            num_field (int): Number of field positions sampled from on-axis
                to full-field. Defaults to 3.
            depth (float): Object distance in mm.  Use ``float('inf')`` for
                collimated light. Defaults to ``float('inf')``.
            wvln_list (list[float]): Wavelengths in micrometers to pool for
                the polychromatic centroid/RMS. Defaults to ``WAVE_RGB``.
            num_rays (int): Rays sampled per field point per wavelength.
                Defaults to ``SPP_PSF``.

        Returns:
            dict[str, dict[str, float]]: Spot analysis results keyed by field
                position string (e.g., ``'fov0.0'``, ``'fov0.5'``, ``'fov1.0'``).
                Each value is a dict with:
                    - ``'rms'``: Polychromatic RMS spot radius in μm.
                    - ``'radius'``: Polychromatic geometric spot radius in μm.
        """
        depth = self.obj_depth if depth is None else depth
        wvln_list = self.wvln_ls if wvln_list is None else wvln_list

        # Trace each wavelength and pool rays across wavelengths per field
        xy_list = []
        valid_list = []
        for wvln in wvln_list:
            ray = self.sample_radial_rays(
                num_field=num_field, depth=depth, num_rays=num_rays, wvln=wvln
            )
            ray = self.trace2sensor(ray)
            xy_list.append(ray.o[..., :2])
            valid_list.append(ray.is_valid)

        # Pool over wavelengths, shape [num_field, 3*num_rays, 2] and [num_field, 3*num_rays]
        xy_all = torch.cat(xy_list, dim=-2)
        valid_all = torch.cat(valid_list, dim=-1)

        # Combined polychromatic centroid per field, shape [num_field, 1, 2]
        valid_mask = valid_all.unsqueeze(-1)
        center = (xy_all * valid_mask).sum(-2) / (
            valid_all.sum(-1, keepdim=True) + EPSILON
        )
        center = center.unsqueeze(-2)

        # Squared distance to combined centroid, shape [num_field, 3*num_rays]
        dist_sq = ((xy_all - center) ** 2).sum(-1)

        # Polychromatic RMS spot radius per field, shape [num_field]
        spot_rms = (
            (dist_sq * valid_all).sum(-1) / (valid_all.sum(-1) + EPSILON)
        ).sqrt()
        # Geometric spot radius (max distance among valid rays)
        dist_masked = torch.where(
            valid_all > 0, dist_sq, torch.full_like(dist_sq, -1.0)
        )
        spot_radius = dist_masked.max(dim=-1).values.clamp(min=0.0).sqrt()

        # Convert mm → μm
        avg_rms_radius_um = spot_rms * 1000.0
        avg_geo_radius_um = spot_radius * 1000.0

        # Print results
        print(f"Ray spot analysis results for depth {depth}:")
        print(
            f"RMS radius: FoV (0.0) {avg_rms_radius_um[0]:.3f} um, FoV (0.5) {avg_rms_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_rms_radius_um[-1]:.3f} um"
        )
        print(
            f"Geo radius: FoV (0.0) {avg_geo_radius_um[0]:.3f} um, FoV (0.5) {avg_geo_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_geo_radius_um[-1]:.3f} um"
        )

        # Save to dict
        rms_results = {}
        fov_ls = torch.linspace(0, 1, num_field)
        for i in range(num_field):
            fov = round(fov_ls[i].item(), 2)
            rms_results[f"fov{fov}"] = {
                "rms": round(avg_rms_radius_um[i].item(), 4),
                "radius": round(avg_geo_radius_um[i].item(), 4),
            }

        return rms_results

    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        depth=None,
        full_eval=False,
        render=False,
        render_unwarp=False,
        lens_title=None,
        show=False,
        wvln_list=None,
        num_fov=7,
        num_grid=5,
        num_rays=SPP_PSF,
        num_points=GEO_GRID,
        psf_ks=128,
    ):
        """Run a comprehensive optical analysis pipeline for the lens.

        This is the main entry point for evaluating a lens design.  It chains
        multiple evaluation steps in order, saving all plots with a common
        ``save_name`` prefix.

        Execution flow:
            1. **Always**: draw the lens layout (``draw_layout``) and compute
               polychromatic spot RMS/radius (``analysis_spot``).
            2. **If** ``full_eval=True``: additionally generate:
               - Spot diagram (``draw_spot_radial``).
               - MTF grid (``draw_mtf``).
               - Distortion grid (``draw_distortion_map``).
               - Field curvature plot (``draw_field_curvature``).
               - Vignetting map (``draw_vignetting``).
            3. **If** ``render=True``: render a test chart image through the
               lens and report PSNR/SSIM (``analysis_rendering``).

        Args:
            save_name (str): Path prefix for all output files.  Each plot
                appends a suffix (e.g., ``'_spot.png'``, ``'_mtf.png'``).
                Defaults to ``'./lens'``.
            depth (float): Object distance in mm.  ``float('inf')`` is replaced
                by ``DEPTH`` for rendering and vignetting.
                Defaults to ``float('inf')``.
            full_eval (bool): If ``True``, run all evaluation plots.  If
                ``False``, only layout + spot RMS. Defaults to ``False``.
            render (bool): If ``True``, render a test image through the lens.
                Defaults to ``False``.
            render_unwarp (bool): If ``True`` (and ``render=True``), also
                produce an unwarped rendering. Defaults to ``False``.
            lens_title (str | None): Title string for the layout plot.
                Defaults to ``None``.
            show (bool): If ``True``, display all plots interactively.
                Defaults to ``False``.
            wvln_list (list[float]): Wavelengths in micrometers used by
                ``analysis_spot``, ``draw_spot_radial``, ``draw_mtf``,
                ``draw_field_curvature``, and the layout title.
                Defaults to ``WAVE_RGB``.
            num_fov (int): Number of radial field positions for
                ``draw_spot_radial``. Defaults to 7.
            num_grid (int): Grid resolution for ``draw_distortion_map``.
                Defaults to 5.
            num_rays (int): Rays per field point per wavelength used by
                ``analysis_spot`` and ``draw_spot_radial``. Defaults to
                ``SPP_PSF``.
            num_points (int): Number of samples for ``draw_field_curvature``.
                Defaults to ``GEO_GRID``.
            psf_ks (int): PSF kernel size passed to ``draw_mtf``. Defaults to 128.
        """
        if depth is None:
            depth = self.obj_depth
        if wvln_list is None:
            wvln_list = self.wvln_ls

        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
            show=show,
            num_views=num_fov,
            wvln_list=wvln_list,
            spot_num_rays=num_rays,
        )

        # Calculate RMS error
        self.analysis_spot(
            num_field=num_fov,
            depth=depth,
            wvln_list=wvln_list,
            num_rays=num_rays,
        )

        # Comprehensive optical evaluation
        if full_eval:
            # Draw spot diagram
            self.draw_spot_radial(
                save_name=f"{save_name}_spot.png",
                num_fov=num_fov,
                depth=depth,
                num_rays=num_rays,
                wvln_list=wvln_list,
                show=show,
            )

            # Draw MTF
            mtf_depth = self.obj_depth if depth == float("inf") else depth
            self.draw_mtf(
                depth_list=[mtf_depth],
                wvln_list=wvln_list,
                psf_ks=psf_ks,
                save_name=f"{save_name}_mtf.png",
                show=show,
            )

            # Draw distortion grid
            eval_depth = self.obj_depth if depth == float("inf") else depth
            self.draw_distortion_map(
                save_name=f"{save_name}_distortion.png",
                num_grid=num_grid,
                depth=eval_depth,
                show=show,
            )

            # Draw field curvature
            self.draw_field_curvature(
                save_name=f"{save_name}_field_curvature.png",
                num_points=num_points,
                wvln_list=wvln_list,
                show=show,
            )

            # Draw vignetting
            eval_depth = self.obj_depth if depth == float("inf") else depth
            self.draw_vignetting(
                filename=f"{save_name}_vignetting.png",
                depth=eval_depth,
                show=show,
            )

        # Render an image, compute PSNR and SSIM
        if render:
            depth = self.obj_depth if depth == float("inf") else depth
            img_org = Image.open("./datasets/charts/NBS_1963_1k.png").convert("RGB")
            img_org = np.array(img_org)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                show=show,
            )
