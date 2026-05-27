# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Geometric lens model. Differentiable ray tracing is used to simulate light propagation through a geometric lens. Accuracy is aligned with Zemax.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_WAVE,
    DELTA_PARAXIAL,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
)
from .geolens_pkg.eval import GeoLensEval
from .geolens_pkg.io import GeoLensIO
from .geolens_pkg.optim import GeoLensOptim
from .geolens_pkg.paraxial import GeoLensParaxial
from .geolens_pkg.psf_compute import GeoLensPSF
from .geolens_pkg.eval_seidel import GeoLensSeidel
from .geolens_pkg.optim_ops import GeoLensSurfOps
from .geolens_pkg.eval_tolerance import GeoLensTolerance
from .geolens_pkg.vis3d import GeoLensVis3D
from .geolens_pkg.vis import GeoLensVis
from .imgsim import backward_integral
from .lens import Lens
from .geometric_surface import Aperture
from .material import Material
from .ray import Ray

class GeoLens(
    GeoLensPSF,
    GeoLensEval,
    GeoLensSeidel,
    GeoLensOptim,
    GeoLensSurfOps,
    GeoLensVis,
    GeoLensIO,
    GeoLensTolerance,
    GeoLensVis3D,
    GeoLensParaxial,
    Lens,
):
    """Differentiable geometric lens using vectorised ray tracing.

    The primary lens model in DeepLens.  Supports multi-element refractive
    (and partially reflective) systems loaded from JSON, Zemax ``.zmx``, or
    Code V ``.seq`` files.  Accuracy is aligned with Zemax OpticStudio.

    Uses a **mixin architecture** – eight specialised mixin classes are
    composed at class definition time to keep each concern isolated:

    * :class:`~deeplens.optics.geolens_pkg.psf_compute.GeoLensPSF` – PSF
      computation (geometric, coherent, Huygens models).
    * :class:`~deeplens.optics.geolens_pkg.eval.GeoLensEval` – optical
      performance evaluation (spot, MTF, distortion, vignetting).
    * :class:`~deeplens.optics.geolens_pkg.optim.GeoLensOptim` – loss
      functions and gradient-based optimisation.
    * :class:`~deeplens.optics.geolens_pkg.optim_ops.GeoLensSurfOps` –
      surface geometry operations (aspheric conversion, pruning, shape
      correction, material matching).
    * :class:`~deeplens.optics.geolens_pkg.vis.GeoLensVis` – 2-D layout
      and ray visualisation.
    * :class:`~deeplens.optics.geolens_pkg.io.GeoLensIO` – read/write
      JSON, Zemax ``.zmx``.
    * :class:`~deeplens.optics.geolens_pkg.eval_tolerance.GeoLensTolerance` –
      manufacturing tolerance analysis.
    * :class:`~deeplens.optics.geolens_pkg.vis3d.GeoLensVis3D` – 3-D
      mesh visualisation.

    **Key differentiability trick**: Ray-surface intersection
    (:meth:`~deeplens.optics.geometric_surface.base.Surface.newtons_method`)
    uses a non-differentiable Newton loop followed by one differentiable
    Newton step to enable gradient flow.

    Attributes:
        surfaces (list[Surface]): Ordered list of optical surfaces.
        materials (list[Material]): Optical materials between surfaces.
        d_sensor (torch.Tensor): Back focal distance [mm].
        foclen (float): Effective focal length [mm].
        fnum (float): F-number.
        rfov_eff (float): Effective half-diagonal field of view [radians] (pinhole model).
        rfov (float): Half-diagonal field of view [radians] (ray-traced).
        sensor_size (tuple): Physical sensor size (W, H) [mm].
        sensor_res (tuple): Sensor resolution (W, H) [pixels].
        pixel_size (float): Pixel pitch [mm].

    References:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," *Nature Communications* 2024.
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
        use_ray_aiming=True,
        obj_depth=DEPTH,
        wvln_primary=DEFAULT_WAVE,
        wvln_ls=WAVE_RGB,
    ):
        """Initialize a refractive lens.

        There are two ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces and materials

        Args:
            filename (str, optional): Path to lens file (.json, .zmx, or .seq). Defaults to None.
            device (torch.device, optional): Device for tensor computations. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float32.
            use_ray_aiming (bool, optional): If False, use the paraxial
                post-computation fast path for EFL/FoV/pupil setup instead of
                the default ray-traced path. Defaults to True.
            obj_depth (float, optional): Default object depth in
                millimeters used when depth-dependent methods are called
                without an explicit ``depth=`` override. Defaults to ``DEPTH``.
            wvln_primary (float, optional): Primary wavelength in
                micrometers used when monochromatic methods are called
                without an explicit ``wvln=`` override. Defaults to
                ``DEFAULT_WAVE``.
            wvln_ls (list[float], optional): RGB wavelengths in micrometers
                used when polychromatic methods are called without an
                explicit ``wvln_list=`` override. Defaults to ``WAVE_RGB``.
        """
        super().__init__(device=device, dtype=dtype)

        self.aper_idx = None
        self.use_ray_aiming = bool(use_ray_aiming)
        self.obj_depth = float(obj_depth)
        self.wvln_primary = float(wvln_primary)
        self.wvln_ls = [float(w) for w in wvln_ls]

        # Load lens file
        if filename is not None:
            self.read_lens(filename)
        else:
            self.surfaces = []
            self.materials = []
            # Set default sensor size and resolution
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            self.to(self.device)

    def read_lens(self, filename):
        """Read a GeoLens from a file.

        Supported file formats:
            - .json: DeepLens native JSON format
            - .zmx: Zemax lens file format
            - .seq: CODE V sequence file format

        Args:
            filename (str): Path to the lens file.

        Note:
            Sensor size and resolution will usually be overwritten by values from the file.
        """
        # Load lens file
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        elif filename[-4:] == ".seq":
            self.read_lens_seq(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # Complete sensor size and resolution if not set from lens file
        if not hasattr(self, "sensor_size"):
            self.sensor_size = (8.0, 8.0)
            print(
                f"Sensor_size not found in lens file. Using default: {self.sensor_size} mm. "
                "Consider specifying sensor_size in the lens file or using set_sensor()."
            )

        if not hasattr(self, "sensor_res"):
            self.sensor_res = (2000, 2000)
            print(
                f"Sensor_res not found in lens file. Using default: {self.sensor_res} pixels. "
                "Consider specifying sensor_res in the lens file or using set_sensor()."
            )
            self.set_sensor_res(self.sensor_res)

        # After loading lens, find aperture and compute derived properties
        self.to(self.device)
        self.astype(self.dtype)
        if self.aper_idx is None:
            self.find_aperture()
        self.post_computation()

    def post_computation(self):
        """Compute derived optical properties after loading or modifying lens.

        Calculates and caches:
            - Effective focal length (EFL)
            - Entrance and exit pupil positions and radii
            - Field of view (FoV) in horizontal, vertical, and diagonal directions
            - F-number
            - Lens design constraint bounds (edge thickness, BFL, profile, etc.)
              via :meth:`init_constraints`. User overrides set after this call
              are preserved until ``post_computation`` is invoked again.

        Note:
            This method should be called after any changes to the lens geometry.
        """
        if not getattr(self, "use_ray_aiming", True):
            self.post_computation_paraxial()
            return

        if self.aper_idx is None:
            self.find_aperture()
        self.calc_foclen_paraxial()
        self.calc_pupil()
        self.calc_fov()
        self.init_constraints()

    def __call__(self, ray):
        """Trace rays through the lens system.

        Makes the GeoLens callable, allowing ray tracing with function call syntax.
        """
        return self.trace(ray)

    # ====================================================================================
    # Ray sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_grid_rays(
        self,
        depth=None,
        num_grid=(11, 11),
        num_rays=SPP_PSF,
        wvln=None,
        uniform_fov=True,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """Sample grid rays from object space.
            (1) If depth is infinite, sample parallel rays at different field angles.
            (2) If depth is finite, sample point source rays from the object plane.

        This function is usually used for (1) PSF map, (2) RMS error map, and (3) spot diagram calculation.

        Args:
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_grid (tuple, optional): number of grid points. Defaults to [11, 11].
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            uniform_fov (bool, optional): If True, sample uniform FoV angles.
            sample_more_off_axis (bool, optional): If True, sample more off-axis rays.
            scale_pupil (float, optional): Scale factor for pupil radius.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid[1], num_grid[0], num_rays, 3]
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Normalize num_grid to a tuple if it's an int
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Calculate field angles for grid source. Top-left field has positive fov_x and negative fov_y
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # Calculate FoV_x and FoV_y
        if uniform_fov:
            # Sample uniform FoV angles
            fov_x_list = [x * self.vfov / 2 for x in x_list]
            fov_y_list = [y * self.hfov / 2 for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]
        else:
            # Sample uniform object grid
            fov_x_list = [np.arctan(x * np.tan(self.vfov / 2)) for x in x_list]
            fov_y_list = [np.arctan(y * np.tan(self.hfov / 2)) for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]

        # Sample rays (collimated or point source via unified API)
        rays = self.sample_from_fov(
            fov_x=fov_x_list,
            fov_y=fov_y_list,
            depth=depth,
            num_rays=num_rays,
            wvln=wvln,
            scale_pupil=scale_pupil,
        )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=None,
        num_rays=SPP_PSF,
        wvln=None,
        direction="y",
    ):
        """Sample radial rays at evenly-spaced field angles along a chosen direction.

        Args:
            num_field (int): Number of field angles from on-axis to full-field.
                Defaults to 5.
            depth (float): Object distance in mm. Use ``float('inf')`` for
                collimated light. Defaults to ``float('inf')``.
            num_rays (int): Rays per field position. Defaults to ``SPP_PSF``.
            wvln (float): Wavelength in micrometers. Defaults to ``DEFAULT_WAVE``.
            direction (str): Sampling direction —
                ``"y"`` (meridional, default),
                ``"x"`` (sagittal),
                ``"diagonal"`` (45°, x = y).

        Returns:
            Ray: Ray object with shape ``[num_field, num_rays, 3]``.
        """
        device = self.device
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        fov_deg = float(np.rad2deg(self.rfov))
        fov_list = torch.linspace(0, fov_deg, num_field, device=device)

        if direction == "y":
            ray = self.sample_from_fov(
                fov_x=0.0, fov_y=fov_list, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "x":
            ray = self.sample_from_fov(
                fov_x=fov_list, fov_y=0.0, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "diagonal":
            # sample_from_fov creates a meshgrid; for pairwise diagonal, loop
            rays = [
                self.sample_from_fov(
                    fov_x=f.item(), fov_y=f.item(), depth=depth, num_rays=num_rays, wvln=wvln
                )
                for f in fov_list
            ]
            ray_o = torch.stack([r.o for r in rays], dim=0)
            ray_d = torch.stack([r.d for r in rays], dim=0)
            ray = Ray(ray_o, ray_d, wvln, device=device)
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Use 'x', 'y', or 'diagonal'.")
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=None,
        scale_pupil=1.0,
    ):
        """
        Sample rays from point sources in object space (absolute physical coordinates).

        Used for PSF and chief ray calculation.

        Args:
            points (list or Tensor): Ray origins in shape [3], [N, 3], or [Nx, Ny, 3].
            num_rays (int): Number of rays per point. Default: SPP_PSF.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Sampled rays with shape ``(\\*points.shape[:-1], num_rays, 3)``.
        """
        wvln = self.wvln_primary if wvln is None else wvln
        # Ray origin is given
        if not torch.is_tensor(points):
            ray_o = torch.tensor(points, device=self.device)
        else:
            ray_o = points.to(self.device)

        # Sample points on the pupil
        pupilz, pupilr = self.get_entrance_pupil()
        pupilr *= scale_pupil
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*ray_o.shape[:-1], num_rays)
        )

        # Compute ray directions
        if len(ray_o.shape) == 1:
            # Input point shape is [3]
            ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)  # shape [num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 2:
            # Input point shape is [N, 3]
            ray_o = ray_o.unsqueeze(1).repeat(1, num_rays, 1)  # shape [N, num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 3:
            # Input point shape is [Nx, Ny, 3]
            ray_o = ray_o.unsqueeze(2).repeat(
                1, 1, num_rays, 1
            )  # shape [Nx, Ny, num_rays, 3]
            ray_d = ray_o2 - ray_o

        else:
            raise Exception("The shape of input object positions is not supported.")

        # Calculate rays
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        return rays

    @torch.no_grad()
    def sample_from_fov(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=None,
        num_rays=SPP_CALC,
        wvln=None,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample rays from object space at given field angles.

        For infinite depth, generates collimated parallel rays: origins are
        distributed on the entrance pupil and all rays in a field share the
        same direction determined by the FOV angle.

        For finite depth, generates diverging point-source rays: the point
        source position is determined by FOV angle and depth, and rays fan
        out toward the entrance pupil.

        Args:
            fov_x (float or list): Field angle(s) in the xz plane (degrees).
            fov_y (float or list): Field angle(s) in the yz plane (degrees).
            depth (float): Object distance in mm. ``float('inf')`` for
                collimated rays, finite for point-source rays.
            num_rays (int): Number of rays per field point.
            wvln (float): Wavelength in micrometers.
            entrance_pupil (bool): If True, sample on entrance pupil;
                otherwise on surface 0. Default: True.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Rays with shape ``[..., num_rays, 3]``, where leading dims
                are squeezed when the corresponding fov input is scalar.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Track which inputs were scalar for output shape
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))

        # Accept float/int, list/tuple/ndarray, or torch.Tensor. Tensor inputs
        # stay on-device — avoids the GPU→CPU sync that `.tolist()` callers
        # (e.g. calc_fov) used to pay.
        ray_dtype = self.surfaces[0].d.dtype if len(self.surfaces) > 0 else self.dtype

        def _to_rad_1d(v):
            if torch.is_tensor(v):
                t = v.to(device=self.device, dtype=ray_dtype).reshape(-1)
            elif isinstance(v, (float, int)):
                t = torch.tensor([float(v)], device=self.device, dtype=ray_dtype)
            else:
                t = torch.tensor(
                    [float(x) for x in v], device=self.device, dtype=ray_dtype
                )
            return t * (torch.pi / 180.0)

        fov_x_rad = _to_rad_1d(fov_x)
        fov_y_rad = _to_rad_1d(fov_y)
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x_rad, fov_y_rad, indexing="xy")
        nx, ny = fov_x_rad.shape[0], fov_y_rad.shape[0]

        # Pupil position and radius
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = self.surfaces[0].d.item(), self.surfaces[0].r
        pupilr *= scale_pupil

        if depth == float("inf"):
            # Collimated rays: origins on pupil, uniform direction per field
            ray_o = self.sample_circle(
                r=pupilr, z=pupilz, shape=[ny, nx, num_rays]
            )
            dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
            dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
            dz = torch.ones_like(ray_o[..., 2])
            ray_d = torch.stack((dx, dy, dz), dim=-1)

            if x_scalar:
                ray_o = ray_o.squeeze(1)
                ray_d = ray_d.squeeze(1)
            if y_scalar:
                ray_o = ray_o.squeeze(0)
                ray_d = ray_d.squeeze(0)

            rays = Ray(ray_o, ray_d, wvln, device=self.device)
            rays.prop_to(-1.0)

        else:
            # Point-source rays: origin at object point, fan toward pupil
            x = torch.tan(fov_x_grid) * depth
            y = torch.tan(fov_y_grid) * depth
            z = torch.full_like(x, depth)
            points = torch.stack((x, y, z), dim=-1)

            if x_scalar:
                points = points.squeeze(-2)
            if y_scalar:
                points = points.squeeze(0)

            rays = self.sample_from_points(
                points=points, num_rays=num_rays, wvln=wvln, scale_pupil=scale_pupil
            )

        return rays

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=None, sub_pixel=False):
        """Sample rays from sensor pixels (backward rays). Used for ray tracing rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [H, W, spp, 3]
        """
        wvln = self.wvln_primary if wvln is None else wvln
        w, h = self.sensor_size
        W, H = self.sensor_res
        device = self.device

        # Sample points on sensor plane
        # Use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(
                -w / 2,
                w / 2,
                W + 1,
                device=device,
            )[1:],
            torch.linspace(
                h / 2,
                -h / 2,
                H + 1,
                device=device,
            )[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor)

        # Sample second points on the pupil
        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(H, W, spp))

        # Form rays
        ray_o = torch.stack((x1, y1, z1), 2)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, spp, 1)  # [H, W, spp, 3]

        # Sub-pixel sampling for more realistic rendering
        if sub_pixel:
            delta_ox = (
                torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # Form rays
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=device)
        return ray

    def sample_circle(self, r, z, shape=[16, 16, 512]):
        """Sample points inside a circle.

        Args:
            r (float): Radius of the circle.
            z (float): Z-coordinate for all sampled points.
            shape (list): Shape of the output tensor.

        Returns:
            torch.Tensor: Sampled points, shape ``(\\*shape, 3)``.
        """
        device = self.device

        # Generate random angles and radii
        theta = torch.rand(*shape, device=device) * 2 * torch.pi
        r2 = torch.rand(*shape, device=device) * r**2
        radius = torch.sqrt(r2)

        # Stack to form 3D points
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)
        points = torch.stack((x, y, z_tensor), dim=-1)

        # Manually sample chief ray
        # points[..., 0, :2] = 0.0

        return points

    # ====================================================================================
    # Ray tracing
    # ====================================================================================
    def trace(self, ray, surf_range=None, record=False):
        """Trace rays through the lens.

        Forward or backward tracing is automatically determined by the ray direction.

        Args:
            ray (Ray object): Ray object.
            surf_range (list): Surface index range.
            record (bool): record ray path or not.

        Returns:
            ray_final (Ray object): ray after optical system.
            ray_o_rec (list): list of intersection points.
        """
        if surf_range is None:
            surf_range = range(0, len(self.surfaces))

        if (ray.d[..., 2] > 0).any():
            ray_out, ray_o_rec = self.forward_tracing(ray, surf_range, record=record)
        else:
            ray_out, ray_o_rec = self.backward_tracing(ray, surf_range, record=record)

        return ray_out, ray_o_rec

    def trace2obj(self, ray):
        """Traces rays backwards through all lens surfaces from sensor side
        to object side.

        Args:
            ray (Ray): Ray object to trace backwards.

        Returns:
            Ray: Ray object after backward propagation through the lens.
        """
        ray, _ = self.trace(ray)
        return ray

    def trace2sensor(self, ray, record=False):
        """Forward trace rays through the lens to sensor plane.

        Args:
            ray (Ray object): Ray object.
            record (bool): record ray path or not.

        Returns:
            ray_out (Ray object): ray after optical system.
            ray_o_record (list): list of intersection points.
        """
        # Manually propagate ray to a shallow depth to avoid numerical instability
        if ray.o[..., 2].min() < -100.0:
            ray = ray.prop_to(-10.0)

        # Trace rays
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.prop_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            # Set to NaN to be skipped in 2d layout visualization
            ray_o[ray.is_valid == 0] = float("nan")
            ray_o_record.append(ray_o)
            return ray, ray_o_record
        else:
            return ray

    def trace2exit_pupil(self, ray):
        """Forward trace rays through the lens to exit pupil plane.

        Args:
            ray (Ray): Ray object to trace.

        Returns:
            Ray: Ray object propagated to the exit pupil plane.
        """
        ray = self.trace2sensor(ray)
        pupil_z, _ = self.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        return ray

    def forward_tracing(self, ray, surf_range, record):
        """Forward traces rays through each surface in the specified range from object side to image side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in surf_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    def backward_tracing(self, ray, surf_range, record):
        """Backward traces rays through each surface in reverse order from image side to object side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after backward propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        # Initial material: the material the ray is in when entering the
        # backward trace. If the range ends before the last surface, the ray
        # starts inside surfaces[max_idx].mat2, not air.
        max_idx = max(surf_range)
        if max_idx < len(self.surfaces) - 1:
            mat1 = self.surfaces[max_idx].mat2
        else:
            mat1 = Material("air")

        for i in np.flip(surf_range):
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i - 1].mat2.ior(ray.wvln) if i > 0 else Material("air").ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i - 1].mat2 if i > 0 else Material("air")

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    # ====================================================================================
    # Image simulation
    # ====================================================================================
    def render(self, img_obj, depth=None, method="ray_tracing", **kwargs):
        """Differentiable image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] PSF patch convolution.
            [3] Ray tracing rendering.

        Args:
            img_obj (Tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. One of 'psf_map', 'psf_patch',
                or 'ray_tracing'. Defaults to 'ray_tracing'.
            **kwargs: Additional arguments for different methods:
                - psf_grid (tuple): Grid size for PSF map method. Defaults to (10, 10).
                - psf_ks (int): Kernel size for PSF methods. Defaults to PSF_KS.
                - patch_center (tuple): Center position for PSF patch method.
                - spp (int): Samples per pixel for ray tracing. Defaults to SPP_RENDER.
                - wvln_list (list[float]): Per-channel wavelengths (ray_tracing).
                  Defaults to the lens RGB wavelengths.

        Returns:
            Tensor: Rendered image tensor. Shape of [N, C, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # Image simulation
        if method == "psf_map":
            # PSF rendering - uses PSF map to render image
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_map(
                img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
            )

        elif method == "psf_patch":
            # PSF patch rendering - uses a single PSF to render a patch of the image
            patch_center = kwargs.get("patch_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, patch_center=patch_center, psf_ks=psf_ks
            )

        elif method == "ray_tracing":
            # Ray tracing rendering
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            spp = kwargs.get("spp", SPP_RENDER)
            wvln_list = kwargs.get("wvln_list")
            if wvln_list is None:
                wvln_list = self.wvln_ls
            img_render = self.render_raytracing(
                img_obj, depth=depth, spp=spp, wvln_list=wvln_list
            )

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_raytracing(
        self,
        img,
        depth=None,
        spp=SPP_RENDER,
        wvln_list=None,
        vignetting=False,
    ):
        """Render a 3-channel image using ray tracing rendering.

        Args:
            img (tensor): Image tensor. Shape of [N, 3, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Samples per pixel. Defaults to SPP_RENDER.
            wvln_list (list[float], optional): Per-channel wavelengths in
                micrometers. Must contain exactly 3 entries matching the
                image's channel ordering. Defaults to the lens RGB wavelengths.
            vignetting (bool, optional): Whether to consider vignetting
                effect. Defaults to False.

        Returns:
            img_render (tensor): Rendered image tensor. Shape of [N, 3, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        if wvln_list is None:
            wvln_list = self.wvln_ls
        assert len(wvln_list) == 3, "wvln_list must contain 3 wavelengths"
        img_render = torch.zeros_like(img)
        for i in range(3):
            img_render[:, i, :, :] = self.render_raytracing_mono(
                img=img[:, i, :, :],
                wvln=wvln_list[i],
                depth=depth,
                spp=spp,
                vignetting=vignetting,
            )
        return img_render

    def render_raytracing_mono(
        self, img, wvln, depth=None, spp=SPP_RENDER, vignetting=False
    ):
        """Render monochrome image using ray tracing rendering.

        Args:
            img (tensor): Monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
            wvln (float): Wavelength of the light.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Samples per pixel. Defaults to SPP_RENDER.

        Returns:
            img_mono (tensor): Rendered monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        img = torch.flip(img, [-2, -1])
        scale = self.calc_scale(depth=depth)
        ray = self.sample_sensor(spp=spp, wvln=wvln)
        ray = self.trace2obj(ray)
        img_mono = self.render_compute_image(
            img, depth=depth, scale=scale, ray=ray, vignetting=vignetting
        )
        return img_mono

    def render_compute_image(self, img, depth, scale, ray, vignetting=False):
        """Computes the intersection points between rays and the object image plane, then generates the rendered image following rendering equation.

        Back-propagation gradient flow: image -> w_i -> u -> p -> ray -> surface

        Args:
            img (tensor): [N, C, H, W] or [N, H, W] shape image tensor.
            depth (float): depth of the object.
            scale (float): scale factor.
            ray (Ray object): Ray object. Shape [H, W, spp, 3].
            vignetting (bool): whether to consider vignetting effect.

        Returns:
            image (tensor): [N, C, H, W] or [N, H, W] shape rendered image tensor.
        """
        assert torch.is_tensor(img), "Input image should be Tensor."

        squeeze_channel = img.ndim == 3
        if squeeze_channel:
            img = img.unsqueeze(1)
        elif img.ndim != 4:
            raise ValueError("Input image should be [N, C, H, W] or [N, H, W] tensor.")

        H, W = img.shape[-2:]
        pixel_size = scale * self.pixel_size

        # Propagate to object plane and gate rays that fell outside the image
        ray = ray.prop_to(depth)
        p = ray.o[..., :2]
        ray.is_valid = (
            ray.is_valid
            * (torch.abs(p[..., 0] / pixel_size) < (W / 2 + 1))
            * (torch.abs(p[..., 1] / pixel_size) < (H / 2 + 1))
        )

        image = backward_integral(
            ray,
            img,
            ps=pixel_size,
            H=H,
            W=W,
            interpolate=True,
            energy_correction=None,
            vignetting=vignetting,
        )

        if squeeze_channel:
            image = image.squeeze(1)
        return image

    def unwarp(self, img, depth=None, num_grid=128):
        """Unwarp rendered images using distortion map.

        Args:
            img (tensor): Rendered image tensor. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            num_grid (int, optional): Distortion-map grid resolution.

        Returns:
            img_unwarpped (tensor): Unwarped image tensor. Shape of [N, C, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        # Calculate distortion map, shape (num_grid, num_grid, 2)
        distortion_map = self.calc_distortion_map(depth=depth, num_grid=num_grid)

        # calc_distortion_map stores, for each grid cell, the actual sensor
        # landing position in grid_sample (x, y)-coords. Measurement shows
        # grid col 0 already matches input x=-1 (left), but grid row 0 maps
        # to input y=+1 (bottom), whereas F.interpolate treats grid row 0 as
        # image top. Flip the row axis so the grid aligns with image
        # (row, col) -> (top->bottom, left->right).
        distortion_map = torch.flip(distortion_map, [0])

        # Interpolate distortion map to image resolution
        distortion_map = distortion_map.permute(2, 0, 1).unsqueeze(1)
        distortion_map = F.interpolate(
            distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )  # shape (B, 2, Himg, Wimg)
        distortion_map = distortion_map.permute(1, 2, 3, 0).repeat(
            img.shape[0], 1, 1, 1
        )  # shape (B, Himg, Wimg, 2)

        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(
            img, distortion_map, align_corners=True
        )  # shape (B, C, Himg, Wimg)
        return img_unwarpped

    # ====================================================================================
    # Geometrical optics calculation
    # ====================================================================================

    def find_aperture(self):
        """Find and set the aperture stop index.

        Called after loading when no surface was marked with ``is_aperture``
        in the lens file. Looks for an ``Aperture`` surface instance first,
        then falls back to the surface with the smallest semi-diameter.

        Sets:
            self.aper_idx (int): Index of the aperture surface.
        """
        for i, s in enumerate(self.surfaces):
            if isinstance(s, Aperture):
                self.aper_idx = i
                return

        self.aper_idx = int(np.argmin([s.r for s in self.surfaces]))
        print("No aperture found, use the smallest surface as aperture.")

    @torch.no_grad()
    def calc_foclen(self, test_fov_deg=1.0, wvln=None):
        return self.calc_foclen_paraxial(wvln=wvln)


    @torch.no_grad()
    def calc_numerical_aperture(self, n=1.0):
        """Compute numerical aperture (NA).

        Args:
            n (float, optional): Refractive index. Defaults to 1.0.

        Returns:
            NA (float): Numerical aperture.

        Reference:
            [1] https://en.wikipedia.org/wiki/Numerical_aperture
        """
        return n * math.sin(math.atan(1 / 2 / self.fnum))

    @torch.no_grad()
    def calc_focal_plane(self, wvln=None):
        """Compute the focus distance in the object space. Ray starts from sensor center and traces to the object space.

        Args:
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.

        Returns:
            focal_plane (float): Focal plane in the object space.
        """
        wvln = self.wvln_primary if wvln is None else wvln
        device = self.device

        # Sample point source rays from sensor center
        o1 = torch.zeros(SPP_CALC, 3, device=device)
        o1[:, 2] = self.d_sensor

        # Sample the first surface as pupil
        # o2 = self.sample_circle(self.surfaces[0].r, z=0.0, shape=[SPP_CALC])
        # o2 *= 0.5  # Shrink sample region to improve accuracy
        pupilz, pupilr = self.get_exit_pupil()
        o2 = self.sample_circle(pupilr, pupilz, shape=[SPP_CALC])
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=device)

        # Trace rays to object space
        ray = self.trace2obj(ray)

        # Optical axis intersection — stay on device, one sync at the end.
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2 + EPSILON
        )
        focus_z = ray.o[..., 2] - ray.d[..., 2] * t
        keep = (ray.is_valid > 0) & ~torch.isnan(focus_z) & (focus_z < 0)
        focus_z = focus_z[keep]

        if focus_z.numel() > 0:
            focal_plane = focus_z.mean().item()
        else:
            raise ValueError(
                "No valid rays found, focal plane in the image space cannot be computed."
            )

        return focal_plane

    @torch.no_grad()
    def calc_sensor_plane(
        self, depth=None, wvln=None, num_rays=SPP_CALC
    ):
        """Calculate in-focus sensor plane.

        Args:
            depth (float, optional): Depth of the object plane. Defaults to float("inf").
            wvln (float, optional): Wavelength in micrometers used to trace the
                on-axis fan. Defaults to DEFAULT_WAVE.
            num_rays (int, optional): Number of on-axis rays used to locate the
                focus. Defaults to SPP_CALC.

        Returns:
            d_sensor (torch.Tensor): Sensor plane in the image space.
        """
        depth = self.obj_depth if depth is None else depth
        wvln = self.wvln_primary if wvln is None else wvln
        # Sample and trace rays, shape [num_rays, 3]
        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=0.0, depth=depth, num_rays=num_rays, wvln=wvln
        )
        ray = self.trace2sensor(ray)

        # Calculate in-focus sensor position
        t = (ray.d[:, 0] * ray.o[:, 0] + ray.d[:, 1] * ray.o[:, 1]) / (
            ray.d[:, 0] ** 2 + ray.d[:, 1] ** 2 + EPSILON
        )
        focus_z = ray.o[:, 2] - ray.d[:, 2] * t
        focus_z = focus_z[ray.is_valid > 0]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        d_sensor = torch.mean(focus_z)
        return d_sensor

    @torch.no_grad()
    def calc_fov(self):
        """Compute field of view (FoV) of the lens in radians.

        Calculates FoV using two methods:
            1. **Perspective projection** — from focal length and sensor size
               (effective FoV, ignoring distortion).
            2. **Ray tracing** — traces rays from the sensor edge backwards to
               determine the real FoV including distortion effects.

        Updates:
            self.vfov (float): Vertical FoV in radians.
            self.hfov (float): Horizontal FoV in radians.
            self.dfov (float): Diagonal FoV in radians.
            self.rfov_eff (float): Half-diagonal (radius) FoV in radians.
            self.rfov (float): Real half-diagonal FoV from ray tracing.
            self.real_dfov (float): Real diagonal FoV from ray tracing.
            self.eqfl (float): 35mm equivalent focal length in mm.

        Reference:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        # 1. Perspective projection (effective FoV)
        self.vfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.hfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov_eff = self.dfov / 2  # radius (half diagonal) FoV

        # 2. Forward ray tracing to calculate real FoV (distortion-affected)
        # Sweep FOV angles from object side, trace to sensor, and find which
        # angle produces an image height matching r_sensor.
        num_fov = 64
        fov_lo = float(np.rad2deg(self.rfov_eff)) * 0.5
        fov_hi = min(float(np.rad2deg(self.rfov_eff)) * 1.5, 89.0)
        fov_samples = torch.linspace(fov_lo, fov_hi, num_fov, device=self.device)

        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=fov_samples, num_rays=256
        )
        ray = self.trace2sensor(ray)

        # Centroid image height per FOV angle, shape [num_fov]
        valid = ray.is_valid > 0  # [num_fov, num_rays]
        masked_y = ray.o[..., 1] * valid
        n_valid = valid.sum(dim=-1).clamp(min=1)
        imgh = (masked_y.sum(dim=-1) / n_valid).abs()

        # Find the FOV angle whose image height is closest to r_sensor
        has_valid = valid.sum(dim=-1) > 10
        if has_valid.any():
            imgh = torch.where(has_valid, imgh, imgh.new_full((), float("inf")))
            best_deg = fov_samples[(imgh - self.r_sensor).abs().argmin()]
            rfov = (best_deg * (math.pi / 180.0)).item()
            self.rfov = rfov
            self.real_dfov = 2 * rfov
        else:
            self.rfov = self.rfov_eff
            self.real_dfov = self.dfov

        # 3. Compute 35mm equivalent focal length. 35mm sensor: 36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

    @torch.no_grad()
    def calc_scale(self, depth):
        """Calculate the scale factor (object height / image height).

        Uses the pinhole camera model to compute magnification.

        Args:
            depth (float): Object distance from the lens (negative z direction).

        Returns:
            float: Scale factor relating object height to image height.
        """
        return -depth / self.foclen

    @torch.no_grad()
    def calc_pupil(self):
        """Compute entrance and exit pupil positions and radii.

        The entrance and exit pupils must be recalculated whenever:
            - First-order parameters change (e.g., field of view, object height, image height),
            - Lens geometry or materials change (e.g., surface curvatures, refractive indices, thicknesses),
            - Or generally, any time the lens configuration is modified.

        Updates:
            self.aper_idx: Index of the aperture surface.
            self.exit_pupilz, self.exit_pupilr: Exit pupil position and radius.
            self.entr_pupilz, self.entr_pupilr: Entrance pupil position and radius.
            self.exit_pupilz_parax, self.exit_pupilr_parax: Paraxial exit pupil.
            self.entr_pupilz_parax, self.entr_pupilr_parax: Paraxial entrance pupil.
            self.fnum: F-number calculated from focal length and entrance pupil.
        """
        # Compute entrance and exit pupil
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil(paraxial=False)
        self.entr_pupilz, self.entr_pupilr = self.calc_entrance_pupil(paraxial=False)
        self.exit_pupilz_parax, self.exit_pupilr_parax = self.calc_exit_pupil(
            paraxial=True
        )
        self.entr_pupilz_parax, self.entr_pupilr_parax = self.calc_entrance_pupil(
            paraxial=True
        )

        # Compute F-number
        self.fnum = self.foclen / (2 * self.entr_pupilr)

    def get_entrance_pupil(self, paraxial=False):
        """Get entrance pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            tuple: (z_position, radius) of the entrance pupil in [mm].
        """
        if paraxial:
            return self.entr_pupilz_parax, self.entr_pupilr_parax
        else:
            return self.entr_pupilz, self.entr_pupilr

    def get_exit_pupil(self, paraxial=False):
        """Get exit pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            tuple: (z_position, radius) of the exit pupil in [mm].
        """
        if paraxial:
            return self.exit_pupilz_parax, self.exit_pupilr_parax
        else:
            return self.exit_pupilz, self.exit_pupilr

    @torch.no_grad()
    def calc_exit_pupil(self, paraxial=False):
        """Calculate exit pupil location and radius.

        Paraxial mode:
            Rays are emitted from near the center of the aperture stop and are close to the optical axis.
            This mode estimates the exit pupil position and radius under ideal (first-order) optical assumptions.
            It is fast and stable.

        Non-paraxial mode:
            Rays are emitted from the edge of the aperture stop in large quantities.
            The exit pupil position and radius are determined based on the intersection points of these rays.
            This mode is slower and affected by aperture-related aberrations.

        Use paraxial mode unless precise ray aiming is required.

        Args:
            paraxial (bool): center (True) or edge (False).

        Returns:
            avg_pupilz (float): z coordinate of exit pupil.
            avg_pupilr (float): radius of exit pupil.

        Reference:
            [1] Exit pupil: how many rays can come from sensor to object space.
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            print("No aperture, use the last surface as exit pupil.")
            return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # Sample rays from aperture (edge or center)
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi_rad = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(SPP_CALC, 1)
            rfov_eff = float(np.arctan(self.r_sensor / self.foclen))
            phi_rad = torch.linspace(-rfov_eff / 2, rfov_eff / 2, SPP_CALC, device=self.device)

        d = torch.stack(
            (torch.sin(phi_rad), torch.zeros_like(phi_rad), torch.cos(phi_rad)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to last surface
        surf_range = range(self.aper_idx + 1, len(self.surfaces))
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid != 0][:, 0], ray.o[ray.is_valid != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid != 0][:, 0], ray.d[ray.is_valid != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small pupil
        if len(intersection_points) == 0:
            print("No intersection points found, use the last surface as exit pupil.")
            avg_pupilr = self.surfaces[-1].r
            avg_pupilz = self.surfaces[-1].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative exit pupil is detected, use the last surface as pupil."
                )
                avg_pupilr = self.surfaces[-1].r
                avg_pupilz = self.surfaces[-1].d.item()

        return avg_pupilz, avg_pupilr

    @torch.no_grad()
    def calc_entrance_pupil(self, paraxial=False):
        """Calculate entrance pupil of the lens.

        The entrance pupil is the optical image of the physical aperture stop, as seen through the optical elements in front of the stop. We sample backward rays from the aperture stop and trace them to the first surface, then find the intersection points of the reverse extension of the rays. The average of the intersection points defines the entrance pupil position and radius.

        Args:
            paraxial (bool): Ray sampling mode.  If ``True``, rays are emitted
                near the centre of the aperture stop (fast, paraxially stable).
                If ``False``, rays are emitted from the stop edge in larger
                quantities (slower, accounts for aperture aberrations).
                Defaults to ``False``.

        Returns:
            tuple: (z_position, radius) of entrance pupil.

        Note:
            [1] Use paraxial mode unless precise ray aiming is required.
            [2] This function only works for object at a far distance. For microscopes, this function usually returns a negative entrance pupil.

        References:
            [1] Entrance pupil: how many rays can come from object space to sensor.
            [2] https://en.wikipedia.org/wiki/Entrance_pupil: "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
            [3] Zemax LLC, *OpticStudio User Manual*, Version 19.4, Document No. 2311, 2019.
        """
        if self.aper_idx is None or not hasattr(self, "aper_idx"):
            print("No aperture stop, use the first surface as entrance pupil.")
            return self.surfaces[0].d.item(), self.surfaces[0].r

        # Sample rays from edge of aperture stop
        aper_idx = self.aper_idx
        aper_surf = self.surfaces[aper_idx]
        aper_z = aper_surf.d.item()
        if aper_surf.is_square:
            aper_r = float(np.sqrt(2)) * aper_surf.r
        else:
            aper_r = aper_surf.r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(SPP_CALC, 1)
            rfov_eff = float(np.arctan(self.r_sensor / self.foclen))
            phi = torch.linspace(-rfov_eff / 2, rfov_eff / 2, SPP_CALC, device=self.device)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to first surface
        surf_range = range(0, self.aper_idx)
        if len(surf_range) == 0:
            # Aperture is the first surface — entrance pupil is at the aperture
            return aper_z, aper_r
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid > 0][:, 0], ray.o[ray.is_valid > 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid > 0][:, 0], ray.d[ray.is_valid > 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small entrance pupil
        if len(intersection_points) == 0:
            print(
                "No intersection points found, use the first surface as entrance pupil."
            )
            avg_pupilr = self.surfaces[0].r
            avg_pupilz = self.surfaces[0].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface as entrance pupil."
                )
                avg_pupilr = self.surfaces[0].r
                avg_pupilz = self.surfaces[0].d.item()

        return avg_pupilz, avg_pupilr

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """Compute the intersection points of 2D lines.

        Closed-form 2×2 solve per pair: A @ [s, t]^T = Oj - Oi where
        A = [[Di_x, -Dj_x], [Di_y, -Dj_y]]. Near-parallel pairs are dropped
        (|det| < 1e-12) rather than returning the lstsq min-norm solution,
        which used to skew the downstream mean.

        Args:
            origins (torch.Tensor): Origins of the lines. Shape: [N, 2] or
                [..., N, 2].
            directions (torch.Tensor): Directions of the lines. Shape: [N, 2]
                or [..., N, 2].

        Returns:
            torch.Tensor: For unbatched input, intersection points with shape
            [K, 2], where K <= N*(N-1)/2. For batched input, dense
            intersection points with shape [..., N*(N-1)/2, 2], with invalid
            or near-parallel pairs filled with NaN.
        """
        batched = origins.ndim > 2
        N = origins.shape[-2]
        if N < 2:
            if batched:
                return origins.new_empty((*origins.shape[:-2], 0, 2))
            else:
                return origins.new_empty((0, 2))

        idx = torch.arange(N, device=origins.device)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi, Oj = origins[..., idx_i, :], origins[..., idx_j, :]
        Di, Dj = directions[..., idx_i, :], directions[..., idx_j, :]
        b = Oj - Oi

        det = Dj[..., 0] * Di[..., 1] - Di[..., 0] * Dj[..., 1]
        mask = det.abs() > 1e-12
        det_safe = torch.where(mask, det, torch.ones_like(det))
        s = (-Dj[..., 1] * b[..., 0] + Dj[..., 0] * b[..., 1]) / det_safe
        t = (-Di[..., 1] * b[..., 0] + Di[..., 0] * b[..., 1]) / det_safe

        P_i = Oi + s.unsqueeze(-1) * Di
        P_j = Oj + t.unsqueeze(-1) * Dj
        P = 0.5 * (P_i + P_j)

        if batched:
            return torch.where(
                mask.unsqueeze(-1),
                P,
                P.new_full(P.shape, float("nan")),
            )
        else:
            return P[mask]

    # ====================================================================================
    # Lens operation
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, foc_dist=float("inf")):
        """Refocus the lens to a depth distance by changing sensor position.

        Args:
            foc_dist (float): focal distance.

        Note:
            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Calculate in-focus sensor position
        d_sensor_new = self.calc_sensor_plane(depth=foc_dist)

        # Update sensor position
        assert d_sensor_new > 0, "Obtained negative sensor position."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def set_fnum(self, fnum):
        """Set F-number by resizing the aperture stop.

        A paraxial estimate is used only to center a candidate aperture-radius
        sweep. Candidate radii are evaluated in one batched non-paraxial trace
        from the aperture stop toward object space, and the radius whose
        entrance-pupil radius is closest to the target is selected.

        Args:
            fnum (float): target F-number.
        """
        target_pupil_r = self.foclen / fnum / 2
        aper_surf = self.surfaces[self.aper_idx]
        aper_r = float(aper_surf.r)

        _, pupil_r = self.calc_entrance_pupil(paraxial=True)
        if pupil_r <= 0 or aper_r <= 0:
            logging.warning(
                f"set_fnum: degenerate pupil (aper_r={aper_r:.4f}, pupil_r={pupil_r:.4f}); "
                "leaving aperture unchanged."
            )
            self.calc_pupil()
            return

        paraxial_aper_r = target_pupil_r / (pupil_r / aper_r)
        if paraxial_aper_r <= 0 or not math.isfinite(paraxial_aper_r):
            logging.warning(
                f"set_fnum: invalid paraxial aperture estimate ({paraxial_aper_r:.4f}); "
                "leaving aperture unchanged."
            )
            self.calc_pupil()
            return

        if self.aper_idx == 0:
            edge_scale = math.sqrt(2) if aper_surf.is_square else 1.0
            aper_surf.r = target_pupil_r / edge_scale
            self.calc_pupil()
            return

        num_candidates = 32
        num_rays = SPP_CALC
        candidate_span = 0.5
        dtype = aper_surf.d.dtype
        device = self.device

        radius_lo = max(paraxial_aper_r * (1.0 - candidate_span), EPSILON)
        radius_hi = max(paraxial_aper_r * (1.0 + candidate_span), radius_lo)
        candidate_radii = torch.linspace(
            radius_lo,
            radius_hi,
            num_candidates,
            device=device,
            dtype=dtype,
        )

        aper_z = aper_surf.d.item()
        edge_scale = math.sqrt(2) if aper_surf.is_square else 1.0
        ray_o = torch.zeros((num_candidates, num_rays, 3), device=device, dtype=dtype)
        ray_o[..., 0] = candidate_radii.unsqueeze(-1) * edge_scale
        ray_o[..., 2] = aper_z

        rfov_eff = float(np.arctan(self.r_sensor / self.foclen))
        phi = torch.linspace(
            -rfov_eff / 2,
            rfov_eff / 2,
            num_rays,
            device=device,
            dtype=dtype,
        )
        ray_d_single = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), dim=-1
        )
        ray_d = ray_d_single.unsqueeze(0).expand(num_candidates, -1, -1).clone()

        ray = Ray(ray_o, ray_d, device=device)
        ray, _ = self.trace(ray, surf_range=range(0, self.aper_idx))

        valid = ray.is_valid > 0
        ray_o_2d = torch.stack([ray.o[..., 0], ray.o[..., 2]], dim=-1)
        ray_d_2d = torch.stack([ray.d[..., 0], ray.d[..., 2]], dim=-1)
        ray_o_2d = torch.where(
            valid.unsqueeze(-1),
            ray_o_2d,
            ray_o_2d.new_full(ray_o_2d.shape, float("nan")),
        )
        ray_d_2d = torch.where(
            valid.unsqueeze(-1),
            ray_d_2d,
            ray_d_2d.new_full(ray_d_2d.shape, float("nan")),
        )

        intersection_points = self.compute_intersection_points_2d(ray_o_2d, ray_d_2d)
        finite = torch.isfinite(intersection_points).all(dim=-1)
        valid_count = finite.sum(dim=-1)
        points = torch.where(
            finite.unsqueeze(-1),
            intersection_points,
            intersection_points.new_zeros(intersection_points.shape),
        )
        avg_points = points.sum(dim=-2) / valid_count.clamp(min=1).unsqueeze(-1)
        candidate_pupil_r = avg_points[:, 0]
        valid_pupil = (valid_count > 0) & (candidate_pupil_r > EPSILON)

        if valid_pupil.any():
            err = (candidate_pupil_r - target_pupil_r).abs()
            err = torch.where(valid_pupil, err, err.new_full(err.shape, float("inf")))
            aper_surf.r = candidate_radii[err.argmin()].item()
        else:
            logging.warning(
                "set_fnum: no valid non-paraxial pupil estimate from aperture candidates; "
                "falling back to paraxial aperture estimate."
            )
            aper_surf.r = paraxial_aper_r

        self.calc_pupil()

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov_eff, fnum):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        Args:
            rfov_eff (float): half diagonal-FoV in radian.
            fnum (float): F number.
        """
        if rfov_eff > math.pi:
            self.rfov_eff = rfov_eff / 180.0 * math.pi
        else:
            self.rfov_eff = rfov_eff

        self.rfov = self.rfov_eff
        self.foclen = self.r_sensor / math.tan(self.rfov_eff)
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
        self.fnum = fnum
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].update_r(float(aper_r))

        # Update pupil after setting aperture radius
        self.calc_pupil()

        if hasattr(self, "sensor_cra_cap"):
            self.chief_ray_angle_max = min(
                math.degrees(self.rfov_eff), float(self.sensor_cra_cap)
            )

    @torch.no_grad()
    def set_fov(self, rfov_eff):
        """Set half-diagonal field of view as a design target.

        Unlike ``calc_fov()`` which derives FoV from focal length and sensor
        size, this method directly assigns the target FoV for lens optimisation.

        Args:
            rfov_eff (float): Half-diagonal FoV in radians.
        """
        self.rfov_eff = rfov_eff
        self.rfov = rfov_eff
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
        if hasattr(self, "sensor_cra_cap"):
            self.chief_ray_angle_max = min(
                math.degrees(self.rfov_eff), float(self.sensor_cra_cap)
            )
