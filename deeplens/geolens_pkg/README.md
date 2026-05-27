# GeoLens Package

The `geolens_pkg` is a sub-package of DeepLens (under `optics/`) that provides a comprehensive suite of tools for the classical evaluation, optimization, and analysis of geometric lens systems (`GeoLens`). The functionalities are designed to be accurate and are aligned with industry-standard optical design software like Zemax.

## Key Features

This package offers a wide range of optical performance evaluation capabilities:

-   **Spot Diagram Analysis:** Generate and visualize spot diagrams at various field angles to assess aberrations.
-   **RMS Spot Error:** Calculate RMS spot size maps across different wavelengths and field points.
-   **Distortion Analysis:** Compute and plot distortion maps and curves to quantify image deformation.
-   **Modulation Transfer Function (MTF):** Evaluate the spatial frequency response of the lens system to determine its resolution and contrast performance.
-   **Vignetting Analysis:** Calculate and visualize the reduction in image brightness at the periphery of the field.
-   **3D Visualization:** Render 3D views of the lens system for better understanding of its physical layout.
-   **Tolerance Analysis:** Tools for assessing the impact of manufacturing and assembly errors on lens performance.
-   **Optimization:** Utilities for optimizing lens designs based on various performance metrics.

## Modules

The package is organized into the following modules:

**Evaluation**
-   `eval.py`: Classical optical performance evaluation — spot diagrams, RMS spot maps, distortion, MTF, field curvature, vignetting, wavefront error.
-   `eval_seidel.py`: Seidel aberration coefficient computation.
-   `eval_tolerance.py`: Tolerance analysis for manufacturing/assembly errors.
-   `psf_compute.py`: Point-spread-function computation (geometric, coherent, Huygens).

**Optimization** — three files sharing the `optim*` prefix, covering the full lens-design workflow:
-   `optim_init.py`: Stage 0 — build a flat starting-point `GeoLens` from specs (`create_lens`, `create_surface`).
-   `optim.py`: Gradient-based optimization — `optimize`, `curriculum_design`, loss functions (`loss_rms`, `loss_reg`, `loss_bound`, `loss_profile`, ...), and `init_constraints`.
-   `optim_ops.py`: Discrete surface edits used between optimization stages — `add_aspheric`, `increase_aspheric_order`, `prune_surf`, `correct_shape`, `match_materials`.

**I/O and Visualization**
-   `io.py`: Lens JSON load/save and serialization helpers.
-   `vis.py`: 2-D lens layout plotting and ray-path visualization.
-   `vis3d.py`: 3-D lens rendering.
