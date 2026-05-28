"""Load an existing lens and optimise it using Adam with the RMS spot-size objective.

Usage:
    python 2_lens_optim.py
"""

import logging
import os
import random
import string
from datetime import datetime

import torch

from deeplens import GeoLens
from deeplens.utils_deeplens import set_logger, set_seed


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Result directory
    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-lens-optim-{tag}"
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)
    logging.info(f"Device: {device}")

    # Load lens
    lens = GeoLens(filename="./datasets/lens_zoo/cellphone.json")
    lens.analysis(save_name=f"{result_dir}/initial")
    logging.info(f"Loaded lens: FoV={lens.rfov:.4f} rad, F/{lens.fnum:.2f}")

    # Optimise
    lens.optimize(
        lrs=[1e-3, 1e-3, 1e-3, 1e-4],
        iterations=10000,
        test_per_iter=100,
        shape_control=True,
        optim_mat=True,
        result_dir=result_dir,
    )

    # Final result
    lens.prune_surf()
    lens.post_computation()
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    logging.info(f"Done. Results in {result_dir}")


if __name__ == "__main__":
    main()
