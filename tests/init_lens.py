import random
import string
from datetime import datetime

import yaml
from deeplens import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    WAVE_RGB,
    GeoLens,
    create_camera_lens,
    create_cellphone_lens,
    create_video_from_images,
    set_logger,
    set_seed,
)
from deeplens.optics import *
from transformers import get_cosine_schedule_with_warmup


def config():
    """Config file for training."""
    # Config file
    with open("configs/autolens.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-AutoLens-RMS-" + random_string
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    if args["seed"] is None:
        seed = random.randint(0, 100)
        args["seed"] = seed
    set_seed(args["seed"])

    # Log
    # set_logger(result_dir)
    logging.info(f'EXP: {args["EXP_NAME"]}')

    # Device
    num_gpus = torch.cuda.device_count()
    args["num_gpus"] = num_gpus
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    args["device"] = device
    logging.info(f"Using {num_gpus} {torch.cuda.get_device_name(0)} GPU(s)")

    return args


def init():
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]
    # device = torch.device("cpu")
    # torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    torch.set_printoptions(15)
    # ===> Create a cellphone lens
    lens = create_cellphone_lens(hfov=args["HFOV"], imgh=args["DIAG"], fnum=args["FNUM"], lens_num=args["lens_num"],save_dir=result_dir ,save=True)
    lens.set_target_fov_fnum(hfov=args["HFOV"], fnum=args["FNUM"], imgh=args["DIAG"])
    logging.info(
        f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.'
    )
    depth = -20000.0
    num_grid = 21
    spp = 512
    iterations = 5000
    i = 0
    aper_start = lens.surfaces[lens.aper_idx].r * 0.5
    aper_final = lens.surfaces[lens.aper_idx].r
    lrs = [5e-4, 1e-4, 0.1, 1e-4]
    decay = 0.02
    optimizer = lens.get_optimizer(lrs, decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iterations // 10, num_training_steps=iterations)

    aper_r = min((aper_final - aper_start) * (i / iterations * 1.1) + aper_start, aper_final)
    lens.surfaces[lens.aper_idx].r = aper_r
    lens.fnum = lens.foclen / aper_r / 2
    rays_backup: list[Ray] = []
    for wv in WAVE_RGB:
        scale = lens.calc_scale_pinhole(depth)
        # rays_backup.clear()
        ray = lens.sample_point_source(
            M=num_grid,
            R=lens.sensor_size[0] / 2 * scale,
            depth=depth,
            spp=spp,
            pupil=True,
            wvln=wv,
            importance_sampling=True,
        )
        ray.propagate_to(lens.surfaces[0].d - 10.0)
        rays_backup.append(ray)

    ray = rays_backup[0].clone()
    return lens, ray, rays_backup
