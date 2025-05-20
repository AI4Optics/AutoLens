""" 
Automated lens design from scratch. This code uses classical RMS spot size for lens design, which is much faster than image-based lens design.

Technical Paper:
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint 2023.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import logging
import os
import random
import string
from datetime import datetime

import numpy as np
import torch
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
from deeplens.optics.ray import Ray
from monitor import *
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def config():
    """ Config file for training.
    """
    # Config file
    with open('configs/autolens.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + '-AutoLens-RMS-' + random_string
    result_dir = f'./results/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    args['result_dir'] = result_dir
    
    if args['seed'] is None:
        seed = random.randint(0, 100)
        args['seed'] = seed
    set_seed(args['seed'])
    
    # Log
    set_logger(result_dir)
    logging.info(f'EXP: {args["EXP_NAME"]}')

    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} {torch.cuda.get_device_name(0)} GPU(s)')

    return args

def curriculum_design(self, lrs=[5e-4, 1e-4, 0.1, 1e-4], decay=0.02, iterations=5000, test_per_iter=100, importance_sampling=True, result_dir='./results'):
        """ Optimize the lens by minimizing rms errors.
        """
        # Preparation
        depth = DEPTH
        num_grid = 21
        spp = 512
        
        centroid = False
        sample_rays_per_iter = 5 * test_per_iter if centroid else test_per_iter
        aper_start = self.surfaces[self.aper_idx].r * 0.5
        aper_final = self.surfaces[self.aper_idx].r
        
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(f'lr:{lrs}, decay:{decay}, iterations:{iterations}, spp:{spp}, grid:{num_grid}.')

        optimizer = self.get_optimizer(lrs, decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iterations//10, num_training_steps=iterations)
        # Training
        pbar = tqdm(total=iterations+1, desc='Progress', postfix={'loss': 0})
        for i in range(iterations+1):
            # =====> Evaluate the lens
            if i % test_per_iter == 0:
                # Change aperture, curriculum learning
                aper_r = min((aper_final - aper_start) * (i / iterations * 1.1) + aper_start, aper_final)
                self.surfaces[self.aper_idx].r = aper_r
                self.fnum = self.foclen / aper_r / 2
                
                # Correct shape and evaluate
                if i > 0:   
                    self.correct_shape()
                # self.write_lens_json(f'{result_dir}/iter{i}.json')
                # self.analysis(f'{result_dir}/iter{i}', zmx_format=True, plot_invalid=True, multi_plot=False)

                    
            # =====> Compute centriod and sample new rays
            if i % sample_rays_per_iter == 0:
                with torch.no_grad():
                    # Sample rays
                    scale = self.calc_scale_pinhole(depth)
                    rays_backup, parallel_rays_backup, angle_rays_backup = [], [], []
                    for wv in WAVE_RGB:
                        ray = self.sample_point_source(M=num_grid, R=self.sensor_size[0]/2*scale, depth=depth, spp=spp, pupil=True, wvln=wv, importance_sampling=importance_sampling)
                        rays_backup.append(ray)
                        p_rays = self.sample_parallel(fov=0.0, M=num_grid, wvln=wv, entrance_pupil=True)
                        parallel_rays_backup.append(p_rays)
                        scale = self.calc_scale_pinhole(depth)
                        a_rays = self.sample_point_source(M=num_grid, spp=spp, depth=DEPTH, R=scale * self.sensor_size[0] / 2, pupil=True)
                        angle_rays_backup.append(a_rays)
                    # Calculate ray centers
                    if centroid:
                        center_p = - self.psf_center(point=ray.o[0, ...], method='chief_ray')
                    else:
                        center_p = - self.psf_center(point=ray.o[0, ...], method='pinhole')
                    merged_point_rays = merge_rays(rays_backup)
                    merged_parallel_rays = merge_rays(parallel_rays_backup)
                    merged_angle_rays = merge_rays(angle_rays_backup)
                    point_rays_num, parallel_rays_num, angle_rays_num = merged_point_rays.o.shape[1], 1, merged_angle_rays.o.shape[1]
                    concated_rays = concat_rays(merged_point_rays, merged_parallel_rays, merged_angle_rays)
            all_rays = concated_rays.clone()
            all_rays, _, _ = self.trace(all_rays, use_flash_method=True)
            point_rays, parallel_rays, angle_rays = split_rays(all_rays, point_rays_num, parallel_rays_num, angle_rays_num)

            xy = point_rays.project_to(self.d_sensor)
            # xy: (C,SPP,M,M,D)
            xy_norm = (xy - center_p.unsqueeze(0)) * point_rays.ra.unsqueeze(-1)

            weight_mask = (xy_norm.clone().detach() ** 2).sum([1, -1]) / (point_rays.ra.sum([1]) + EPSILON)  # Use L2 error as weight mask
            weight_mask /= weight_mask.mean([1, 2])[:, None, None]  # shape of [M, M]
            loss_rms = torch.sqrt(
                torch.sum((xy_norm**2 + EPSILON).sum(-1) * weight_mask.unsqueeze(1)) / (torch.sum(point_rays.ra) + EPSILON)
            )
            # Regularization

            p = parallel_rays.project_to(self.d_sensor)
            # Calculate RMS spot size as loss function
            rms_size = torch.sqrt(torch.sum((p**2 + EPSILON) * parallel_rays.ra.unsqueeze(-1)) / (torch.sum(parallel_rays.ra) + EPSILON))
            loss_focus = max(rms_size, 0.005)

            angle_loss = torch.sum(angle_rays.obliq * angle_rays.ra) / (torch.sum(angle_rays.ra) + EPSILON)
            angle_loss = min(angle_loss, 0.6)

            loss_focus_avg = loss_focus

            if self.is_cellphone:
                loss_reg = 2.0 * loss_focus_avg + 0.05 * angle_loss + self.loss_self_intersec_new(dist_bound=0.1, thickness_bound=0.3, flange_bound=0.5)
            else:
                loss_reg = 2.0 * loss_focus_avg + self.loss_self_intersec_new(dist_bound=0.1, thickness_bound=2.0, flange_bound=10.0)

            w_reg = 0.1
            L_total = loss_rms + w_reg * loss_reg

            # Gradient-based optimization
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(rms=loss_rms.item())
            pbar.update(1)
        pbar.close()

def concat_rays(point_rays, parellel_rays, angle_rays):

    # #print(point_rays.o.shape, parellel_rays.o.shape, angle_rays.o.shape)
    o = torch.concat([point_rays.o, parellel_rays.o.unsqueeze(1), angle_rays.o], dim=1)
    d = torch.concat([point_rays.d, parellel_rays.d.unsqueeze(1), angle_rays.d], dim=1)
    ra = torch.concat([point_rays.ra, parellel_rays.ra.unsqueeze(1), angle_rays.ra], dim=1)
    obliq = torch.concat([point_rays.obliq, parellel_rays.obliq.unsqueeze(1), angle_rays.obliq], dim=1)
    # #print(o.shape, d.shape)
    concat_ray = Ray(o, d, device=o.device)
    concat_ray.ra = ra
    concat_ray.obliq = obliq
    return concat_ray


def split_rays(all_rays, point_num, parellel_num, angle_num):
    point_o, parellel_o, angle_o = all_rays.o.split([point_num, parellel_num, angle_num], dim=1)
    point_d, parellel_d, angle_d = all_rays.d.split([point_num, parellel_num, angle_num], dim=1)
    point_ra, parellel_ra, angle_ra = all_rays.ra.split([point_num, parellel_num, angle_num], dim=1)
    point_obliq, parellel_obliq, angle_obliq = all_rays.obliq.split([point_num, parellel_num, angle_num], dim=1)
    point_rays = Ray(point_o, point_d, device=point_o.device)
    point_rays.ra, point_obliq = point_ra, point_obliq
    parellel_rays = Ray(parellel_o, parellel_d, device=parellel_o.device)
    parellel_rays.ra, parellel_obliq = parellel_ra, parellel_obliq
    angle_rays = Ray(angle_o, angle_d, device=angle_o.device)
    angle_rays.ra, angle_obliq = angle_ra, angle_obliq
    return point_rays, parellel_rays, angle_rays


def merge_rays(rays_backup):
    o = torch.stack([ray.o for ray in rays_backup])
    d = torch.stack([ray.d for ray in rays_backup])
    ra = torch.stack([ray.ra for ray in rays_backup])
    obliq = torch.stack([ray.obliq for ray in rays_backup])
    # ##print(o.shape, d.shape)
    merged_ray = Ray(o, d, device=o.device)
    merged_ray.ra = ra
    merged_ray.obliq = obliq
    return merged_ray

if __name__=='__main__':
    args = config()
    result_dir = args['result_dir']
    device = args['device']

    # Bind function
    GeoLens.curriculum_design = curriculum_design

    # ===> Create a cellphone lens
    lens = create_cellphone_lens(hfov=args['HFOV'], imgh=args['DIAG'], fnum=args['FNUM'], lens_num=args['lens_num'], save_dir=result_dir)
    lens.set_target_fov_fnum(hfov=args['HFOV'], fnum=args['FNUM'], imgh=args['DIAG'])
    logging.info(f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.')
    
    # # ===> Create a camera lens
    # lens = create_camera_lens(foclen=args['FOCLEN'], imgh=args['DIAG'], fnum=args['FNUM'], lens_num=args['lens_num'], save_dir=result_dir)
    # lens.set_target_fov_fnum(hfov=float(np.arctan(args['DIAG'] / args['FOCLEN'] / 2)), fnum=args['FNUM'], imgh=args['DIAG'])
    # logging.info(f'==> Design target: FOCLEN {round(args["FOCLEN"], 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}')
    
    # =====> 2. Curriculum learning with RMS errors
    lrs = [float(lr) for lr in args['lrs']]
    lens.curriculum_design(lrs=lrs, decay=0.01, iterations=5000, test_per_iter=100, result_dir=args['result_dir'])

    # Need to train more for the best optical performance

    # =====> 3. Analyze final result
    lens.prune_surf(outer=0.05)
    lens.post_computation()

    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
    lens.write_lens_json(f'{result_dir}/final_lens.json')
    lens.analysis(save_name=f'{result_dir}/final_lens', zmx_format=True)

    # =====> 4. Create video
    create_video_from_images(f'{result_dir}', f'{result_dir}/autolens.mp4', fps=10)