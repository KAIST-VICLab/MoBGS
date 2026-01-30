#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os, sys

import torch
from random import randint
from utils.loss_utils import *
from gaussian_renderer import render, network_gui, get_flow, get_flow_static
import sys
from scene import Scene, GaussianModel, deformation
from utils.general_utils import safe_state

from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, blceParams
from utils.timer import Timer

from utils.scene_utils import render_training_image
import copy
from PIL import Image
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud, getWorld2View2
from main_utils import *
from scene.blce import blceKernel
import copy
from helper_train import controlgaussians
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_pixels(image_size_x, image_size_y, use_center = None):
    """Return the pixel at center or corner."""
    xx, yy = np.meshgrid(
        np.arange(image_size_x, dtype=np.float32),
        np.arange(image_size_y, dtype=np.float32),
    )
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset


def scene_initialization(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         dyn_gaussians, stat_gaussians, scene, stage, tb_writer, train_iter, timer, drop=False, check_seed=False):

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams] # Large CPU usage
    viewpoint_stack = [i for i in train_cams] # Large CPU usage

    with torch.no_grad():
        points_list, colors_list = [], []
        stat_times, stat_colors, stat_points = [], [], []
        for IDX in range(len(viewpoint_stack)):
            if scene.dataset_type == "PanopticSports":
                image_tensor = viewpoint_stack[IDX]['image'][None].cuda()
            else:
                image_tensor = viewpoint_stack[IDX].original_image[None].cuda()
            B, C, H, W = image_tensor.shape

            CVD = viewpoint_stack[IDX].depth[None].cuda()
            pred_R = torch.from_numpy(viewpoint_stack[IDX].R.T[None]).cuda()
            pred_T = torch.from_numpy(viewpoint_stack[IDX].T[None]).cuda()

            K_tensor = torch.zeros(1, 3, 3).type_as(image_tensor)
            K_tensor[:, 0, 0] = viewpoint_stack[IDX].focal
            K_tensor[:, 1, 1] = viewpoint_stack[IDX].focal
            K_tensor[:, 0, 2] = float(viewpoint_stack[IDX].metadata.principal_point_x)
            K_tensor[:, 1, 2] = float(viewpoint_stack[IDX].metadata.principal_point_y)
            K_tensor[:, 2, 2] = float(1)
            w2c_target = torch.cat((pred_R, pred_T[:, :, None]), -1)

            accum_error = 0
            for cam_id, view_pt in enumerate(viewpoint_stack):
                if scene.dataset_type == "PanopticSports":
                    ref_image_tensor = view_pt['image'][None].cuda()
                else:
                    ref_image_tensor = view_pt.original_image[None].cuda()

                ref_R = torch.from_numpy(view_pt.R.T[None]).cuda()
                ref_T = torch.from_numpy(view_pt.T[None]).cuda()
                w2c_ref = torch.cat((ref_R, ref_T[:, :, None]), -1)

                warped_ref, grid_ref = deformation.inverse_warp_rt1_rt2(ref_image_tensor, CVD, w2c_target, w2c_ref,
                                                                        K_tensor, torch.inverse(K_tensor),
                                                                        ret_grid=True)

                out_mask = (torch.sum(warped_ref, dim=1, keepdim=True) > 0).type_as(warped_ref)
                accum_error += torch.mean(out_mask * torch.abs(warped_ref - image_tensor), dim=1, keepdim=True)

            mean_err = torch.mean(accum_error)
            accum_error = (accum_error > mean_err).type_as(accum_error)
            p_im = accum_error.detach().squeeze().cpu().numpy()
            im = Image.fromarray(np.rint(255 * p_im).astype(np.uint8))

            points = deformation.points_from_DRTK(CVD, w2c_target, K_tensor)
            points = torch.permute(points, (0, 2, 1))  # B, N, 3

            # Make point cloud
            colors = torch.permute(image_tensor, (0, 2, 3, 1))  # B, H, W, 3

            #error init
            colors_list.append(colors[0].detach().cpu().numpy())
            points_list.append(points[0].view(H,W,3).detach().cpu().numpy())
            colors = colors[0].view(-1, 3).detach().cpu().numpy()
            points = points[0].view(-1, 3).detach().cpu().numpy()
            coords_2d = get_pixels(W, H)
            
            if IDX == 0:
                accum_error = accum_error[0].squeeze(0).detach().cpu().numpy().reshape(-1)
                motion_mask = viewpoint_stack[IDX].mask.cuda()
                motion_error = motion_mask.squeeze(0).cpu().numpy().reshape(-1)

                # N_pts = opt.stat_npts
                stat_colors.append(colors[(accum_error == 0) & (motion_error==0), :])
                stat_points.append(points[(accum_error == 0) & (motion_error==0), :])               
                stat_times.append(torch.ones(stat_colors[-1].shape[0], 1) * viewpoint_stack[IDX].time)

                N_pts = opt.dyn_npts
                dyn_colors = colors[(accum_error == 1) & (motion_error==1), :]
                dyn_points = points[(accum_error == 1) & (motion_error==1), :]
                dyn_coords_2d = coords_2d.reshape(-1,2)[(accum_error == 1) & (motion_error==1), :]
                if dyn_colors.shape[0] < N_pts:
                    select_inds = random.choices(range(dyn_colors.shape[0]), k=N_pts)
                else:
                    select_inds = random.sample(range(dyn_colors.shape[0]), N_pts)

                dyn_time = torch.ones(dyn_colors[select_inds].shape[0], 1) * viewpoint_stack[IDX].time
                dyn_color = dyn_colors[select_inds]
                dyn_point = dyn_points[select_inds]
                dyn_coord_2d = dyn_coords_2d[select_inds] # N_pts, 2
                
            else:
                accum_error = accum_error[0].squeeze(0).detach().cpu().numpy().reshape(-1)
                motion_mask = viewpoint_stack[IDX].mask.cuda()
                motion_error = motion_mask.squeeze(0).cpu().numpy().reshape(-1)

                # N_pts = opt.stat_npts
                stat_colors.append(colors[(accum_error == 0) & (motion_error==0), :])
                stat_points.append(points[(accum_error == 0) & (motion_error==0), :])
                stat_times.append(torch.ones(stat_colors[-1].shape[0], 1) * viewpoint_stack[IDX].time)
                
        N_pts = opt.stat_npts
        stat_colors = np.concatenate(stat_colors, axis=0)
        stat_points = np.concatenate(stat_points, axis=0)
        stat_times = torch.cat(stat_times, dim=0).numpy()
        select_inds = random.sample(range(stat_colors.shape[0]), N_pts)
        
        stat_color = stat_colors[select_inds]
        stat_point = stat_points[select_inds]
        stat_time = stat_times[select_inds]
                
        # compute dyn tracker
        tracklet = viewpoint_stack[0].tracklet
        start_tracklet = tracklet[0] # time = 0 (cannonical)
        
        
        chunk = dyn_coord_2d.shape[0] // 10
        dyn_tracjectory = []
        points_list = torch.from_numpy(np.stack(points_list, axis=0)).permute(0, 3, 1, 2)
        colors_list = torch.from_numpy(np.stack(colors_list, axis=0)).permute(0, 3, 1, 2)
        
        for i in range(0, dyn_coord_2d.shape[0], chunk):
            dyn_tracklet_index = torch.square(torch.from_numpy(dyn_coord_2d[i:i+chunk, None]).cuda() - start_tracklet[None]).sum(-1).argmin(-1)
            dyn_tracklet = torch.gather(tracklet[None].expand(dyn_coord_2d[i:i+chunk].shape[0], -1, -1, -1), 2, dyn_tracklet_index[:, None, None, None].expand(-1, tracklet.shape[0], -1, 2)).squeeze(2) # N_dyn_pts, N_time, 2
            dyn_tracklet = dyn_tracklet.permute(1, 0, 2)[:, None]
            dyn_tracklet[..., 0] /= W
            dyn_tracklet[..., 1] /= H
            norm_dyn_tracklet = dyn_tracklet * 2 - 1.0 # norm to -1, 1
            dyn_tracjectory.append(torch.nn.functional.grid_sample(points_list.cuda(), norm_dyn_tracklet, mode="nearest").squeeze().permute(2,0,1)) # N_pts, N_times, 3
 
        dyn_tracjectory = torch.cat(dyn_tracjectory, dim=0) # N_pts, N_times, 3

        new_dyn_tracjectory = dyn_tracjectory
        new_dyn_color = dyn_color
        new_dyn_point = dyn_point
        new_dyn_time = dyn_time


        stat_pc = BasicPointCloud(colors=stat_color, points=stat_point, normals=None, times=stat_time)
        dyn_pc = BasicPointCloud(colors=new_dyn_color, points=new_dyn_point, normals=None, times=new_dyn_time)
        return stat_pc, dyn_pc, new_dyn_tracjectory


def scene_reconstruction(dataset, opt, hyper, pipe, blceopt, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         dyn_gaussians, stat_gaussians, scene, stage, tb_writer, train_iter, timer, drop=False, check_seed=False):

    flag_d = 0
    flag_s = 0
    densify = 2

    BEST_PSNR, BEST_ITER = 0, 0
    all_test_poses = None
    first_iter = 0
    dyn_gaussians.training_setup(opt, stage=stage)
    stat_gaussians.training_setup(opt)

    currentxyz_d = dyn_gaussians._xyz 
    maxx_d, maxy_d, maxz_d = torch.amax(currentxyz_d[:,0]), torch.amax(currentxyz_d[:,1]), torch.amax(currentxyz_d[:,2])# z wrong...
    minx_d, miny_d, minz_d = torch.amin(currentxyz_d[:,0]), torch.amin(currentxyz_d[:,1]), torch.amin(currentxyz_d[:,2])

    maxbounds_d = [maxx_d, maxy_d, maxz_d]
    minbounds_d = [minx_d, miny_d, minz_d]

    currentxyz_s = stat_gaussians._xyz 
    maxx_s, maxy_s, maxz_s = torch.amax(currentxyz_s[:,0]), torch.amax(currentxyz_s[:,1]), torch.amax(currentxyz_s[:,2])# z wrong...
    minx_s, miny_s, minz_s = torch.amin(currentxyz_s[:,0]), torch.amin(currentxyz_s[:,1]), torch.amin(currentxyz_s[:,2])

    maxbounds_s = [maxx_s, maxy_s, maxz_s]
    minbounds_s = [minx_s, miny_s, minz_s]

    if stage == "fine":
        bg_color = [1, 1, 1, -10] if dataset.white_background else [0, 0, 0, -10]
    else:
        bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viewpoint_stack_ids = []
    ema_loss_for_log_photo = 0.0
    ema_loss_for_log_reg = 0.0
    ema_loss_for_log_mask = 0.0
    ema_loss_for_log_path_rot = 0.0
    ema_loss_for_log_path_trans = 0.0
    ema_loss_for_log_flow = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams] # Large CPU usage
    viewpoint_stack = [i for i in train_cams] # Large CPU usage

    blcekernel = blceKernel(num_views=len(viewpoint_stack),
                            view_dim=blceopt.view_dim,
                            num_warp=blceopt.num_warp,
                            method=blceopt.method,
                            adjoint=blceopt.adjoint,
                            iteration=opt.iterations).cuda()

    print(f"SSIM: {opt.lambda_dssim}")

    # Get GT cam to worlds for testing
    gt_train_pose_list = []
    for view_p in viewpoint_stack:
        gt_Rt = getWorld2View2(view_p.R, view_p.T, view_p.trans, view_p.scale)
        gt_C2W = np.linalg.inv(gt_Rt)
        gt_train_pose_list.append(gt_C2W)

    gt_test_pose_list = []
    for view_p in my_test_cams:
        gt_Rt = getWorld2View2(view_p.R, view_p.T, view_p.trans, view_p.scale)
        gt_C2W = np.linalg.inv(gt_Rt)
        gt_test_pose_list.append(gt_C2W)

    batch_size = opt.batch_size

    # sliding_window_size = 2
    # sliding_window_size = 0
    print("data loading done")

    mask_dice_loss = BinaryDiceLoss(from_logits=False)
    
    for iteration in range(first_iter, final_iter + 1):
        if check_seed:
            if stage != 'warm' and iteration > 5000:
                return BEST_PSNR, BEST_ITER
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = \
                    render(custom_cam, dyn_gaussians, stat_gaussians, pipe, background, scaling_modifer, stage=stage,
                           cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).
                                                 byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        dyn_gaussians.update_learning_rate(iteration)
        stat_gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration > 2000:
            dyn_gaussians.oneupSHdegree()
            stat_gaussians.oneupSHdegree()

        # Store the imgs/error/depths/disps/normals to visualize in this dict (use detach())
        debug_dict = {}

        # Pick a random Camera
        viewpoint_cams = []
        fwd_viewpoint_cams = []
        bwd_viewpoint_cams = []
        
        idx = 0
        
        while idx < batch_size:
            if not viewpoint_stack_ids:
                viewpoint_stack_ids = list(range(len(viewpoint_stack))) 

            id = randint(0, len(viewpoint_stack_ids) - 1)
            id = viewpoint_stack_ids.pop(id)
            viewpoint_cams.append(viewpoint_stack[id])
            idx += 1

            if id == 0:
                fwd_id, bwd_id = id+1, id
                fwd_viewpoint_cams.append(viewpoint_stack[fwd_id])
                bwd_viewpoint_cams.append(viewpoint_stack[bwd_id])
            elif id == len(viewpoint_stack) - 1:
                fwd_id, bwd_id = id, id - 1
                fwd_viewpoint_cams.append(viewpoint_stack[fwd_id])
                bwd_viewpoint_cams.append(viewpoint_stack[bwd_id])
            else:
                fwd_id, bwd_id = id + 1, id - 1
                fwd_viewpoint_cams.append(viewpoint_stack[fwd_id])
                bwd_viewpoint_cams.append(viewpoint_stack[bwd_id])

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        s_images = []
        d_images = []
        gt_images = []

        pred_normals = []
        gt_normals = []
        gt_pixels = []
        gt_depths = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        depth_list = []
        s_depth_list = []
        alpha_list = []
        Ks = []

        motion_masks = []
        d_alphas = []
        d_depths = []
        s_alphas = []

        time = []

        dmeans_3d_final_list = []
        image_ori_list = []

        labels = []
        centroids = []
        means3d = []

        warped_rotations = []
        warped_translations = []
        view_cams_R = []
        view_cams_R_fwd = []
        view_cams_R_bwd = []
        
        view_cams_t = []
        view_cams_t_fwd = []
        view_cams_t_bwd = []

        latent_img_final_list = []
        exp2mid_coord_final_list = []
        mid2exp_coord_final_list = []
        latent_alpha_final_list = []

        for n_batch in range(len(viewpoint_cams)):
            time.append(torch.tensor(viewpoint_cams[n_batch].time).float().cuda())
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cams[n_batch].original_image.cuda()
            else:
                gt_image = viewpoint_cams[n_batch]['image'].cuda()

            gt_images.append(gt_image.unsqueeze(0))
            if viewpoint_cams[n_batch].a_chann is not None:
                alpha_list.append(viewpoint_cams[n_batch].a_chann[None].cuda())

            gt_normals.append(viewpoint_cams[n_batch].normal[None].cuda())
            gt_depths.append(viewpoint_cams[n_batch].depth[None].cuda())
            pixels = viewpoint_cams[n_batch].metadata.get_pixels(normalize=True)
            pixels = torch.from_numpy(pixels).cuda()
            gt_pixels.append(pixels)

        if len(alpha_list) > 0:
            alpha_tensor = torch.cat(alpha_list, 0)
        else:
            alpha_tensor = 1
        gt_image_tensor = torch.cat(gt_images, 0)
        B, C, H, W = gt_image_tensor.shape
        gt_depth_tensor = torch.cat(gt_depths, 0)
        
        no_stat_gs = stat_gaussians.get_xyz.shape[0]
        no_dyn_gs = dyn_gaussians.get_xyz.shape[0]

        for n_batch, viewpoint_cam in enumerate(viewpoint_cams):
            camera_metadata = viewpoint_cam.metadata
            K = torch.zeros(3, 3).type_as(gt_image_tensor)
            K[0, 0] = float(camera_metadata.focal_length)
            K[1, 1] = float(camera_metadata.focal_length)
            K[0, 2] = float(camera_metadata.principal_point_x)
            K[1, 2] = float(camera_metadata.principal_point_y)
            K[2, 2] = float(1)
            Ks.append(K[None])

            if stage != "warm":
                render_pkg = render(viewpoint_cam, stat_gaussians, dyn_gaussians, pipe, background, stage=stage,
                                    cam_type=scene.dataset_type, get_static=True, get_dynamic=True,iter_fact=iteration,
                                    ref_wc=None, flow=None, target_ts = viewpoint_cam.target_ts, target_w2cs=None)

                s_images.append(render_pkg["s_render"].unsqueeze(0))
                s_depth_list.append(render_pkg["s_depth"].unsqueeze(0))
                pred_image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
                ori_visibility_filter, ori_radii = render_pkg["visibility_filter"], render_pkg["radii"]
                pred_depth = render_pkg["depth"]

                # if iteration < blceopt.start_warp:
                #     radii_list.append(radii.unsqueeze(0))
                #     visibility_filter_list.append(visibility_filter.unsqueeze(0))
                #     viewspace_point_tensor_list.append(viewspace_point_tensor)
                
                dmeans_3d_final_list.append(render_pkg["means_3d_final"][no_stat_gs:].unsqueeze(0))

                # labels.append(render_pkg["labels"].unsqueeze(0))
                # centroids.append(render_pkg["centroids"].unsqueeze(0))
                # means3d.append(render_pkg["means_3d"].unsqueeze(0))

                d_alphas.append(render_pkg["d_alpha"].unsqueeze(0))
                d_depths.append(render_pkg["d_depth"].unsqueeze(0))
                s_alphas.append(render_pkg["s_alpha"].unsqueeze(0))

                # if iteration > blceopt.start_warp:
                image_ori = pred_image
                depth_ori = pred_depth
                image_ori_list.append(image_ori.unsqueeze(0))
                                
                if iteration > blceopt.start_warp:
                    warped_cams, exposure_time = blcekernel.get_warped_cams(viewpoint_cam, fwd_viewpoint_cams[n_batch], bwd_viewpoint_cams[n_batch])
                    
                    if iteration > blceopt.start_warp_exposure and iteration % 10 == 0:                  
                        with torch.no_grad():          
                            start_latent = warped_cams[0]
                            end_latent = warped_cams[-1]
                            
                            _, rendered_cam_flow = get_flow_static(bwd_viewpoint_cams[n_batch], fwd_viewpoint_cams[n_batch], viewpoint_cam, scene.stat_gaussians, scene.dyn_gaussians,  pipe, background)
                            _, rendered_latent_flow = get_flow_static(start_latent, end_latent, viewpoint_cam, scene.stat_gaussians, scene.dyn_gaussians,  pipe, background)
                            
                            cam_flow_mag = torch.norm(rendered_cam_flow, dim=-1)
                            latent_flow_mag = torch.norm(rendered_latent_flow, dim=-1)
                            
                            valid_id = cam_flow_mag > torch.quantile(cam_flow_mag, 0.01)
                            cam_flow_mag = cam_flow_mag[valid_id]
                            latent_flow_mag = latent_flow_mag[valid_id]
                            
                            new_exposure_time = torch.median(latent_flow_mag /cam_flow_mag)
                            if viewpoint_cam.uid == 0 or viewpoint_cam.uid == len(viewpoint_cams)-1:
                                new_exposure_time = new_exposure_time * 0.5
                            blcekernel.model.update_exposure_time(viewpoint_cam.uid, new_exposure_time)
                    half = len(warped_cams) // 2
                    rendered_images = []
                    rendered_depths = []
                                        
                    warped_rotations_batch = []
                    warped_translations_batch = []
                    warped_radii = []
                    warped_visibility_filter = []
                    warped_viewspace_point_tensor = []
                    for latent_sharp_id, cam in enumerate(warped_cams):
                        if iteration > blceopt.start_warp_dynamic:
                            delta_exposure = exposure_time[latent_sharp_id]
                        else:
                            delta_exposure = 0
                        
                        if latent_sharp_id == half:
                            rendered_images.append(image_ori)
                            rendered_depths.append(depth_ori)
                        else:
                            render_pkg = render(cam, stat_gaussians, dyn_gaussians, pipe, background, stage=stage,
                                                cam_type=scene.dataset_type, get_static=True, get_dynamic=True,iter_fact=iteration,
                                                ref_wc=None, flow=None, target_ts = None, target_w2cs=None, delta_exposure=delta_exposure)
                            rendered_images.append(render_pkg["render"])
                            rendered_depths.append(render_pkg["depth"])
                        warped_rotations_batch.append(warped_cams[latent_sharp_id].R)
                                                
                                                
                        
                        if latent_sharp_id == half:
                            warped_radii.append(ori_radii)
                            warped_visibility_filter.append(ori_visibility_filter)
                            warped_viewspace_point_tensor.append(viewspace_point_tensor)

                        warped_cam_R_np = warped_cams[latent_sharp_id].R.detach().cpu().numpy()
                        warped_cam_T_np = warped_cams[latent_sharp_id].T.detach().cpu().numpy()
                        warped_cam_w2c = torch.tensor(getWorld2View2(warped_cam_R_np, warped_cam_T_np)).to(warped_cams[latent_sharp_id].R.device)
                        warped_cam_c2w = torch.inverse(warped_cam_w2c)
                        warped_translations_batch.append(warped_cam_c2w[:3, 3])
                    
                    
                    radii_list.append(torch.stack(warped_radii, dim=0))
                    visibility_filter_list.append(torch.stack(warped_visibility_filter, dim=0))
                    viewspace_point_tensor_list.append(warped_viewspace_point_tensor)
                     
                    warped_rotations_batch = torch.stack(warped_rotations_batch, dim=0)
                    warped_translations_batch = torch.stack(warped_translations_batch, dim=0)

                    rendered_images = torch.stack(rendered_images, dim=0)
                    pred_image = torch.mean(rendered_images, dim=0) + 1e-10
                    
                    pred_depth = depth_ori

                    warped_rotations.append(warped_rotations_batch.unsqueeze(0))
                    warped_translations.append(warped_translations_batch.unsqueeze(0))
                    
                    view_cams_R.append(torch.tensor(viewpoint_cam.R).to(pred_image.device).unsqueeze(0))
                    view_cams_R_fwd.append(torch.tensor(fwd_viewpoint_cams[n_batch].R).to(pred_image.device).unsqueeze(0))
                    view_cams_R_bwd.append(torch.tensor(bwd_viewpoint_cams[n_batch].R).to(pred_image.device).unsqueeze(0))
                    
                    view_cam_w2c = torch.tensor(getWorld2View2(viewpoint_cam.R, viewpoint_cam.T)).to(pred_image.device)
                    view_cam_c2w = torch.inverse(view_cam_w2c)
                    view_cams_t.append(view_cam_c2w[:3, 3].unsqueeze(0))

                    view_cam_fwd_w2c = torch.tensor(getWorld2View2(fwd_viewpoint_cams[n_batch].R, fwd_viewpoint_cams[n_batch].T)).to(pred_image.device)
                    view_cam_fwd_c2w = torch.inverse(view_cam_fwd_w2c)
                    view_cams_t_fwd.append(view_cam_fwd_c2w[:3, 3].unsqueeze(0))

                    view_cam_bwd_w2c = torch.tensor(getWorld2View2(bwd_viewpoint_cams[n_batch].R, bwd_viewpoint_cams[n_batch].T)).to(pred_image.device)
                    view_cam_bwd_c2w = torch.inverse(view_cam_bwd_w2c)
                    view_cams_t_bwd.append(view_cam_bwd_c2w[:3, 3].unsqueeze(0))        

                curr_exp2mid_coord = []
                curr_mid2exp_coord = []
                curr_latent_img_list = []
                curr_latent_alpha_list = []
                exposure_length = len(warped_cams)
                exposure_max_delta = 1.0
                for latent_sharp_id in range(exposure_length):
                    
                    exposure_ratio = (latent_sharp_id - half) / half
                    delta_exposure = exposure_max_delta * exposure_ratio
                        
                    exp2mid_coord_map, mid2exp_coord_map, latent_img, latent_alpha = get_flow(viewpoint_cam, stat_gaussians, dyn_gaussians, pipe, background, delta_exposure=delta_exposure)
                    curr_latent_img_list.append(latent_img.unsqueeze(0))
                    curr_exp2mid_coord.append(exp2mid_coord_map)
                    curr_mid2exp_coord.append(mid2exp_coord_map)
                    curr_latent_alpha_list.append(latent_alpha.unsqueeze(0))
                    
                latent_img_final_list.append(torch.cat(curr_latent_img_list, dim=0).unsqueeze(0))
                latent_alpha_final_list.append(torch.cat(curr_latent_alpha_list, dim=0).unsqueeze(0))
                exp2mid_coord_final_list.append(torch.cat(curr_exp2mid_coord, dim=0).unsqueeze(0))
                mid2exp_coord_final_list.append(torch.cat(curr_mid2exp_coord, dim=0).unsqueeze(0))

                images.append(pred_image.unsqueeze(0))
                d_images.append(pred_image.unsqueeze(0))
                depth_list.append(pred_depth.unsqueeze(0))

                pred_normal = get_normals(pred_depth + 1e-6, camera_metadata)
                pred_normals.append(pred_normal)
                motion_masks.append(viewpoint_cam.mask.unsqueeze(0))
                
                if torch.isnan(pred_normal).any():
                    print("NaN found in pred normal!")

        loss = 0

        if stage != "warm":

            radii = torch.stack(radii_list, dim=0)
            visibility_filter = torch.stack(visibility_filter_list, dim=0)
            
            s_image_tensor = torch.cat(s_images, 0)
            image_tensor = torch.cat(images, 0)
            depth_tensor = torch.cat(depth_list, 0)
            ori_image_tensor = torch.cat(image_ori_list, 0)
            
            latent_img_final_tensor = torch.cat(latent_img_final_list, dim=0)
            latent_alpha_final_tensor = torch.cat(latent_alpha_final_list, dim=0)
            exp2mid_coord_final_tensor = torch.cat(exp2mid_coord_final_list, dim=0)
            mid2exp_coord_final_tensor = torch.cat(mid2exp_coord_final_list, dim=0)
            
            d_image_tensor = torch.cat(d_images, 0)
            normal_tensor = torch.cat(pred_normals, 0)
            motion_mask_tensor = torch.cat(motion_masks, 0)
            d_alpha_tensor = torch.cat(d_alphas, 0)
            s_alpha_tensor = torch.cat(s_alphas, 0)

            # Main losses (L1 and SSIM) for GS densification
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            psnr_ = psnr(image_tensor, gt_image_tensor).detach().mean().double()

            ssim_loss = 0
            if opt.lambda_dssim != 0:
                ssim_loss = ssim(image_tensor, gt_image_tensor)

            photo_loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)
            photo_loss.backward(retain_graph=True)
            reg_loss = 0


            # Split static and dynamic gradients (we know their indices because of cat in render)
            stat_viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor.squeeze(0)[0:no_stat_gs])
            stat_radii = radii[..., :no_stat_gs].flatten(0,1).max(dim=0).values
            stat_visibility_filter = visibility_filter[..., :no_stat_gs].flatten(0,1).any(dim=0)
            for grad_idx in range(0, len(viewspace_point_tensor_list)):
                stat_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx][0].grad.squeeze(0)[0:no_stat_gs]
            stat_viewspace_point_tensor_grad[:, 0] *= W * 0.5
            stat_viewspace_point_tensor_grad[:, 1] *= H * 0.5

            dyn_viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor.squeeze(0)[no_stat_gs:no_stat_gs+no_dyn_gs])
            dyn_radii = radii[..., no_stat_gs:no_stat_gs+no_dyn_gs].flatten(0,1).max(dim=0).values
            dyn_visibility_filter = visibility_filter[..., no_stat_gs:no_stat_gs+no_dyn_gs].flatten(0,1).any(dim=0)
            for grad_idx in range(0, len(viewspace_point_tensor_list)):
                dyn_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx][0].grad.squeeze(0)[no_stat_gs:no_stat_gs + no_dyn_gs]
            dyn_viewspace_point_tensor_grad[:, 0] *= W * 0.5
            dyn_viewspace_point_tensor_grad[:, 1] *= H * 0.5

                        
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor)
            reg_loss += 0.2 * depth_loss

            mask_loss = 1e-7 * entropy_loss(d_alpha_tensor) + 1e-7 * sparsity_loss(d_alpha_tensor)
            reg_loss += mask_loss


            # exp2mid warping
            if iteration > blceopt.start_warp:
                norm_exp2mid_coord_final_tensor = exp2mid_coord_final_tensor
                norm_exp2mid_coord_final_tensor[..., 0] = norm_exp2mid_coord_final_tensor[..., 0] / (W - 1)
                norm_exp2mid_coord_final_tensor[..., 1] = norm_exp2mid_coord_final_tensor[..., 1] / (H - 1)
                norm_exp2mid_coord_final_tensor = 2.0 * norm_exp2mid_coord_final_tensor - 1.0
                norm_exp2mid_coord_final_tensor = norm_exp2mid_coord_final_tensor.flatten(0, 1)
                warped_exp2mid_img_tensor = F.grid_sample(ori_image_tensor.unsqueeze(1).expand(-1, exposure_length, -1, -1, -1).flatten(0,1), norm_exp2mid_coord_final_tensor, mode='bilinear', padding_mode='border').reshape(-1, exposure_length, 3, H, W)
                
                # mid2exp warping
                norm_mid2exp_coord_final_tensor = mid2exp_coord_final_tensor
                norm_mid2exp_coord_final_tensor[..., 0] = norm_mid2exp_coord_final_tensor[..., 0] / (W - 1)
                norm_mid2exp_coord_final_tensor[..., 1] = norm_mid2exp_coord_final_tensor[..., 1] / (H - 1)
                norm_mid2exp_coord_final_tensor = 2.0 * norm_mid2exp_coord_final_tensor - 1.0
                norm_mid2exp_coord_final_tensor = norm_mid2exp_coord_final_tensor.flatten(0, 1)
                warped_mid2exp_img_tensor = F.grid_sample(latent_img_final_tensor.flatten(0,1), norm_mid2exp_coord_final_tensor, mode='bilinear', padding_mode='border').reshape(-1, exposure_length, 3, H, W)
                
                flow_loss = opt.lambda_flow_loss * (l1_loss(warped_exp2mid_img_tensor.flatten(0,1), latent_img_final_tensor.flatten(0,1), mask=latent_alpha_final_tensor.flatten(0,1)) + l1_loss(warped_mid2exp_img_tensor.flatten(0,1), ori_image_tensor.unsqueeze(1).expand(-1, exposure_length, -1, -1, -1).flatten(0,1), mask=d_alpha_tensor.unsqueeze(1).expand(-1, exposure_length, -1, -1, -1).flatten(0,1)))
                reg_loss += flow_loss

            loss += reg_loss

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, ending program now.")
            exit()
        iter_end.record()

        # Debug intermediate results
        if dataset.debug_process and (iteration == 1 or iteration % 300 == 0):
            b_id = 0
            debug_path = os.path.join(scene.model_path, f"{stage}_debug")
            if not os.path.exists(debug_path):
                os.makedirs(debug_path)

            plot_path = os.path.join(scene.model_path, f"{stage}_gs_plot")
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            norm_fact = torch.max(depth_tensor.detach())

            if stage != "warm":
                debug_dict['image'] = image_tensor.detach()
                debug_dict['depth_gs'] = depth_tensor.detach() / norm_fact
                debug_dict['d_alpha'] = d_alpha_tensor.detach()
                debug_dict['s_alpha'] = s_alpha_tensor.detach()

            if stage == "fine":
                debug_dict['image_s'] = s_image_tensor.detach()

            debug_dict['gt_image'] = gt_image_tensor.detach()
            debug_dict['gt_depth'] = gt_depth_tensor.detach()
            
            save_debug_imgs(debug_dict, b_id, epoch=iteration, deb_path=debug_path)

        with torch.no_grad():
            # Progress bar   
            if stage != "warm":                
                ema_loss_for_log_photo = 0.4 * photo_loss.detach().item() + 0.6 * ema_loss_for_log_photo
                ema_loss_for_log_reg = 0.4 * reg_loss.detach().item() + 0.6 * ema_loss_for_log_reg
                ema_loss_for_log_mask = 0.4 * mask_loss.detach().item() + 0.6 * ema_loss_for_log_mask
                ema_psnr_for_log = 0.4 * psnr_.detach() + 0.6 * ema_psnr_for_log
                
                if iteration > blceopt.start_warp:
                    ema_loss_for_log_flow = 0.4 * flow_loss.detach().item() + 0.6 * ema_loss_for_log_flow
            else:
                ema_psnr_for_log = 0
            if stage != "warm":
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"photo loss": f"{ema_loss_for_log_photo:.{6}f}",
                                            "reg loss": f"{ema_loss_for_log_reg:.{6}f}",
                                            "mask loss": f"{ema_loss_for_log_mask:.{6}f}",
                                            "flow loss": f"{ema_loss_for_log_flow:.{6}f}",
                                            "psnr": f"{ema_psnr_for_log:.{2}f}",
                                            "Pts (static, dynamic)": f"{no_stat_gs}, {no_dyn_gs}",
                                            "Focal": f"{viewpoint_stack[0].focal}",
                                            "MinCtrl": f"{dyn_gaussians.current_control_num.min().item()}"})
                    progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

        # Log and save
        timer.pause()
        with torch.no_grad():
            if iteration in testing_iterations and stage != "warm":
                if iteration > blceopt.start_warp:
                    print("Exposure time: ",  blcekernel.model.exposure_time_expo)
                aligned_my_test_cams = []
                for idx, viewpoint in enumerate(viewpoint_stack):
                    if idx == 0:
                        warped_cams, _ = blcekernel.get_warped_cams(viewpoint, viewpoint_stack[idx+1], viewpoint)
                    elif idx == len(viewpoint_stack) - 1:
                        warped_cams, _ = blcekernel.get_warped_cams(viewpoint, viewpoint, viewpoint_stack[idx-1])
                    else:
                        warped_cams, _ = blcekernel.get_warped_cams(viewpoint, viewpoint_stack[idx+1], viewpoint_stack[idx-1])
                        
                    middle_cam = warped_cams[len(warped_cams) // 2]
                    
                    input_train_pose = viewpoint_stack[idx].world_view_transform
                    input_test_pose = my_test_cams[idx].world_view_transform
                    output_train_pose = middle_cam.world_view_transform
                    output_test_pose = input_test_pose @ torch.inverse(input_train_pose) @ output_train_pose
                    
                    aligned_test_cam = copy.deepcopy(my_test_cams[idx])
                    aligned_test_cam.world_view_transform = output_test_pose
                    aligned_my_test_cams.append(aligned_test_cam)
                
                test_psnr, cur_iter = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, aligned_my_test_cams, render, [pipe, background], stage,
                                scene.dataset_type, final_iter)
                
                if test_psnr > BEST_PSNR:
                    BEST_PSNR = test_psnr
                    BEST_ITER = cur_iter
                    scene.save_best_psnr(iteration, stage, blcekernel)


            if dataset.render_process:
                if iteration in testing_iterations:
                    if stage != "warm":
                        render_training_image(scene, stat_gaussians, dyn_gaussians, my_test_cams, render, pipe,
                                              background, stage, iteration, timer.get_elapsed_time(),
                                              scene.dataset_type)
                        
                    if iteration > blceopt.start_warp:
                        render_training_image(scene, stat_gaussians, dyn_gaussians, viewpoint_stack, render, pipe,
                                              background, stage, iteration, timer.get_elapsed_time(),
                                              scene.dataset_type, blcekernel=blcekernel, is_train=True)
                    else:
                        render_training_image(scene, stat_gaussians, dyn_gaussians, viewpoint_stack, render, pipe,
                                            background, stage, iteration, timer.get_elapsed_time(), 
                                            scene.dataset_type, is_train=True)
                    
            timer.start()

        # Optimizer step
        if iteration < opt.iterations:
            stat_gaussians.optimizer.step()
            stat_gaussians.optimizer.zero_grad(set_to_none=True)

            dyn_gaussians.optimizer.step()
            dyn_gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration > blceopt.start_warp:
                # for param in blcekernel.model.parameters():
                #     if param.requires_grad:
                blcekernel.optimizer.step()
                blcekernel.optimizer.zero_grad()
                blcekernel.adjust_lr()

        # Densification        
        if stage != "warm":
            with torch.no_grad():
                if iteration < opt.densify_until_iter :
                    dyn_gaussians.max_radii2D[dyn_visibility_filter] = torch.max(dyn_gaussians.max_radii2D[dyn_visibility_filter], dyn_radii[dyn_visibility_filter])
                    dyn_gaussians.add_densification_stats(dyn_viewspace_point_tensor_grad, dyn_visibility_filter)

                    stat_gaussians.max_radii2D[stat_visibility_filter] = torch.max(stat_gaussians.max_radii2D[stat_visibility_filter], stat_radii[stat_visibility_filter])
                    stat_gaussians.add_densification_stats(stat_viewspace_point_tensor_grad, stat_visibility_filter)
                    
                    flag_d = controlgaussians(opt, dyn_gaussians, densify, iteration, scene, dyn_visibility_filter, dyn_radii, dyn_viewspace_point_tensor_grad, flag_d, traincamerawithdistance=None, maxbounds=maxbounds_d, minbounds=minbounds_d, is_dynamic=True)
                    flag_s = controlgaussians(opt, stat_gaussians, densify, iteration, scene, stat_visibility_filter, stat_radii, stat_viewspace_point_tensor_grad, flag_s, traincamerawithdistance=None, maxbounds=maxbounds_s, minbounds=minbounds_s)

    scene.save(iteration, stage, blcekernel)
    return BEST_PSNR, BEST_ITER


def training(dataset, hyper, opt, pipe, blceopt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname, check_seed=False):
    tb_writer = prepare_output_and_logger(expname)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)
    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)  # large CPU usage
    timer.start()

    stat_pc, dyn_pc, dyn_tracjectory = scene_initialization(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                           checkpoint_iterations, checkpoint, debug_from,
                                           dyn_gaussians, stat_gaussians, scene, "warm", tb_writer, opt.coarse_iterations, timer)

    stat_gaussians.create_from_pcd(pcd=stat_pc, spatial_lr_scale=5, time_line=0)
    dyn_gaussians.create_from_pcd_dynamic(pcd=dyn_pc, spatial_lr_scale=5, time_line=0, dyn_tracjectory=dyn_tracjectory)
    xyz_max = stat_pc.points.max(axis=0)
    xyz_min = stat_pc.points.min(axis=0)
    dyn_gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min, ref_type=dyn_gaussians.get_xyz)

    
    BEST_PSNR, BEST_ITER = scene_reconstruction(dataset, opt, hyper, pipe, blceopt, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            dyn_gaussians, stat_gaussians, scene, "fine", tb_writer, opt.iterations, timer, drop=True, check_seed=check_seed)


    return BEST_PSNR, BEST_ITER


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, test_cams,
                    renderFunc, renderArgs, stage, dataset_type, final_iter):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    test_psnr = 0.0
    cur_iter = 0.0

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': test_cams},
                              {'name': 'train', 'cameras': []})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.stat_gaussians, scene.dyn_gaussians, stage=stage,
                                            cam_type=dataset_type, iter_fact=iteration, *renderArgs)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/render".
                                                 format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/ground_truth".
                                                     format(viewpoint.image_name), gt_image[None],
                                                     global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if config['name'] == 'test':
                test_psnr = psnr_test
                cur_iter = iteration
        # if tb_writer:
        #     tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        #     tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        #     tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
        #     tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)

        torch.cuda.empty_cache()
        return test_psnr, cur_iter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    cp = blceParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--check_seed", action="store_true")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100 * i for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 15000, 20000, 25000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-render_process", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # with torch.autograd.profiler.profile(use_cuda=False) as prof:
    #     BEST_PSNR, BEST_ITER = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
    #             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # #print(prof.key_averages().table(sort_by="cpu_time_total"))
    # prof.export_stacks("results.prof", "cpu")  # 프로파일링 데이터를 파일로 저장
    # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # num_threads = torch.get_num_threads()
    # print(f"Current number of threads: {num_threads}")

    torch.set_num_threads(16)

    BEST_PSNR, BEST_ITER = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), cp.extract(args), args.test_iterations,
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.check_seed)

    if args.check_seed:
        with open(os.path.join(args.model_path, "seed.txt"), 'a') as f:
            f.write("BEST PSNR : " + str(BEST_PSNR) + " SEED : " + str(args.seed) + "\n")

    # All done
    print("\nTraining complete.")
    print("BEST PSNR : ", BEST_PSNR)
    print("BEST ITER : ", BEST_ITER)
