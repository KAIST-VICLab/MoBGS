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

import torch

from scene.gaussian_model import GaussianModel
from gsplat.rendering import rasterization, fully_fused_projection

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

# @torch.jit.script
def interpolate_cubic_hermite(signal, times, N):
    # start.record()
    times_scaled = times * (N - 1)[:,None]
    indices = torch.floor(times_scaled).long()
    # Clamping to avoid out-of-bounds indices

    indices = torch.clamp(indices, torch.zeros_like(N)[:,None].expand(-1, 3, -1), (N - 2)[:,None].expand(-1, 3, -1)).long()
    left_indices = torch.clamp(indices - 1, torch.zeros_like(N)[:,None].expand(-1, 3, -1), (N - 1)[:,None].expand(-1, 3, -1)).long()
    right_indices = torch.clamp(indices + 1, torch.zeros_like(N)[:,None].expand(-1, 3, -1), (N - 1)[:,None].expand(-1, 3, -1)).long()
    right_right_indices = torch.clamp(indices + 2, torch.zeros_like(N)[:,None].expand(-1, 3, -1), (N - 1)[:,None].expand(-1, 3, -1)).long()

    t = (times_scaled - indices.float())
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # if len(signal.shape) == 3:  # remove extra singleton dimension
    interpolation = interpolation.squeeze(-1)
    # end.record()
    # torch.cuda.synchronize()
    # print('v1:', start.elapsed_time(end))
    return interpolation

#gsplat
def render(viewpoint_camera, stat_pc : GaussianModel, dyn_pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color=None, stage="fine", cam_type=None, is_static=False,
           over_t=None, over_vde=None, get_static=False, get_dynamic=False, stat_stat=True, ref_wc=None,
           iter_fact=1, flow=None, coherent=None, target_ts = None, target_w2cs=None, get_heatmap=False, w2c=None, delta_exposure=None, get_flow=False, cluster=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling

    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = torch.ones((dyn_pc.get_xyz.shape[0],1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    if w2c is not None:
        viewmat = w2c
    K = viewpoint_camera.K
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color,bg_color,bg_color], dim=-1)
        
    trbfcenter = dyn_pc.get_trbfcenter
    if delta_exposure is not None:
        trbfdistanceoffset = (viewpoint_camera.time + delta_exposure / viewpoint_camera.max_time) * pointtimes - trbfcenter
        ori_trbfdistanceoffset = viewpoint_camera.time * pointtimes - trbfcenter
        ori_tforpoly = ori_trbfdistanceoffset.detach()
        ori_rotations = dyn_pc.get_rotation_dy(rotations, ori_tforpoly)
    else:
        trbfdistanceoffset = viewpoint_camera.time * pointtimes - trbfcenter

    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()   
    if delta_exposure is not None:
        curr_time = torch.tensor(viewpoint_camera.time).cuda() + delta_exposure / viewpoint_camera.max_time
        curr_time = torch.clamp(curr_time, 0, 1)
        
        ori_time = torch.tensor(viewpoint_camera.time).cuda()
        ori_deform_means3D = interpolate_cubic_hermite(control_xyz.permute(0,2,1), ori_time[None, None].expand(control_xyz.shape[0], 3, 1), N=dyn_pc.current_control_num)
        ori_means3D = ori_deform_means3D * 1e-2
    else:
        curr_time = torch.tensor(viewpoint_camera.time).cuda() 
    deform_means3D = interpolate_cubic_hermite(control_xyz.permute(0,2,1), curr_time[None, None].expand(control_xyz.shape[0], 3, 1), N=dyn_pc.current_control_num)
    means3D = deform_means3D * 1e-2

    if coherent is not None:
        means3D = means3D + coherent

    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    rotations = dyn_pc.rotation_activation(rotations)

    colors_precomp = dyn_pc.get_features(tforpoly)

    cov3D_precomp = None

    smeans3D_final, sscales_final, srotations_final, sopacity_final = stat_means3D, stat_scales, stat_rotations, stat_opacity
    means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity

    d_alpha = None
    if get_dynamic:
        d_ss_points = torch.zeros(no_dyn_gs, 3, dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
        d_means2D = d_ss_points

        dmeans3D_final = means3D_final
        dscales_final = scales_final
        drotations_final = rotations_final
        dopacity_final = opacity_final
        dcolors_precomp = colors_precomp

        d_img, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=dcolors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )

        d_depth = d_img[...,-1]
        d_img = d_img[...,:-1].permute(0,3,1,2)
        d_image = dyn_pc.rgbdecoder(d_img, viewpoint_camera.cam_ray)
        d_image = d_image.squeeze(0)

        d_alpha, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=torch.ones(dcolors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        d_alpha = d_alpha[...,0]


    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)
    
    if delta_exposure is not None and get_flow:
        ori_means3D_final = torch.cat((smeans3D_final, ori_means3D), 0)
        ori_rotations_final = torch.cat((srotations_final, ori_rotations), 0)
        _, ori_means2D_final, _, _, _ = fully_fused_projection(
            means = ori_means3D_final,
            covars=None,
            quats=ori_rotations_final,
            scales=scales_final,
            viewmats = viewmat[None],
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
        ) # B, N, 2
    
    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
    )

    depth = rendered_image[...,-1]
    rendered_image = rendered_image[...,:-1].permute(0,3,1,2)
    radii = info["radii"].squeeze(0) 

    try:
        info["means2d"].retain_grad()
    except:
        pass
    
    # rendered_image = torch.cat((rendered_image1, rendered_image2, rendered_image3), dim=0)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray)
    rendered_image = rendered_image.squeeze(0)    

    # Get scene flow
    wc_list = None

    s_rendered_image = None
    s_depth = None
    s_alpha = None
    if get_static:
        s_rendered_image, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_colors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = rendered_image[...,-1]
        s_rendered_image = s_rendered_image[...,:-1].permute(0,3,1,2)
        s_rendered_image = dyn_pc.rgbdecoder(s_rendered_image, viewpoint_camera.cam_ray)
        s_rendered_image = s_rendered_image.squeeze(0)

        s_alpha, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        s_alpha = s_alpha[...,0]
        
    # generate exp optical flow
    if delta_exposure is not None and get_flow:
        flow_2d = (ori_means2D_final - info["means2d"].clone().detach()).squeeze(0)
        rendered_flow, _, _ = rasterization(
            means=means3D_final,
            quats=rotations_final,
            scales=scales_final,
            opacities=opacity_final.squeeze(-1),
            colors=flow_2d,
            backgrounds=None,
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        # rendered_flow = torch.trunc(rendered_flow)
        ori_coord_map = torch.tensor(viewpoint_camera.get_pixels(int(viewpoint_camera.image_width), int(viewpoint_camera.image_height), use_center=False)).type_as(rendered_flow) + rendered_flow


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "s_render": s_rendered_image,
            "s_depth": s_depth,
            "d_render": d_image if get_dynamic else None,
            "d_depth": d_depth if get_dynamic else None,
            "d_alpha": d_alpha if get_dynamic else None,
            "d_means3d": dmeans3D_final if get_dynamic else None,
            "s_alpha": s_alpha,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "blending_factor": None,
            "world_coordinates": wc_list,
            "splat_center": None,
            "means_3d_final": means3D_final * 1e2,
            "colors_precomp_final": colors_precomp_final,
            "ori_flow": rendered_flow if delta_exposure is not None and get_flow else None,
            "ori_coord_map": ori_coord_map if delta_exposure is not None and get_flow else None,
            "means_3d": means3D,
            "labels": labels if cluster is not None else None,  
            "centroids": centroids if cluster is not None else None
            }

def get_flow(viewpoint_camera, stat_pc : GaussianModel, dyn_pc : GaussianModel, pipe, bg_color : torch.Tensor, delta_exposure=None):
    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = torch.ones((dyn_pc.get_xyz.shape[0],1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    K = viewpoint_camera.K
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color,bg_color,bg_color], dim=-1)
    trbfcenter = dyn_pc.get_trbfcenter
    
        
    exp_trbfdistanceoffset = (viewpoint_camera.time + delta_exposure / viewpoint_camera.max_time) * pointtimes - trbfcenter
    mid_trbfdistanceoffset = viewpoint_camera.time * pointtimes - trbfcenter
    
    
    mid_tforpoly = mid_trbfdistanceoffset.detach()
    exp_tforpoly = exp_trbfdistanceoffset.detach()
    
    mid_rotations = dyn_pc.get_rotation_dy(rotations, mid_tforpoly)
    exp_rotations = dyn_pc.get_rotation_dy(rotations, exp_tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()   
    mid_time = torch.tensor(viewpoint_camera.time).cuda()
    mid_time = torch.clamp(mid_time, 0, 1)
    exp_time = torch.tensor(viewpoint_camera.time).cuda() + delta_exposure / viewpoint_camera.max_time
    exp_time = torch.clamp(exp_time, 0, 1)
    

    mid_deform_means3D = interpolate_cubic_hermite(control_xyz.permute(0,2,1), mid_time[None, None].expand(control_xyz.shape[0], 3, 1), N=dyn_pc.current_control_num)
    mid_means3D = mid_deform_means3D * 1e-2
    exp_deform_means3D = interpolate_cubic_hermite(control_xyz.permute(0,2,1), exp_time[None, None].expand(control_xyz.shape[0], 3, 1), N=dyn_pc.current_control_num)
    exp_means3D = exp_deform_means3D * 1e-2

    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    
    mid_rotations = dyn_pc.rotation_activation(mid_rotations)
    exp_rotations = dyn_pc.rotation_activation(exp_rotations)
    
    mid_colors_precomp = dyn_pc.get_features(mid_tforpoly)
    exp_colors_precomp = dyn_pc.get_features(exp_tforpoly)
    
    smeans3D_final, sscales_final, srotations_final, sopacity_final = stat_means3D, stat_scales, stat_rotations, stat_opacity
    
    mid_means3D_final, scales_final, mid_rotations_final, opacity_final = mid_means3D, scales, mid_rotations, opacity
    exp_means3D_final, exp_rotations_final = exp_means3D, exp_rotations


    latent_alpha, _, _ = rasterization(
        means=exp_means3D_final,
        quats=exp_rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=torch.ones(exp_colors_precomp.shape[0], 1).cuda(),
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    latent_alpha = latent_alpha[...,0]

    # Combine stat and dyn gaussians
    mid_means3D_final = torch.cat((smeans3D_final, mid_means3D_final), 0)
    mid_rotations_final = torch.cat((srotations_final, mid_rotations_final), 0)
    
    exp_means3D_final = torch.cat((smeans3D_final, exp_means3D_final), 0)
    exp_rotations_final = torch.cat((srotations_final, exp_rotations_final), 0)
    
    scales_final = torch.cat((sscales_final, scales_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    
    exp_colors_precomp_final = torch.cat((stat_colors_precomp, exp_colors_precomp), 0)




    # project to 2D
    _, mid_means2D_final, _, _, _ = fully_fused_projection(
        means = mid_means3D_final,
        covars=None,
        quats=mid_rotations_final,
        scales=scales_final,
        viewmats = viewmat[None],
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
    ) # B, N, 2
    
    _, exp_means2D_final, _, _, _ = fully_fused_projection(
        means = exp_means3D_final,
        covars=None,
        quats=exp_rotations_final,
        scales=scales_final,
        viewmats = viewmat[None],
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
    ) # B, N, 2
    
    
        
    # generate exp to mid optical flow
    exp2mid_flow_2d = (mid_means2D_final - exp_means2D_final).squeeze(0)
    exp2mid_rendered_flow, _, _ = rasterization(
        means=exp_means3D_final,
        quats=exp_rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=exp2mid_flow_2d,
        backgrounds=None,
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    # rendered_flow = torch.trunc(rendered_flow)
    exp2mid_coord_map = torch.tensor(viewpoint_camera.get_pixels(int(viewpoint_camera.image_width), int(viewpoint_camera.image_height), use_center=False)).type_as(exp2mid_rendered_flow) + exp2mid_rendered_flow

    # generate mid to exp optical flow
    mid2exp_flow_2d = - exp2mid_flow_2d
    mid2exp_rendered_flow, _, _ = rasterization(
        means=mid_means3D_final,
        quats=mid_rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=mid2exp_flow_2d,
        backgrounds=None,
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    # rendered_flow = torch.trunc(rendered_flow)
    mid2exp_coord_map = torch.tensor(viewpoint_camera.get_pixels(int(viewpoint_camera.image_width), int(viewpoint_camera.image_height), use_center=False)).type_as(mid2exp_rendered_flow) + mid2exp_rendered_flow
    
    latent_img, _, _ = rasterization(
        means=exp_means3D_final,
        quats=exp_rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=exp_colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
    )

    latent_img = latent_img[...,:-1].permute(0,3,1,2)
    latent_img = dyn_pc.rgbdecoder(latent_img, viewpoint_camera.cam_ray)
    latent_img = latent_img.squeeze(0)    
    
    return exp2mid_coord_map, mid2exp_coord_map, latent_img, latent_alpha   

def get_flow_static(source_camera, target_camera, splat_camera, stat_pc : GaussianModel, dyn_pc : GaussianModel, pipe, bg_color : torch.Tensor):

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    source_viewmat = source_camera.world_view_transform.transpose(0, 1)
    target_viewmat = target_camera.world_view_transform.transpose(0, 1)
    viewmat = splat_camera.world_view_transform.transpose(0, 1)
    K = source_camera.K
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color,bg_color,bg_color], dim=-1)
    smeans3D_final, sscales_final, srotations_final, sopacity_final = stat_means3D, stat_scales, stat_rotations, stat_opacity

    # project to 2D 
    _, source_means2D_final, _, _, _ = fully_fused_projection(
        means = smeans3D_final,
        covars=None,
        quats=srotations_final,
        scales=sscales_final,
        viewmats = source_viewmat[None],
        Ks=K[None],  # [C, 3, 3]
        width=int(source_camera.image_width),
        height=int(source_camera.image_height),
    ) # B, N, 2
    
    _, target_means2D_final, _, _, _ = fully_fused_projection(
        means = smeans3D_final,
        covars=None,
        quats=srotations_final,
        scales=sscales_final,
        viewmats = target_viewmat[None],
        Ks=K[None],  # [C, 3, 3]
        width=int(target_camera.image_width),
        height=int(target_camera.image_height),
    ) # B, N, 2    
    
    # generate exp to mid optical flow
    flow_2d = (source_means2D_final - target_means2D_final).squeeze(0)
    
    rendered_flow, _, _ = rasterization(
        means=smeans3D_final,
        quats=srotations_final,
        scales=sscales_final,
        opacities=sopacity_final.squeeze(-1),
        colors=flow_2d,
        backgrounds=None,
        viewmats=viewmat[None],  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(splat_camera.image_width),
        height=int(splat_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    
    return flow_2d, rendered_flow