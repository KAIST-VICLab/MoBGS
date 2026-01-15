import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np
from scene import deformation
import torch.nn.functional as F

import copy
import flow_vis

@torch.no_grad()
def render_training_image(scene, stat_gaussians, dyn_gaussians, viewpoints, render_func, pipe, background, stage,
                          iteration, time_now, dataset_type, is_train=False, over_t=None, blcekernel=None):
    # Get base parameters for warping
    if stage == "warm":
        viewpoint = viewpoints[0]
        gt_normal = viewpoint.normal[None].cuda()
        normal_in = gt_normal.reshape(-1, 3)
        pixels = viewpoint.metadata.get_pixels(normalize=True)
        pixels = torch.from_numpy(pixels).cuda()
        pixels_in = pixels.reshape(-1, 2)
        gt_depth = viewpoint.depth[None].cuda()
        # gt_depth = gt_depth / F.avg_pool2d(gt_depth.detach(), kernel_size=gt_depth.shape[2::])
        depth_in = gt_depth.reshape(-1, 1)
        time_in_0 = torch.tensor(0).float().cuda()
        time_in_0 = time_in_0.view(1, 1)
        # pred_R0, pred_T0, depth_scale_T0, depth_shift_T0 = dyn_gaussians._posenet(time_in_0, depth=depth_in, normals=normal_in, pixel=pixels_in)
        # pred_R0, pred_T0, CVD0 = dyn_gaussians._posenet(time_in_0, depth=depth_in, normals=normal_in, pixel=pixels_in)


    def render(static_gaussians, dynamic_gaussians, viewpoint, fwd_viewpoint, bwd_viewpoint, path, scaling, cam_type, over_t=None):
        if stage == "warm":
            # Get warped images (all warped to time step 0)
            if dataset_type == "PanopticSports":
                image_tensor = viewpoint['image'][None].cuda()
            else:
                image_tensor = viewpoint.original_image[None].cuda()
            B, C, H, W = image_tensor.shape

            normal_in = viewpoint.normal[None].cuda().reshape(-1, 3)
            pixels = viewpoint.metadata.get_pixels(normalize=True)
            pixels_in = torch.from_numpy(pixels).cuda().reshape(-1, 2)
            gt_depth = viewpoint.depth[None].cuda()
            # gt_depth = gt_depth / F.avg_pool2d(gt_depth.detach(), kernel_size=gt_depth.shape[2::])
            depth_in = gt_depth.reshape(-1, 1)
            time_in = torch.tensor(viewpoint.time).float().cuda()
            time_in = time_in.view(1, 1)
            # CVD = depth_in.view(1, 1, H, W)
            
            # CVD = CVD_T0.view(1, 1, H, W)
            # pred_R, pred_T, _, _ = dynamic_gaussians._posenet(time_in, depth=depth_in, normals=normal_in, pixel=pixels_in)
            # pred_R, pred_T, CVD = dynamic_gaussians._posenet(time_in, depth=depth_in, normals=normal_in, pixel=pixels_in)

            # depth_in = depth_in.view(1, 1, H, W)
            # depth_shift = depth_shift_T0.view(1, 1, H, W)
            # depth_scale = depth_scale_T0.view(1, 1, 1, 1)
            # CVD = depth_scale * depth_in + depth_shift

            K_tensor = torch.zeros(1, 3, 3).type_as(image_tensor)
            K_tensor[:, 0, 0] = float(viewpoint.metadata.scale_factor_x)
            K_tensor[:, 1, 1] = float(viewpoint.metadata.scale_factor_y)
            K_tensor[:, 0, 2] = float(viewpoint.metadata.principal_point_x)
            K_tensor[:, 1, 2] = float(viewpoint.metadata.principal_point_y)
            K_tensor[:, 2, 2] = float(1)

            w2c_target = torch.cat((pred_R0, pred_T0[:, :, None]), -1)
            w2c_prev = torch.cat((pred_R, pred_T[:, :, None]), -1)
            warped_img = deformation.inverse_warp_rt1_rt2(image_tensor, CVD, w2c_target, w2c_prev,
                                                           K_tensor, torch.inverse(K_tensor))

            p_im = warped_img.detach().squeeze().cpu().numpy()
            im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
            im.save(path.replace('.jpg', '_warped.jpg'))
            return

        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, static_gaussians, dynamic_gaussians, pipe, background, stage=stage,
                                 cam_type=cam_type, get_static=True, get_dynamic=True, over_t=viewpoint.time)     
        
        if is_train and blcekernel is not None:
            exposure_length = 9
            latent_images = []
            half = exposure_length // 2
            warped_cams, exposure_time = blcekernel.get_warped_cams(viewpoint, fwd_viewpoint, bwd_viewpoint)
            for latent_sharp_id, cam in enumerate(warped_cams):     
                
                delta_exposure = exposure_time[latent_sharp_id]
                
                latent_render_pkg = render_func(cam, static_gaussians, dynamic_gaussians, pipe, background, stage="fine",
                                    cam_type=cam_type, get_static=True, get_dynamic=True, delta_exposure=delta_exposure,  get_flow=True)
                flow_color = flow_vis.flow_to_color(latent_render_pkg["ori_flow"].squeeze(0).detach().cpu().numpy(), convert_to_bgr=False)
                flow_color_save = Image.fromarray(flow_color)
                flow_color_save.save(path.replace('.jpg', f'_flow_{str(latent_sharp_id).zfill(2)}.jpg'))
                
                latent_image_np = latent_render_pkg["render"].permute(1, 2, 0).cpu().numpy()
                latent_image_save = Image.fromarray((np.clip(latent_image_np,0,1) * 255).astype('uint8'))
                latent_image_save.save(path.replace('.jpg', f'_latent_{str(latent_sharp_id).zfill(2)}.jpg'))
                latent_images.append(latent_render_pkg["render"])
            latent_images.append(render_pkg["render"])
            # render_pkg["render"] = torch.mean(torch.stack(latent_images), dim=0)
            blurry_image = torch.mean(torch.stack(latent_images), dim=0)
            blurry_image_np = blurry_image.permute(1, 2, 0).cpu().numpy()
            
        coarse_render_pkg = render_func(viewpoint, static_gaussians, dynamic_gaussians, pipe, background, stage="coarse", 
                                        cam_type=cam_type, get_static=True, get_dynamic=True, over_t=viewpoint.time)
        # static_render_pkg = render_func(viewpoint, static_gaussians, pipe, background, stage=stage, cam_type=dataset_type, is_static=True)
        
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end

        image = render_pkg["render"] #* soft_blending[:, 0] + static_render_pkg["render"] * soft_blending[:, 1]
        coarse_image = coarse_render_pkg["render"] #* soft_blending[:, 0] + static_render_pkg["render"] * soft_blending[:, 1]

        d_image = render_pkg["d_render"]
        s_image = render_pkg["s_render"]

        d_alpha = render_pkg["d_alpha"]

        depth = render_pkg["depth"]
        st_depth = coarse_render_pkg["depth"]

        z = render_pkg["depth"] + 1e-6
        camera_metadata = viewpoint.metadata
        pixels = camera_metadata.get_pixels()
        y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.focal_length
        x = (
                    pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
            ) / camera_metadata.focal_length
        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        viewdirs = torch.from_numpy(viewdirs).to(z.device)

        coords = viewdirs[None] * z[..., None]
        coords = coords.permute(0, 3, 1, 2)

        dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
        dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
        dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
        dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
        dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
        dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

        dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
        dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
        dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

        dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
        dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
        dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

        n_x = dydv * dzdu - dydu * dzdv
        n_y = dzdv * dxdu - dzdu * dxdv
        n_z = dxdv * dydu - dxdu * dydv

        pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
        pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)

        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        d_image_np = d_image.permute(1, 2, 0).cpu().numpy()
        s_image_np = s_image.permute(1, 2, 0).cpu().numpy()
        
        d_alpha_np = d_alpha.permute(1, 2, 0).cpu().numpy()
        d_alpha_np = np.repeat(d_alpha_np, 3, axis=2)

        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)

        st_depth_np = st_depth.permute(1, 2, 0).cpu().numpy()
        st_depth_np /= depth_np.max()
        st_depth_np = np.repeat(st_depth_np, 3, axis=2)

        pred_normal_np = (pred_normal[0].permute(1, 2, 0).cpu().numpy() + 1) / 2

        error = (image_np - gt_np)**2
        error_np = (error - np.min(error)) / (max(np.max(error) - np.min(error), 1e-8))

        if is_train:
            gt_normal = (viewpoint.normal.cuda() + 1) / 2
            gt_normal_np = gt_normal.permute(1, 2, 0).cpu().numpy()

            gt_depth = viewpoint.depth.cuda()
            # gt_depth = gt_depth / F.avg_pool2d(gt_depth.detach(), kernel_size=gt_depth.shape[1::])
            gt_depth_np = gt_depth.permute(1, 2, 0).cpu().numpy()
            gt_depth_np /= gt_depth_np.max()
            gt_depth_np = np.repeat(gt_depth_np, 3, axis=2)

            decomp_image_np = np.concatenate((gt_normal_np, pred_normal_np, gt_depth_np, depth_np), axis=1)
            
            mask_np = viewpoint.mask.permute(1,2,0).cpu().numpy()
            mask_np = np.repeat(mask_np, 3, axis=2)
            if blcekernel is not None:
                image_np = np.concatenate((gt_np, blurry_image_np, image_np, d_alpha_np), axis=1)
            else:
                image_np = np.concatenate((gt_np, image_np, d_alpha_np, d_image_np, s_image_np), axis=1)
        else:
            decomp_image_np = np.concatenate((pred_normal_np, depth_np), axis=1)
            image_np = np.concatenate((gt_np, image_np, error_np, d_alpha_np, d_image_np, s_image_np), axis=1)
        
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
        decomp_image_with_labels = Image.fromarray((np.clip(decomp_image_np,0,1) * 255).astype('uint8'))
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
        decomp_image_with_labels.save(path.replace('.jpg', '_decomp.jpg'))

    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    if is_train:
        image_path = os.path.join(render_base_path,"train/images")
    else:
        image_path = os.path.join(render_base_path,"val/images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # image:3,800,800
    
    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{viewpoints[idx].image_name}.jpg")
        if idx == 0:
            render(stat_gaussians, dyn_gaussians, viewpoints[idx], viewpoints[idx+1], viewpoints[idx], image_save_path,
                scaling = 1, cam_type=dataset_type, over_t=over_t)
        elif idx == len(viewpoints) - 1:
            render(stat_gaussians, dyn_gaussians, viewpoints[idx], viewpoints[idx], viewpoints[idx-1], image_save_path,
                scaling = 1, cam_type=dataset_type, over_t=over_t)
        else:
            render(stat_gaussians, dyn_gaussians, viewpoints[idx], viewpoints[idx+1], viewpoints[idx-1], image_save_path,
                scaling = 1, cam_type=dataset_type, over_t=over_t)
    # render(gaussians,point_save_path,scaling = 0.1)
    # 保存带有标签的图像

    pc_mask = dyn_gaussians.get_opacity
    pc_mask = pc_mask > 0.1
    xyz = dyn_gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1, 0).numpy()
    # visualize_and_save_point_cloud(xyz, viewpoint.R, viewpoint.T, point_save_path)
    # 如果需要，您可以将PIL图像转换回PyTorch张量
    # return image
    # image_with_labels_tensor = torch.tensor(image_with_labels, dtype=torch.float32).permute(2, 0, 1) / 255.0


def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    # 应用旋转和平移变换
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)  # 转置点云数据以匹配Open3D的格式
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 可视化点云
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # 保存渲染结果为图片
    plt.savefig(filename)
