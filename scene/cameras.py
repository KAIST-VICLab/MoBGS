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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal, getWorld2View2_torch
import dycheck_geometry

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, max_time = 0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, metadata=None, normal=None, depth=None, sem_mask=None,
                 fwd_flow=None, bwd_flow=None, fwd_flow_mask=None, bwd_flow_mask=None,
                 instance_mask=None, tracklet=None, 
                    query_tracks_2d= None,
                    target_ts= None,
                    target_tracks_2d= None,
                    target_visibles= None,
                    target_invisibles= None,
                    target_confidences= None,
                    target_track_depths= None,
                    target_ts_all= None,
                    target_tracks_2d_all= None,
                    target_visibles_all= None,
                    target_invisibles_all= None,
                    target_confidences_all= None,
                    target_track_depths_all= None,
                    K = None,
                    covisible=None,
                    depth_mask=None,
                    sharp_img=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.max_time = max_time
        self.target_num = 3
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.sharp_img = sharp_img
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
        
        self.normal = torch.Tensor(normal).permute(2,0,1).to(self.data_device) if normal is not None else None
        self.metadata = metadata if metadata is not None else None
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        self.sharp_img_tensor = sharp_img.clamp(0.0, 1.0)[:3,:,:] if sharp_img is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.a_chann = image[3, None] if len(image.shape) == 4 else None
        self.focal = fov2focal(FoVx, self.image_width)

        if self.a_chann is not None and self.a_chann.mean() < 1:
                print(f"Alpha channel is less than 1: {image[3].mean()}")

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
        self.depth = torch.Tensor(depth).permute(2,0,1).to(self.data_device) if depth is not None else None
        self.sem_mask = torch.Tensor(sem_mask).to(self.data_device) if sem_mask is not None else None
        self.instance_mask = torch.Tensor(instance_mask).to(self.data_device) if instance_mask is not None else None
        self.tracklet = torch.Tensor(tracklet).to(self.data_device) if tracklet is not None else None
        self.mask = torch.Tensor(mask).permute(2,0,1).to(self.data_device) if mask is not None else None
        self.fwd_flow = torch.Tensor(fwd_flow).permute(2,0,1).to(self.data_device) if fwd_flow is not None else None
        self.bwd_flow = torch.Tensor(bwd_flow).permute(2,0,1).to(self.data_device) if bwd_flow is not None else None
        self.fwd_flow_mask = torch.Tensor(fwd_flow_mask).permute(2,0,1).to(self.data_device) if fwd_flow_mask is not None else None
        self.bwd_flow_mask = torch.Tensor(bwd_flow_mask).permute(2,0,1).to(self.data_device) if bwd_flow_mask is not None else None
        self.query_tracks_2d = torch.Tensor(query_tracks_2d).to(self.data_device) if query_tracks_2d is not None else None
        self.zfar = 100.0
        self.znear = 0.01

        self.target_ts = torch.Tensor(target_ts).to(self.data_device) if target_ts is not None else None
        self.target_tracks_2d = torch.Tensor(target_tracks_2d).to(self.data_device) if target_tracks_2d is not None else None
        self.target_visibles = torch.Tensor(target_visibles).to(self.data_device) if target_visibles is not None else None
        self.target_invisibles = torch.Tensor(target_invisibles).to(self.data_device) if target_invisibles is not None else None
        self.target_confidences = torch.Tensor(target_confidences).to(self.data_device) if target_confidences is not None else None
        self.target_track_depths = torch.Tensor(target_track_depths).to(self.data_device) if target_track_depths is not None else None

        self.target_ts_all = torch.Tensor(target_ts_all).to(self.data_device) if target_ts_all is not None else None
        self.target_tracks_2d_all = torch.Tensor(target_tracks_2d_all).to(self.data_device) if target_tracks_2d_all is not None else None
        self.target_visibles_all = torch.Tensor(target_visibles_all).to(self.data_device) if target_visibles_all is not None else None
        self.target_invisibles_all = torch.Tensor(target_invisibles_all).to(self.data_device) if target_invisibles_all is not None else None
        self.target_confidences_all = torch.Tensor(target_confidences_all).to(self.data_device) if target_confidences_all is not None else None
        self.target_track_depths_all = torch.Tensor(target_track_depths_all).to(self.data_device) if target_track_depths_all is not None else None

        self.K = torch.Tensor(K).to(self.data_device) if K is not None else None
        self.covisible = torch.Tensor(covisible).to(self.data_device) if covisible is not None else None
        self.depth_mask = torch.Tensor(depth_mask).to(self.data_device).permute(2,0,1) if depth_mask is not None else None

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        if type(R) == np.ndarray:
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        else:
            self.world_view_transform = getWorld2View2_torch(R, T, torch.tensor(trans), scale).transpose(0, 1)
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = torch.inverse(self.world_view_transform)[3, :3]

        if type(self.camera_center) == np.ndarray:     
            pixels = self.get_pixels(self.metadata.image_size_x, self.metadata.image_size_y, use_center=True)
            viewdirs = self.pixels_to_viewdirs(pixels)
            cam_origin = np.broadcast_to(self.camera_center, viewdirs.shape)
            cam_ray = np.concatenate((cam_origin, viewdirs), axis=-1)
            cam_ray = np.transpose(cam_ray, (2,0,1))[None, ...]
            cam_ray = torch.tensor(cam_ray)
            self.cam_ray = cam_ray.to(self.data_device)
        else:
            pixels = self.get_pixels_torch(self.metadata.image_size_x, self.metadata.image_size_y, use_center=True)      
            viewdirs = self.pixels_to_viewdirs_torch(pixels)
            cam_origin, _ = torch.broadcast_tensors(self.camera_center, viewdirs)
            cam_ray = torch.cat((cam_origin, viewdirs), dim=-1)
            cam_ray = cam_ray.permute(2, 0, 1).unsqueeze(0)
            self.cam_ray = cam_ray
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)

    def update_target_ts(self, rand_target=None, init=False):
        # if init:
        #     self.target_ts = self.target_ts_all[self.init_target_ts]
        #     self.target_tracks_2d = self.target_tracks_2d_all[self.init_target_ts]
        #     self.target_visibles = self.target_visibles_all[self.init_target_ts]
        #     self.target_invisibles = self.target_invisibles_all[self.init_target_ts]
        #     self.target_confidences = self.target_confidences_all[self.init_target_ts]
        #     self.target_track_depths = self.target_track_depths_all[self.init_target_ts]
        # else:

        self.target_ts = self.target_ts_all[rand_target]
        self.target_tracks_2d = self.target_tracks_2d_all[rand_target]
        self.target_visibles = self.target_visibles_all[rand_target]
        self.target_invisibles = self.target_invisibles_all[rand_target]
        self.target_confidences = self.target_confidences_all[rand_target]
        self.target_track_depths = self.target_track_depths_all[rand_target]

    def update_cam(self, R, T, local_viewdirs, batch_shape, focal=None):
        if focal is not None:
            self.focal = focal
            new_metadata =  dycheck_geometry.Camera(
                orientation=self.metadata.orientation,
                position=self.metadata.position,
                focal_length=np.array([focal]).astype(np.float32),
                principal_point=self.metadata.principal_point,
                image_size=self.metadata.image_size,
            )
            self.metadata = new_metadata

            self.FoVy = focal2fov(focal, self.image_height)
            self.FoVx = focal2fov(focal, self.image_width)

        self.R = R
        self.T = T
        self.trans = np.array([.0, .0, .0])
        self.scale = 1.0
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        viewdirs = (torch.tensor(self.R).to(self.data_device) @ torch.tensor(local_viewdirs[..., None]).to(self.data_device))[..., 0]
        viewdirs /= torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs.reshape((*batch_shape, 3))
        
        cam_origin = self.camera_center.to(self.data_device).expand(viewdirs.shape) # -> [270, 480, 3]
        cam_ray = torch.cat((cam_origin, viewdirs), dim=-1)
        cam_ray = cam_ray.permute(2,0,1).unsqueeze(0)

        self.cam_ray = cam_ray

    def get_pixels(self, image_size_x, image_size_y, use_center = None):
        """Return the pixel at center or corner."""
        xx, yy = np.meshgrid(
            np.arange(image_size_x, dtype=np.float32),
            np.arange(image_size_y, dtype=np.float32),
        )
        offset = 0.5 if use_center else 0
        return np.stack([xx, yy], axis=-1) + offset


    def pixels_to_local_viewdirs(self, pixels: np.ndarray):
        """Return the local ray viewdirs for the provided pixels."""
        y = (pixels[..., 1] - self.metadata.principal_point_y) / self.metadata.scale_factor_y
        x = (pixels[..., 0] - self.metadata.principal_point_x) / self.metadata.scale_factor_x

        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)

        return viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)


    def pixels_to_viewdirs(self, pixels: np.ndarray) -> np.ndarray:
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")

        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))

        local_viewdirs = self.pixels_to_local_viewdirs(pixels)
        if type(self.R) == np.ndarray:
            viewdirs = (self.R @ local_viewdirs[..., None])[..., 0]
        else:
            viewdirs = (self.R.clone().detach().cpu().numpy() @ local_viewdirs[..., None])[..., 0]

        # Normalize rays.
        viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = viewdirs.reshape((*batch_shape, 3))
        return viewdirs

    def get_pixels_torch(self, image_size_x, image_size_y, use_center=None):
        """Return the pixel at center or corner using PyTorch tensors."""
        xx, yy = torch.meshgrid(
            torch.arange(image_size_x, dtype=torch.float32, device=self.data_device),
            torch.arange(image_size_y, dtype=torch.float32, device=self.data_device),
            indexing="xy"
        )
        offset = 0.5 if use_center else 0
        pixels = torch.stack([xx, yy], dim=-1) + offset
        return pixels.to(self.data_device) # 마지막 차원으로 좌표 결합

    def pixels_to_local_viewdirs_torch(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the local ray viewdirs for the provided pixels using PyTorch."""

        scale_factor_y = torch.tensor(self.metadata.scale_factor_y, dtype=torch.float32, device=self.data_device)
        scale_factor_x = torch.tensor(self.metadata.scale_factor_x, dtype=torch.float32, device=self.data_device)

        y = (pixels[..., 1] - self.metadata.principal_point_y) / scale_factor_y
        x = (pixels[..., 0] - self.metadata.principal_point_x) / scale_factor_x

        viewdirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)

        return (viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)).to(self.data_device)

    def pixels_to_viewdirs_torch(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")

        batch_shape = pixels.shape[:-1]
        pixels = pixels.view(-1, 2)  # Reshape (-1, 2)

        local_viewdirs = self.pixels_to_local_viewdirs_torch(pixels)  # 이미 Tensor라고 가정
        if isinstance(self.R, torch.Tensor):
            viewdirs = torch.matmul(self.R, local_viewdirs[..., None])[..., 0]
        else:
            viewdirs = torch.matmul(torch.tensor(self.R, dtype=torch.float32, device=self.data_device), local_viewdirs[..., None])[..., 0]

        # Normalize rays
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs.view(*batch_shape, 3)
        return viewdirs


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

