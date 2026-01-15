import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid


# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min, ref_type=None):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min, ref_type)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        # self.trbfcenter_mlp = nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.trbfscale_mlp = nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.motion_mlp = nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 9))
        # self.omega_mlp = nn.Sequential(nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))

        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 7))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))

    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            hidden = torch.cat([rays_pts_emb[:,:3], time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 

        hidden = self.feature_out(hidden)   

        return hidden

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)

        if self.args.static_mlp:
            # print("Using static mlp")
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            # print("Using empy voxel")
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            # print("Using ones")
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)

        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            # max_dx = torch.abs(self.get_aabb[0] - self.get_aabb[1]).type_as(dx).detach()
            # dx = torch.tanh(dx) * max_dx.view(1, 3)
            pts = rays_pts_emb[:,:3] * mask + dx[:, 0:3]
            se3rot = quat2mat(dx[:, 3::]) # nGs, 3, 3
            pts = se3rot.bmm(pts.view(pts.shape[0], 3, 1)).view(pts.shape[0], 3)

        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            ds[ds > math.log(100)] = math.log(100)
            ds[ds < -math.log(100)] = -math.log(100)
            scales = scales_emb[:,:3] * mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr
            rotations = batch_quaternion_multiply(rotations, dx[:, 3::])

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden)
            opacity = opacity_emb[:,:1]*mask + do

        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs

    def forward_dynamic2(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, time_emb)

        if self.args.static_mlp:
            # print("Using static mlp")
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            # print("Using empy voxel")
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            # print("Using ones")
            mask = torch.ones_like(rays_pts_emb[:,0]).unsqueeze(-1)

        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            # max_dx = torch.abs(self.get_aabb[0] - self.get_aabb[1]).type_as(dx).detach()
            # dx = torch.tanh(dx) * max_dx.view(1, 3)
            pts = rays_pts_emb[:,:3] * mask + dx[:, 0:3]
            se3rot = quat2mat(dx[:, 3::]) # nGs, 3, 3
            pts = se3rot.bmm(pts.view(pts.shape[0], 3, 1)).view(pts.shape[0], 3)

        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            ds[ds > math.log(100)] = math.log(100)
            ds[ds < -math.log(100)] = -math.log(100)
            scales = scales_emb[:,:3] * mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr
            rotations = batch_quaternion_multiply(rotations, dx[:, 3::])

        return pts, scales, rotations

    def forward_poly(self, rays_pts_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, time_emb)

        # trbfcenter = self.trbfcenter_mlp(hidden)
        # trbfcenter = (1 + torch.tanh(trbfcenter)) / 2 # [0,1]
         
        # trbfscale = self.trbfscale_mlp(hidden)
        motion = self.motion_mlp(hidden)
        omega = self.omega_mlp(hidden)

        return motion, omega

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list


class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales, rotations, times_sel):
        return self.forward_dynamic2(point, scales, rotations, times_sel)
        # return self.forward_poly(point, times_sel)

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb

    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs


    def forward_dynamic2(self, point, scales=None, rotations=None, times_sel=None):
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        means3D, scales, rotations = self.deformation_net.forward_dynamic2(point_emb, scales_emb, rotations_emb, times_sel)
        return means3D, scales, rotations


    def forward_poly(self, point, times_sel=None):
        point_emb = poc_fre(point, self.pos_poc)
        motion, omega = self.deformation_net.forward_poly(point_emb, times_sel)
        return motion, omega


    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


class pose_network(nn.Module):
    def __init__(self, args, train_cams=None):
        super(pose_network, self).__init__()
        timebase_pe = 10# args.timebase_pe
        timenet_width = 256 #args.timenet_width
        timenet_output = 6
        times_ch = 2 * timebase_pe + 1
        # Input to timenet (time to pose) is times_ch (try depth.mean + depth.min + depth.max + depth.var?)
        self.timenet0 = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_width), nn.ReLU())
        self.timenet1 = nn.Sequential(
            nn.Linear(timenet_width + times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_width), nn.ReLU())

        self.timenet_out = nn.Linear(timenet_width, timenet_output, bias=False)
        self.depth_scale_net_out = nn.Linear(timenet_width, 1, bias=False)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))

        pixel_base_pe = 5
        pixel_ch = 2 * (2 * pixel_base_pe + 1)
        self.depth_net = nn.Sequential(
            nn.Linear(times_ch + pixel_ch + 1 + 3, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_width), nn.ReLU())

        # self.depth_out = nn.Linear(timenet_width, 1, bias=False)
        self.depth_out = nn.Linear(timenet_width, 1, bias=False)

        self.register_buffer('pixel_poc', torch.FloatTensor([(2**i) for i in range(pixel_base_pe)]))

        self.apply(initialize_weights)
        self.timenet_out.weight.data.fill_(1e-6)
        self.focal_bias = nn.Parameter(torch.ones(1) * math.log(500), requires_grad=True)
        
        # if train_cams is not None:
        #     self.instance_scale_list = []
        #     self.instance_mask_list = []
        #     for cam in train_cams:
        #         self.instance_mask_list.append(cam.instance_mask[None])
        #         self.instance_scale_list.append(torch.ones(cam.instance_mask.shape[0])[None])
        #     self.instance_mask_list = nn.Parameter(torch.cat(self.instance_mask_list, 0), requires_grad=False)
        #     self.instance_scale_list = nn.Parameter(torch.cat(self.instance_scale_list, 0), requires_grad=True)
        #     self.max_time = train_cams[0].max_time
        
        if train_cams is not None:
            self.instance_scale_list = []
            for i, cam in enumerate(train_cams):
                if i == 0:
                    instance_init = torch.ones(1)[None] * 1.0
                else:
                    instance_init = torch.ones(1)[None] * 1.0
                self.instance_scale_list.append(instance_init)
            self.instance_scale_list = nn.Parameter(torch.cat(self.instance_scale_list, 0), requires_grad=True)
            self.max_time = train_cams[0].max_time
            self.H = train_cams[0].image_height
            self.W = train_cams[0].image_width

    def forward(self, times_sel, depth=None, normals=None, pixel=None):
        times_emb = poc_fre(times_sel[:, None], self.time_poc)
        time_index = (times_sel * self.max_time).long()
        pose_feature = self.timenet0(times_emb[:, 0])
        pose_feature = self.timenet_out(self.timenet1(torch.cat((pose_feature, times_emb[:, 0]), 1)))
        
        if depth is None:
            return euler2mat(pose_feature[:, 0:3]), pose_feature[:, 3::]

        # N_time, N_instance, H, W, _ = self.instance_mask_list.shape
        # instance_mask = torch.gather(self.instance_mask_list, 0, time_index[:,None, None, None].expand(-1, N_instance, H, W, -1)) # B, N_instance, H, W, 1
        # instance_scale = torch.gather(self.instance_scale_list[...,None], 0, time_index[:,None].expand(-1, N_instance, -1)) # B, N_instance, 1
        # cannonical_scale = self.instance_scale_list[0].detach()[None, None] # 1, N_instance, 1
        # instance_scale = instance_scale / cannonical_scale
        
        # instance_scale_mask = instance_mask * instance_scale[:,:, None, None] # B, N_instance, H, W, 1
        # instance_scale_mask = instance_scale_mask.sum(1).permute(0,3,1,2) # B, 1, H, W
        # instance_scale_mask[instance_scale_mask == 0] = 1.0 # fill zeros scaling with 1.0
        # CVD = depth.view(time_index.shape[0], 1, H, W) * instance_scale_mask
        
        instance_scale = torch.gather(self.instance_scale_list, 0, time_index) # B, 1
        cannonical_scale = self.instance_scale_list[0].detach()[None] # 1, 1
        instance_scale = instance_scale / cannonical_scale
        
        CVD = depth.view(time_index.shape[0], 1, self.H, self.W) * instance_scale[:,:, None, None]
        
        # import cv2
        # img = depth.view(time_index.shape[0], 1, H, W)[0].squeeze().cpu().numpy()
        # img = (img) / (img.max())
        # cv2.imwrite('depth.png', (img * 255.0).astype(np.uint8))
        
        # img = CVD.view(time_index.shape[0], 1, H, W)[0].squeeze().detach().cpu().numpy()
        # img = (img) / (img.max())
        # cv2.imwrite('depth_cvd.png', (img * 255.0).astype(np.uint8))
        # breakpoint()

        return euler2mat(pose_feature[:, 0:3]), pose_feature[:, 3::], CVD

    def get_mlp_parameters(self):
        exclude_params = ['instance_scale_list', 'focal_bias']
        filtered_params = [param for name, param in self.named_parameters() if name not in exclude_params]
        filtered_params_generator = (param for param in filtered_params)
        return filtered_params_generator
    def get_scale_parameters(self):
        return [self.instance_scale_list]
    def get_focal_parameters(self):
        return [self.focal_bias]
    def get_all_parameters(self):
        return self.parameters()


def quat2mat(quat):
   """Convert quaternion coefficients to rotation matrix.

   Args:
       quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
   Returns:
       Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
   """
   norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
   norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
   w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

   B = quat.size(0)

   w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
   wx, wy, wz = w*x, w*y, w*z
   xy, xz, yz = x*y, x*z, y*z

   rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                         2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                         2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
   return rotMat


def euler2mat(angle):
   """Convert euler angles to rotation matrix.

    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

   Args:
       angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
   Returns:
       Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
   """
   B = angle.size(0)
   x, y, z = angle[:,0], angle[:,1], angle[:,2]

   cosz = torch.cos(z)
   sinz = torch.sin(z)

   zeros = z.detach()*0
   ones = zeros.detach()+1
   zmat = torch.stack([cosz, -sinz, zeros,
                       sinz,  cosz, zeros,
                       zeros, zeros,  ones], dim=1).view(B, 3, 3)

   cosy = torch.cos(y)
   siny = torch.sin(y)

   ymat = torch.stack([cosy, zeros,  siny,
                       zeros,  ones, zeros,
                       -siny, zeros,  cosy], dim=1).view(B, 3, 3)

   cosx = torch.cos(x)
   sinx = torch.sin(x)

   xmat = torch.stack([ones, zeros, zeros,
                       zeros,  cosx, -sinx,
                       zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

   rotMat = xmat.bmm(ymat).bmm(zmat)
   return rotMat


pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w)  # .type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w)  # .type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(i_range)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords

    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h or pixel_coords.size(3) < w:
        set_id_grid(depth)

    # Convert pixel locations to camera locations
    current_pixel_coords = pixel_coords[:, :, :h, :w].type_as(depth).expand(b, 3, h, w).contiguous().view(b, 3,
                                                                                                          -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)

    # Weight these locations by the normalized depth? this means min depth of 1m max depth 1/beta = 100m?
    return cam_coords * depth.unsqueeze(1)


def direct_warp_rt1_rt2(img, depth, w2c1, w2c2, intrinsics, intrinsics_inv, padding_mode='zeros',
                        ret_grid=False, no_bucket=64):
    B, C, H, W = img.shape

    # Convert to disparity and get min max
    disp = 1 / depth
    min_disp = 1 / 100
    max_disp = 1 / 0.1

    # Discretize disparity (using min-max would help in not wasting discretization bins)
    zero2one = torch.linspace(0, 1, steps=no_bucket).type_as(disp)
    # disp_buckets = max_disp * (max_disp / min_disp) ** (zero2one - 1) # exp
    disp_buckets = zero2one * (max_disp - min_disp) + min_disp # linear
    disp_vol = torch.bucketize(disp.view(B, H, W), disp_buckets, right=True)
    disp_vol = F.one_hot(disp_vol, num_classes=no_bucket) # B, H, W, no_buckets
    disp_vol = torch.permute(disp_vol, (0, 3, 1, 2))

    # Get warping grid for each disp level
    depth_levels = 1 / disp_buckets.view(1, no_bucket, 1, 1, 1).repeat(B, 1, 1, 1, 1)
    depth_levels = torch.ones(B * no_bucket, 1, H, W).type_as(disp) * depth_levels.view(-1, 1, 1, 1)
    grid_vol = inverse_warp_grid_rt1_rt2(depth_levels,
                                         torch.repeat_interleave(w2c1, repeats=no_bucket, dim=0),
                                         torch.repeat_interleave(w2c2, repeats=no_bucket, dim=0),
                                         torch.repeat_interleave(intrinsics, repeats=no_bucket, dim=0),
                                         torch.repeat_interleave(intrinsics_inv, repeats=no_bucket, dim=0))

    # Warp disp and input image volumes and dot product for final forward warped image
    o_im0 = torch.repeat_interleave(img, repeats=no_bucket, dim=0)
    fw_img = F.grid_sample(o_im0, grid_vol, align_corners=True, padding_mode=padding_mode).view(B, no_bucket, C, H, W)

    disp_vol = disp_vol.reshape(-1, 1, H, W).float()
    Dprob_ = F.grid_sample(disp_vol, grid_vol, align_corners=True, padding_mode=padding_mode).view(B, -1, H, W)
    Dprob = F.softmax(Dprob_, dim=1)

    fw_img = (fw_img * Dprob.unsqueeze(2)).sum(1)

    # Get occlusion map
    with torch.no_grad():
        occ_map = Dprob_.sum(1).unsqueeze(1)
        occ_map[occ_map > 1] = 1

    if ret_grid:
        return fw_img, occ_map, grid_vol
    else:
        return fw_img, occ_map


def occ_rt1_rt2(depth, w2c1, w2c2, intrinsics, intrinsics_inv, padding_mode='zeros', no_bucket=128):
    with torch.no_grad():
        B, _, H, W = depth.shape

        # Convert to disparity and get min max
        disp = 1 / depth
        min_disp = 1/100
        max_disp = 1/0.1

        # Discretize disparity (using min-max helps in not wasting discretization bins)
        zero2one = torch.linspace(0, 1, steps=no_bucket).type_as(disp)
        # disp_buckets = max_disp * (max_disp / min_disp) ** (zero2one - 1) # exp
        disp_buckets = zero2one * (max_disp - min_disp) + min_disp # linear
        disp_vol = torch.bucketize(disp.view(B, H, W), disp_buckets, right=False)
        disp_vol = F.one_hot(disp_vol, num_classes=no_bucket) # B, H, W, no_buckets
        disp_vol = torch.permute(disp_vol, (0, 3, 1, 2))

        # Get warping grid for each disp level
        depth_levels = 1 / disp_buckets.view(1, no_bucket, 1, 1, 1).repeat(B, 1, 1, 1, 1)
        depth_levels = torch.ones(B * no_bucket, 1, H, W).type_as(disp) * depth_levels.view(-1, 1, 1, 1)
        grid_vol = inverse_warp_grid_rt1_rt2(depth_levels,
                                             torch.repeat_interleave(w2c1, repeats=no_bucket, dim=0),
                                             torch.repeat_interleave(w2c2, repeats=no_bucket, dim=0),
                                             torch.repeat_interleave(intrinsics, repeats=no_bucket, dim=0),
                                             torch.repeat_interleave(intrinsics_inv, repeats=no_bucket, dim=0))

        disp_vol = disp_vol.reshape(-1, 1, H, W).float()
        Dprob_ = F.grid_sample(disp_vol, grid_vol, align_corners=True, padding_mode=padding_mode).view(B, -1, H, W)
        occ_map = Dprob_.sum(1).unsqueeze(1)
        occ_map[occ_map > 1] = 1

        return occ_map


def warp_pc2flow(img, pc2, w2c2, intrinsics, padding_mode='zeros', ret_grid=False):
    """
    Optical flow from projected pointclouds (same pixel contains world coordinate of correspoding point).

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, _, H, W = pc2.shape
    R2 = w2c2[:, :, 0:3]
    t2 = w2c2[:, :, 3, None]

    # 1. Get camera coordinates for PC2 seen from w2c2
    c2 = torch.bmm(R2, pc2.view(B, 3, -1)) + t2

    # 2. Get pixel coordinates of PC2 seen from frame at w2c2
    z = c2[:, 2, None, :]
    z[torch.abs(z) < 1e-6] = 1e-6
    c2_ = c2 / z
    p2 = torch.bmm(intrinsics, c2_)

    # 3. Warp and grid
    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = 2 * X / (W - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode,
                                                    align_corners=True)

    if ret_grid:
        return projected_img, src_pixel_coords
    else:
        return projected_img



def inverse_warp_rt1_rt2(img, depth, w2c1, w2c2, intrinsics, intrinsics_inv, padding_mode='zeros', ret_grid=False):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    depth = depth[:, 0]
    B, H, W = depth.shape

    R1 = w2c1[:, :, 0:3]
    t1 = w2c1[:, :, 3, None]
    R1_ = torch.transpose(R1, 2, 1)
    t1_ = -torch.bmm(R1_, t1)
    R2 = w2c2[:, :, 0:3]
    t2 = w2c2[:, :, 3, None]

    # 1. Lift into 3D cam coordinates, pixel coordinates is p=[u,v,1]: c1 = D1 * K_invp1
    c1 = pixel2cam(depth, intrinsics_inv) # [B,3,H,W]
    # c1[:, 1::, :, :] *= -1  # make (y?) z negative

    # 2. Get world coordinates from c1: w = R1'c1 - R1't1
    w = torch.bmm(R1_, c1.view(B, 3, -1)) + t1_

    # 3. Get camera coordinates in c2: c2 = R2w + R2t2
    c2 = torch.bmm(R2, w) + t2

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    # z = torch.abs(c2[:, 2, None, :])
    z = c2[:, 2, None, :]
    z[torch.abs(z) < 1e-6] = 1e-6
    c2_ = c2 / z
    # c2_[:, 2, :] = 1
    # c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = 2 * X / (W - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode,
                                                    align_corners=True)

    if ret_grid:
        return projected_img, src_pixel_coords
    else:
        return projected_img


def inverse_warp_grid_rt1_rt2(depth, w2c1, w2c2, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Returns the inverse warp grid to the target image plane.

    Args:
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    depth = depth[:, 0]
    B, H, W = depth.shape

    R1 = w2c1[:, :, 0:3]
    t1 = w2c1[:, :, 3, None]
    R1_ = torch.transpose(R1, 2, 1)
    t1_ = -torch.bmm(R1_, t1)
    R2 = w2c2[:, :, 0:3]
    t2 = w2c2[:, :, 3, None]

    # 1. Lift into 3D cam coordinates, pixel coordinates is p=[u,v,1]: c1 = D1 * K_invp1
    c1 = pixel2cam(depth, intrinsics_inv) # [B,3,H,W]

    # 2. Get world coordinates from c1: w = R1'c1 - R1't1
    w = torch.bmm(R1_, c1.view(B, 3, -1)) + t1_

    # 3. Get camera coordinates in c2: c2 = R2w + R2t2
    c2 = torch.bmm(R2, w) + t2

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = c2[:, 2, None, :]
    z[torch.abs(z) < 1e-6] = 1e-6
    c2_ = c2 / z
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = 2 * X / (W - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)

    return src_pixel_coords


def points_from_DRTK(depth, w2c1, intrinsics):
    """
    Get world coordinates

    Args:
        depth: depth map of the target image -- [B, H, W]
        w2c1: 6DoF pose parameters from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    depth = depth[:, 0]
    B, H, W = depth.shape

    R1 = w2c1[:, :, 0:3]
    t1 = w2c1[:, :, 3, None]
    R1_ = torch.transpose(R1, 2, 1)
    t1_ = -torch.bmm(R1_, t1)

    # 1. Lift into 3D cam coordinates, pixel coordinates is p=[u,v,1]: c1 = D1 * K_invp1
    c1 = pixel2cam(depth, torch.inverse(intrinsics)) # [B,3,H,W]

    # 2. Get world coordinates from c1: w = R1'c1 - R1't1
    world_points = torch.bmm(R1_, c1.view(B, 3, -1)) + t1_

    return world_points


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb