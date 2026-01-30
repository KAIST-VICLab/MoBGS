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

import os
import sys
import cv2
from PIL import Image
from scene.cameras import Camera
import glob

from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm

from easydict import EasyDict as edict
from .colmap import get_colmap_camera_params
import imageio

import dycheck_geometry
from utils.dycheck_utils import io

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    max_time: float
    mask: np.array
    sem_mask: Optional[np.array] = None
    metadata: Optional[dycheck_geometry.Camera] = None
    normal: Optional[np.array] = None
    depth: Optional[np.array] = None
    fwd_flow: Optional[np.array] = None
    bwd_flow: Optional[np.array] = None
    fwd_flow_mask: Optional[np.array] = None
    bwd_flow_mask: Optional[np.array] = None
    instance_mask: Optional[np.array] = None
    tracklet: Optional[np.array] = None
    # tpair data
    query_tracks_2d: Optional[np.array] = None
    target_ts: Optional[np.array] = None
    target_tracks_2d: Optional[np.array] = None
    target_visibles: Optional[np.array] = None
    target_invisibles: Optional[np.array] = None
    target_confidences: Optional[np.array] = None
    target_track_depths: Optional[np.array] = None

    target_ts_all: Optional[np.array] = None
    target_tracks_2d_all: Optional[np.array] = None
    target_visibles_all: Optional[np.array] = None
    target_invisibles_all: Optional[np.array] = None
    target_confidences_all: Optional[np.array] = None
    target_track_depths_all: Optional[np.array] = None

    K: Optional[np.array] = None
    covisible: Optional[np.array] = None
    depth_mask: Optional[np.array] = None
    sharp_img: Optional[np.array] = None
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}

def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
                pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
        ) / camera_metadata.scale_factor_x
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
    return pred_normal

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

# def fetchPly(path):
#     plydata = PlyData.read(path)
#     vertices = plydata['vertex']
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)

# def storePly(path, xyz, rgb):
#     # Define the dtype for the structured array
#     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#             ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
#     normals = np.zeros_like(xyz)

#     elements = np.empty(xyz.shape[0], dtype=dtype)
#     # breakpoint()
#     attributes = np.concatenate((xyz, normals, rgb), axis=1)
#     elements[:] = list(map(tuple, attributes))

#     # Create the PlyData object and write to file
#     vertex_element = PlyElement.describe(elements, 'vertex')
#     ply_data = PlyData([vertex_element])
#     ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos
def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        # breakpoint()
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            
            # breakpoint()
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            # breakpoint()
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            # breakpoint()
            # print(w2c,focal,image_name)
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # breakpoint()
    return cam_infos, max_time, scene_radius

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

# def readIphoneCameras_record3D(path):
#     scene_json = io.load(os.path.join(path, "scene.json"))
#     coord_scale = np.array(scene_json['scale']).astype(np.float32)
#     scene_center = np.array(scene_json['center']).astype(np.float32)
#     scene_factor = 2
#
#     train_split_json = io.load(os.path.join(path, "splits", "train.json"))
#     test_split_json = io.load(os.path.join(path, "splits", "val.json"))
#     train_times = np.array(train_split_json["time_ids"], np.uint32)
#     max_time = train_times.max()
#     train_cam_infos = []
#     test_cam_infos = []
#
#     # Load train infos
#     for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
#         frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
#         img_path = os.path.join(
#                         path,
#                         'rgb_sharp',
#                         "{}x".format(str(scene_factor)),
#                         frame_name + ".png",
#                     )
#         normal_path = img_path.replace('rgb_sharp', 'blurry_normal').replace('png', 'npy')
#         normal = io.load(normal_path)
#         curr_img = np.array(Image.open(img_path))
#         curr_img = Image.fromarray((curr_img).astype(np.uint8))
#         image_name = Path(img_path).stem
#
#         camera = (
#                     dycheck_geometry.Camera.fromjson(
#                         os.path.join(path, 'camera', frame_name + ".json")
#                     )
#                     .rescale_image_domain(1 / scene_factor)
#                     .translate(-scene_center)
#                     .rescale(coord_scale)
#                 )
#
#         R = camera.orientation
#         T = camera.translation #w2c
#         focal = camera.focal_length
#         fid = time_id / max_time
#
#         FovY = focal2fov(focal, curr_img.size[1])
#         FovX = focal2fov(focal, curr_img.size[0])
#
#         cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img,
#                               image_path=img_path, image_name=image_name, width=curr_img.size[
#                                   0], height=curr_img.size[1],
#                               time=fid, mask=None, metadata=camera, normal=normal)
#         train_cam_infos.append(cam_info)
#
#     # Load test cam
#     for idx, (time_id, camera_id) in enumerate(zip(np.array(test_split_json["time_ids"], np.uint32), np.array(test_split_json["camera_ids"], np.uint32))):
#         frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
#         img_path = os.path.join(
#                         path,
#                         'blurry_rgb',
#                         "{}x".format(str(scene_factor)),
#                         frame_name + ".png",
#                     )
#         curr_img = np.array(Image.open(img_path))
#         curr_img = Image.fromarray((curr_img).astype(np.uint8))
#         image_name = Path(img_path).stem
#
#
#         camera = (
#                     dycheck_geometry.Camera.fromjson(
#                         os.path.join(path, 'camera', frame_name + ".json")
#                     )
#                     .rescale_image_domain(1 / scene_factor)
#                     .translate(-scene_center)
#                     .rescale(coord_scale)
#                 )
#
#         R = camera.orientation #w2c
#         T = camera.translation #w2c
#         focal = camera.focal_length
#         fid = time_id / max_time
#
#         FovY = focal2fov(focal, curr_img.size[1])
#         FovX = focal2fov(focal, curr_img.size[0])
#         cam_info = CameraInfo(uid=idx + len(train_cam_infos), R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img,
#                               image_path=img_path, image_name=image_name, width=curr_img.size[
#                                   0], height=curr_img.size[1],
#                               time=fid, mask=None, metadata=camera)
#         test_cam_infos.append(cam_info)
#
#     return train_cam_infos, test_cam_infos, scene_center, coord_scale, max_time
#
# def readIphoneInfo_record3D(path):
#     print("Reading iPhone Info")
#     train_cam_infos, test_cam_infos, scene_center, scene_scale, max_time = readIphoneCameras(path)
#     nerf_normalization = getNerfppNorm(train_cam_infos)
#
#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         print(f"Generating point cloud from iphone...")
#
#         xyz = np.load(os.path.join(path, "points.npy"))
#         xyz = (xyz - scene_center) * scene_scale
#         num_pts = xyz.shape[0]
#
#         fps_inds = furthest_point_sample(torch.from_numpy(xyz[None]).float().cuda(), num_pts//15).long()
#         xyz = torch.gather(torch.from_numpy(xyz).float().cuda(),0, fps_inds.squeeze(0)[...,None].expand(-1,xyz.shape[-1]))
#         xyz = xyz.cpu().numpy()
#         num_pts = xyz.shape[0]
#
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
#             shs), normals=np.zeros((num_pts, 3)))
#
#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None
#
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            video_cameras=None,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path, maxtime=max_time)
#     return scene_info


def readIphoneCameras_record3D(path):
    # # iphone norm
    # coord_scale = np.array(scene_json['scale']).astype(np.float32)
    # scene_center = np.array(scene_json['center']).astype(np.float32)

    # som norm
    cached_scene_norm_dict_path = os.path.join(
        path, "flow3d_preprocessed/cache/scene_norm_dict.pth"
    )
    use_iphone_pose = False
    # if os.path.exists(cached_scene_norm_dict_path):
    if False:
        scene_norm_dict = torch.load(
            cached_scene_norm_dict_path
        )
        # coord_scale = 1.0 / float(scene_norm_dict['scale'])
        coord_scale = 1.0
        scene_transfm = (scene_norm_dict['transfm']).numpy()
    else:
        scene_json = io.load(os.path.join(path, "scene.json"))
        coord_scale = np.array(scene_json['scale']).astype(np.float32)
        scene_center = np.array(scene_json['center']).astype(np.float32)
        use_iphone_pose = True

    scene_factor = 2

    train_split_json = io.load(os.path.join(path, "splits", "train.json"))
    test_split_json = io.load(os.path.join(path, "splits", "val.json"))
    train_times = np.array(train_split_json["time_ids"], np.uint32)
    max_time = train_times.max()
    # max_time = 199
    train_cam_infos = []
    test_cam_infos = []

    os.makedirs(os.path.join(path, 'uni_normal'), exist_ok=True)

    # read tapir
    num_frames = max_time + 1
    tracks_dir = os.path.join(path, 'bootstapir/cam0')

    concat_target_tracks_all = []
    concat_target_visibles_all = []
    target_inds_all = torch.from_numpy(np.arange(num_frames))   
    for idx in  target_inds_all.tolist():
        target_tracks_all = load_target_tracks(tracks_dir,  idx, target_inds_all.tolist(), dim=0)[:,::5] # (N, P, 4)
        # (N, P)
        (
            target_visibles_all ,
            _ ,
            _
        ) = parse_tapir_track_info(target_tracks_all[..., 2], target_tracks_all[..., 3])
        # (N, H, W)
        concat_target_tracks_all.append(target_tracks_all)
        concat_target_visibles_all.append(target_visibles_all)

    concat_target_tracks_all = torch.concat(concat_target_tracks_all, dim = 1)
    concat_target_visibles_all = torch.concat(concat_target_visibles_all, dim = 1)

    if use_iphone_pose:
        # iphone depth
        depth_list = []
        for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            depth_path = os.path.join(
                            path,
                            'depth',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".npy",
                        )
            depth = np.load(depth_path) * coord_scale
            depth_list.append(depth)
        mean_depth = np.mean(np.stack(depth_list, 0))
    else:
        # som depth
        depth_list = []
        ori_H, ori_W = None, None
        for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            depth_path = os.path.join(
                            path,
                            'flow3d_preprocessed/aligned_depth_anything_colmap',
                            # "{}x".format(str(scene_factor)),
                            "1x",
                            frame_name + ".npy",
                        )
            depth = np.load(depth_path) # this is disp
            depth[depth < 1e-3] = 1e-3
            depth = 1.0 / depth

            depth = depth * coord_scale
            ori_H, ori_W = depth.shape
            depth = cv2.resize(depth, (int(ori_W/scene_factor), int(ori_H/scene_factor)), cv2.INTER_NEAREST)
            depth_list.append(depth)
        mean_depth = np.mean(np.stack(depth_list, 0))

    if not use_iphone_pose:
        train_Ks, train_w2cs = get_colmap_camera_params(
            os.path.join(path, "flow3d_preprocessed/colmap/sparse/"),
            ["{}_{}".format(str(c), str(t).zfill(5)) + ".png" for (t,c) in zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))],
        )
        test_Ks, test_w2cs = get_colmap_camera_params(
            os.path.join(path, "flow3d_preprocessed/colmap/sparse/"),
            ["{}_{}".format(str(c), str(t).zfill(5)) + ".png" for (t,c) in zip(np.array(test_split_json["time_ids"], np.uint32), np.array(test_split_json["camera_ids"], np.uint32))],
        )

    # Load train infos
    for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
        if time_id <= max_time:
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            img_path = os.path.join(
                            path,
                            'rgb',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".png",
                        )

            curr_img_rgba = Image.open(img_path)
            curr_img = np.array(curr_img_rgba.convert('RGB'))
            curr_img = Image.fromarray((curr_img).astype(np.uint8))

            image_name = Path(img_path).stem

            # nerfies scale first, depth scale later
            if use_iphone_pose:
                # iphone cam
                camera = (
                            dycheck_geometry.Camera.fromjson(
                                os.path.join(path, 'camera', frame_name + ".json")
                            )
                            .rescale_image_domain(1 / scene_factor)
                            .translate(-scene_center)
                            .rescale(coord_scale)
                            .rescale(1/mean_depth)
                        )
                K = camera.intrin
            else:            
                K = train_Ks[idx]
                w2c = train_w2cs[idx]
                
                w2c = w2c @ np.linalg.inv(scene_transfm)
                c2w = np.linalg.inv(w2c)
                camera =  dycheck_geometry.Camera(
                    orientation=w2c[:3,:3],
                    position=c2w[:3,3],
                    focal_length=np.array([K[0][0]]).astype(np.float32),
                    principal_point=np.array([K[0][2], K[1][2]]).astype(np.float32),
                    image_size=np.array([ori_W, ori_H]).astype(np.float32),
                ).rescale_image_domain(1 / scene_factor).rescale(coord_scale).rescale(1/mean_depth)

                K = (K[:3, :3].astype(np.float32))
                K[:2] /= scene_factor

            R = camera.orientation.T
            T = camera.translation #w2c
            focal = camera.focal_length
            fid = time_id / max_time
            FovY = focal2fov(focal, curr_img.size[1])
            FovX = focal2fov(focal, curr_img.size[0])

            if use_iphone_pose:
                depth_path = os.path.join(
                                path,
                                'depth',
                                "{}x".format(str(scene_factor)),
                                frame_name + ".npy",
                            )
                depth = np.load(depth_path) * coord_scale 
                depth = depth / mean_depth

                # align unidepth
                unidepth_path = img_path.replace('rgb/2x', 'uni_depth_v2').replace('.png', '.npy')
                unidepth = np.load(unidepth_path)
                scale = np.median(depth / unidepth)
                shift = np.median(depth - scale * unidepth)
                align_unidepth = scale * unidepth + shift
                depth = align_unidepth
            else:
                # som depth
                depth_path = os.path.join(
                                path,
                                'flow3d_preprocessed/aligned_depth_anything_colmap',
                                # "{}x".format(str(scene_factor)),
                                "1x",
                                frame_name + ".npy",
                            )
                depth = np.load(depth_path) # this is disp
                depth[depth < 1e-3] = 1e-3
                depth = 1.0 / depth
                depth = depth * coord_scale
                depth = cv2.resize(depth, (int(ori_W/scene_factor), int(ori_H/scene_factor)), cv2.INTER_NEAREST)
                depth = depth[...,None] / mean_depth

            normal_path = img_path.replace('rgb/2x', 'uni_normal').replace('.png', '.npy')
            if not os.path.exists(normal_path):
                normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], camera)
                normal = normal.squeeze(0).cpu().numpy().transpose(1,2,0)
                np.save(normal_path, normal)
            else:
                normal = np.load(normal_path)
            if np.isnan(normal).any():
                breakpoint()

            normal = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()

            # read cotracker
            # tracklet_path = os.path.join(path, 'cotrackers/forward_tracks_dynamic.npy')
            # tracklet = np.load(tracklet_path) # N_times, N_tracks, 2

            # visibility_path = os.path.join(path, 'cotrackers/forward_visibility_dynamic.npy')
            # visibility = np.load(visibility_path)
            target_tracks_all, target_visibles_all = None, None
            if idx == 0:
                target_tracks_all = concat_target_tracks_all[..., :2].numpy()
                target_visibles_all = concat_target_visibles_all.numpy()


            # read instance mask
            instance_path = img_path.replace('rgb/2x', 'masks/cam0')
            instance_mask_list = []

            instance_mask = np.array(Image.open(instance_path))[..., None]

            if instance_mask.shape[2] != 1:
                instance_mask = instance_mask[..., 0, :]

            instance_mask[instance_mask > 0] = 1
            instance_mask_list.append(instance_mask)

            instance_mask_list = np.stack(instance_mask_list, 0)
            if instance_mask.sum() == 0:
                breakpoint()

            new_mask = instance_mask_list[0]
            
            depth_mask = (depth > 0).astype(np.float32)

            cam_info = CameraInfo(uid=int(image_name[2:]), R=R.astype(np.float32), T=T.astype(np.float32), FovY=float(FovY), FovX=float(FovX), image=curr_img, image_path=img_path, max_time=float(max_time),
                                image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                                time=float(fid), mask=new_mask, metadata=camera, normal=normal.astype(np.float32), depth=depth.astype(np.float32), sem_mask=None,
                                instance_mask=instance_mask_list, tracklet=target_tracks_all, K=K, depth_mask=depth_mask,
                                )
            train_cam_infos.append(cam_info)

    # Load test cam
    for idx, (time_id, camera_id) in enumerate(zip(np.array(test_split_json["time_ids"], np.uint32), np.array(test_split_json["camera_ids"], np.uint32))):
        if time_id <= max_time:
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            img_path = os.path.join(
                            path,
                            'rgb',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".png",
                        )

            curr_img_rgba = Image.open(img_path)
            curr_img = np.array(curr_img_rgba.convert('RGB'))
            curr_img = Image.fromarray((curr_img).astype(np.uint8))

            image_name = Path(img_path).stem

            if use_iphone_pose:
                camera = (
                            dycheck_geometry.Camera.fromjson(
                                os.path.join(path, 'camera', frame_name + ".json")
                            )
                            .rescale_image_domain(1 / scene_factor)
                            .translate(-scene_center)
                            .rescale(coord_scale)
                            .rescale(1/mean_depth)
                        )
                K = camera.intrin
            else:
                K = test_Ks[idx]
                w2c = test_w2cs[idx]
                
                w2c = w2c @ np.linalg.inv(scene_transfm)
                c2w = np.linalg.inv(w2c)
                camera =  dycheck_geometry.Camera(
                    orientation=w2c[:3,:3],
                    position=c2w[:3,3],
                    focal_length=np.array([K[0][0]]).astype(np.float32),
                    principal_point=np.array([K[0][2], K[1][2]]).astype(np.float32),
                    image_size=np.array([ori_W, ori_H]).astype(np.float32),
                ).rescale_image_domain(1 / scene_factor).rescale(coord_scale).rescale(1/mean_depth)

                K = (K[:3, :3].astype(np.float32))
                K[:2] /= scene_factor

            R = camera.orientation.T
            T = camera.translation
            focal = camera.focal_length
            fid = time_id / max_time

            FovY = focal2fov(focal, curr_img.size[1])
            FovX = focal2fov(focal, curr_img.size[0])
            
            # covisible mask
            covisible = (imageio.imread(os.path.join(path, 'covisible/2x/val', f"{frame_name}.png")) > 0).astype(float)[..., None]           

            cam_info = CameraInfo(uid=int(image_name[2:]), R=R.astype(np.float32), T=T.astype(np.float32), FovY=float(FovY), FovX=float(FovX), image=curr_img, image_path=img_path, max_time=float(max_time),
                                image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                                time=float(fid), mask=None, metadata=camera, normal=normal, depth=None, sem_mask=None, K=K,
                                instance_mask=None, tracklet=None, covisible=covisible,
                                )
            test_cam_infos.append(cam_info)

    return train_cam_infos, test_cam_infos, max_time


def readIphoneInfo_record3D(path):
    print("Reading Iphone Info")
    
    train_cam_infos, test_cam_infos, max_time = readIphoneCameras_record3D(path)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    start_time = 0
    totalxyz = []
    totalrgb = []
    totaltime = []

    xyz = rgb = np.zeros((100, 3)) # dummy

    for i in range(start_time, start_time+max_time):        
        totalxyz.append(xyz)
        totalrgb.append(rgb)
        totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)

    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)   

    ply_path = os.path.join(path, "points3D.ply")
    xyzt = np.concatenate((xyz, totaltime), axis=1)   
    assert xyz.shape[0] == rgb.shape[0]   
    storePly(ply_path, xyzt, rgb)    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=None, maxtime=max_time)
    return scene_info

def readIphoneCameras(path, cam_extrinsics, cam_intrinsics, images_folder):
    scene_json = io.load(os.path.join(path, "scene.json"))
    coord_scale = np.array(scene_json['scale']).astype(np.float32) 
    scene_center = np.array(scene_json['center']).astype(np.float32)
    scene_factor = 2

    train_split_json = io.load(os.path.join(path, "splits", "train.json"))
    test_split_json = io.load(os.path.join(path, "splits", "val.json"))
    train_times = np.array(train_split_json["time_ids"], np.uint32)
    max_time = train_times.max()
    train_cam_infos = []
    test_cam_infos = []
    colmap_dict = {}

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        colmap_dict[extr.name] = edict({
            'R':R,
            'T': T,
            'FovY':FovY,
            'FovX':FovX,
            'focal_length': focal_length_x,
            'height' : height,
            'width' : width,
        })

    # Load train infos
    for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
        frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
        img_path = os.path.join(
                        path,
                        'rgb_sharp',
                        "{}x".format(str(scene_factor)),
                        frame_name + ".png",
                    )
        normal_path = img_path.replace('rgb_sharp', 'blurry_normal').replace('png', 'npy')
        depth_path = img_path.replace('rgb_sharp', 'blurry_depth').replace('png', 'npy')
        sem_mask_path = os.path.join(path, 'sem_mask', frame_name, "mask.npy")
        sem_mask = io.load(sem_mask_path)
        normal = io.load(normal_path)
        depth = io.load(depth_path)
        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))
        image_name = Path(img_path).stem

        colmap_data = colmap_dict[f"{frame_name}.png"]
        R = colmap_data.R
        T = colmap_data.T

        fid = time_id / max_time
        FovY = colmap_data.FovY
        FovX = colmap_data.FovX

        W2C = np.eye(4)
        W2C[:3,:3] = R
        W2C[:3,3] = T
        C2W = np.linalg.inv(W2C)
        

        metadata =  dycheck_geometry.Camera(
            orientation=R,
            position=C2W[:3,3],
            focal_length=np.array([colmap_data.focal_length]).astype(np.float32),
            principal_point=np.array([colmap_data.width/2.0, colmap_data.height/2.0]).astype(np.float32),
            image_size=np.array([colmap_data.width, colmap_data.height]).astype(np.float32),
        )

        # mask = io.load(os.path.join(path, "bkgd_mask", str(time_id).zfill(5) + ".png"))[...,0:1]
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img,
                              image_path=img_path, image_name=image_name, width=curr_img.size[0],
                              height=curr_img.size[1], time=fid, mask=None, metadata=metadata,
                              normal=normal, depth=depth, sem_mask=sem_mask
                              )
        train_cam_infos.append(cam_info)

    # Load test cam
    for idx, (time_id, camera_id) in enumerate(zip(np.array(test_split_json["time_ids"], np.uint32), np.array(test_split_json["camera_ids"], np.uint32))):
        frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
        img_path = os.path.join(
                        path,
                        'rgb_sharp',
                        "{}x".format(str(scene_factor)),
                        frame_name + ".png",
                    )
        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))
        image_name = Path(img_path).stem
        
        for k in colmap_dict.keys():
            if f"{camera_id}_" in k:
                colmap_data = colmap_dict[k]
        R = colmap_data.R
        T = colmap_data.T

        fid = time_id / max_time

        FovY = colmap_data.FovY
        FovX = colmap_data.FovX

        W2C = np.eye(4)
        W2C[:3,:3] = R
        W2C[:3,3] = T
        C2W = np.linalg.inv(W2C)
        

        metadata =  dycheck_geometry.Camera(
            orientation=R,
            position=C2W[:3,3],
            focal_length=np.array([colmap_data.focal_length]).astype(np.float32),
            principal_point=np.array([colmap_data.width/2.0, colmap_data.height/2.0]).astype(np.float32),
            image_size=np.array([colmap_data.width, colmap_data.height]).astype(np.float32),
        )

        #
        cam_info = CameraInfo(uid=idx + len(train_cam_infos), R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img,
                              image_path=img_path, image_name=image_name, width=curr_img.size[
                                  0], height=curr_img.size[1], time=fid, mask=None, metadata=metadata
                              )
        test_cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos, scene_center, coord_scale, max_time

def readIphoneInfo(path):
    print("Reading iPhone Info")

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "rgb_sharp/2x"
    train_cam_infos, test_cam_infos, _, _, max_time = readIphoneCameras(path, cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    # if not os.path.exists(ply_path):
    if True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, maxtime=max_time)
    return scene_info

def read_flow(flow_path, img_size):
    flow_info = np.load(flow_path)
    flow, mask = flow_info['flow'], flow_info['mask']
    # flow[mask == False] = [0, 0]

    # flow_color = flow_to_color(flow)
    # cv2.imwrite("debug.png", flow_color)

    # flow[mask == False] = [0, 0]
    # flow_color = flow_to_color(flow)
    # cv2.imwrite("debug_mask.png", flow_color)
    # breakpoint()

    H,W,_ = flow.shape
    flow[...,0] = flow[...,0] / W
    flow[...,1] = flow[...,1] / H

    flow = cv2.resize(flow, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.float32), (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    return flow, mask

def load_target_tracks(tracks_dir, query_index, target_indices, dim = 1, scale=1.0):
    """
    tracks are 2d, occs and uncertainties
    :param dim (int), default 1: dimension to stack the time axis
    return (N, T, 4) if dim=1, (T, N, 4) if dim=0
    """
    q_name =  str(query_index).zfill(5)
    all_tracks = []
    for ti in target_indices:
        t_name =  str(ti).zfill(5)
        path = f"{tracks_dir}/{q_name}_{t_name}.npy"
        tracks = np.load(path).astype(np.float32)
        tracks[:,:2] = tracks[:,:2] / scale
        all_tracks.append(tracks)
    return torch.from_numpy(np.stack(all_tracks, axis=dim))


def parse_tapir_track_info(occlusions, expected_dist):
    """
    return:
        valid_visible: mask of visible & confident points
        valid_invisible: mask of invisible & confident points
        confidence: clamped confidence scores (all < 0.5 -> 0)
    """
    visiblility = 1 - F.sigmoid(occlusions)
    confidence = 1 - F.sigmoid(expected_dist)
    valid_visible = visiblility * confidence > 0.5
    valid_invisible = (1 - visiblility) * confidence > 0.5
    # set all confidence < 0.5 to 0
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence

def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0

prior_list = [
    # [[(288, 70),(291, 248)],[(313,55),(315,235+ 4)],[(326,74),(328,251+ 4)],[(339,55),(336,230+ 4)],[(359,70),(349,248+ 4)],[(386,86),(379,263+ 4)],[(363,106),(344,261+ 4)],[(345,109),(357,256+ 4)],[(334,130),(347,267)],[(321,126),(334,254+ 4)],[(313,124),(325,247+ 4)],[(292,141),(305,263+ 4)]],
    [[(288, 70),(291, 243)],[(313,55),(315,230)],[(326,74),(328,246)],[(339,55),(336,225)],[(359,70),(349,243)],[(386,86),(379,258)],[(363,106),(344,256)],[(345,109),(357,251)],[(334,130),(347,262)],[(321,126),(334,249)],[(313,124),(325,242)],[(292,141),(305,257)]],
    [],
    []
]

def readStereoCameras(path):
    final_height = 288
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy'))
    scene_center = np.array(json.load(open(os.path.join(path, 'scene.json')))["center"])
    
    poses = poses_arr[:, :15].reshape(-1, 3, 5)
    hwf = poses[0, :, -1]
    
    ori_h = 720
    factor =  ori_h / final_height
    
    # sh = hwf[:2] / factor
    sh = np.array([288, 512]) #! hardcode here
    focal_length = hwf[-1] / factor

    max_time = min((poses.shape[0]) // 2 - 1, 23)
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])
    c2ws = poses[:,:3,:4]

    train_c2ws = c2ws[::2]
    test_c2ws = c2ws[1::2]

    train_cam_infos = []
    test_cam_infos = []
    os.makedirs(os.path.join(path, 'uni_normal'), exist_ok=True)
    
    # normal_dir = os.path.join(path, 'normal')
    # disp_dir = os.path.join(path, 'norm_disp')
    # if not os.path.exists(normal_dir):
    #     os.mkdir(normal_dir)
    # if not os.path.exists(disp_dir):
    #     os.mkdir(disp_dir)

    # read tapir
    num_frames = max_time + 1
    tracks_dir = os.path.join(path, 'bootstapir')

    concat_target_tracks_all = []
    concat_target_visibles_all = []
    target_inds_all = torch.from_numpy(np.arange(num_frames))   
    for idx in  target_inds_all.tolist():
        target_tracks_all = load_target_tracks(tracks_dir,  idx, target_inds_all.tolist(), dim=0) # (N, P, 4)
        # (N, P)
        (
            target_visibles_all ,
            _ ,
            _
        ) = parse_tapir_track_info(target_tracks_all[..., 2], target_tracks_all[..., 3])
        # (N, H, W)
        concat_target_tracks_all.append(target_tracks_all)
        concat_target_visibles_all.append(target_visibles_all)

    concat_target_tracks_all = torch.concat(concat_target_tracks_all, dim = 1)
    concat_target_visibles_all = torch.concat(concat_target_visibles_all, dim = 1)

    depth_list = []
    for idx in range(max_time + 1):
        frame_name = f"{idx:05d}.png"
        depth_path = os.path.join(path, 'align_uni_depth_noclip', frame_name.replace('.png', '.npy'))
        depth = np.load(depth_path)[..., None]
        depth_list.append(depth)
    mean_depth = np.mean(np.stack(depth_list, 0))
    
    for idx in range(max_time + 1):
        frame_name = f"{idx:05d}.png"
        img_path = os.path.join(path, 'images_512x288', frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        curr_c2w = train_c2ws[idx]
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        K = metadata.intrin

        depth_path = os.path.join(path, 'align_uni_depth_noclip', frame_name.replace('.png', '.npy'))
        uni_depth = np.load(depth_path)[..., None] / mean_depth
        depth = uni_depth

        normal_path = os.path.join(path, 'uni_normal', frame_name.replace('.png', '.npy'))
        if not os.path.exists(normal_path):
            normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], metadata)
            normal = normal.squeeze(0).cpu().numpy().transpose(1,2,0)
            np.save(normal_path, normal)
        else:
            normal = np.load(normal_path)
        if np.isnan(normal).any():
            breakpoint()
            
        # apply avg pooling to normal
        normal = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()
        target_tracks_all, target_visibles_all = None, None
        if idx == 0:
            target_tracks_all = concat_target_tracks_all[..., :2].numpy()
            target_visibles_all = concat_target_visibles_all.numpy()
            
        motion_mask_path = os.path.join(path, 'motion_masks_manual', frame_name)
        motion_mask = PILtoTorch(Image.open(motion_mask_path), (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1).numpy()
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=motion_mask, metadata=metadata, normal=normal, depth=depth, sem_mask=None,
                              instance_mask=None, tracklet=target_tracks_all, K=K, sharp_img=None,
                              )
        train_cam_infos.append(cam_info)

    for idx in range(max_time + 1):
        frame_name = f"{idx:05d}.png"
        img_path = os.path.join(path, 'inference_images', frame_name)
        image_name = Path(img_path).stem

        curr_img = cv2.resize(np.array(Image.open(img_path)), (sh[1], sh[0]))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        curr_c2w = test_c2ws[idx]
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )
        K = metadata.intrin
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=None, metadata=metadata, K=K)
        test_cam_infos.append(cam_info)
    return train_cam_infos, test_cam_infos, max_time

def readNvidiaCameras(path):
    # load img factor of 2
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy'))
    scene_center = np.array(json.load(open(os.path.join(path, 'scene.json')))["center"])
    # poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # bds = poses_arr[:, -2:].transpose([1,0])
    factor = 2

    # max_time = poses.shape[-1] - 1 # 12-1 = 11
    # hwf = poses[:,4,0]
    # sh = hwf[:2] / factor
    # focal_length = hwf[-1] / factor

    # poses = np.concatenate([poses[:, 1:2, :],
    #                        -poses[:, 0:1, :],
    #                         poses[:, 2:, :]], 1)
    # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    
    poses = poses_arr[:, :15].reshape(-1, 3, 5)
    hwf = poses[0, :, -1]
    sh = hwf[:2] / factor
    focal_length = hwf[-1] / factor

    max_time = poses.shape[0] - 1
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])
    c2ws = poses[:,:3,:4]

    train_cam_infos = []
    test_cam_infos = []
    normal_dir = os.path.join(path, 'normal')
    disp_dir = os.path.join(path, 'norm_disp')
    if not os.path.exists(normal_dir):
        os.mkdir(normal_dir)
    if not os.path.exists(disp_dir):
        os.mkdir(disp_dir)

    # max_depth = 0.0
    # min_depth = 1e6
    # for idx in range(max_time + 1):
    #     frame_name = f"{idx:03d}.png"
    #     depth_path = os.path.join(path, 'uni_depth', frame_name.replace('.png', '.npy'))
    #     depth = np.load(depth_path)[..., None]
    #     if depth.max() > max_depth:
    #         max_depth = depth.max()
    #     if depth.min() < min_depth:
    #         min_depth = depth.min()
    # min_depth = min_depth - 1e-2 * (max_depth - min_depth)
    
    depth_list = []
    for idx in range(max_time + 1):
        frame_name = f"{idx:03d}.png"
        depth_path = os.path.join(path, 'align_uni_depth_noclip', frame_name.replace('.png', '.npy'))
        depth = np.load(depth_path)[..., None]
        depth_list.append(depth)
    mean_depth = np.mean(np.stack(depth_list, 0))
    
    # target_tracks_all = []
    # for idx in range((max_time + 1)):
    #     frame_name = f"{idx:03d}.png"
    #     img_path = os.path.join(path, 'images_2', frame_name)
    #     # read tapir
    #     num_frames = max_time + 1
    #     tracks_dir = os.path.join(img_path.replace('images_2', 'bootstapir/cam0')[:-8])

    #     target_inds_all = torch.from_numpy(np.arange(num_frames))
    #     # (N, P, 4)
    #     target_tracks_all.append(load_target_tracks(tracks_dir, idx, target_inds_all.tolist(), dim=0))

    # target_tracks_all = torch.cat(target_tracks_all, dim=1)

    # # (N, P)
    # (
    #     target_visibles_all ,
    #     target_invisibles_all ,
    #     target_confidences_all
    # ) = parse_tapir_track_info(target_tracks_all[..., 2], target_tracks_all[..., 3])
    # # (N, H, W)

    for idx in range(max_time + 1):
        frame_name = f"{idx:03d}.png"
        img_path = os.path.join(path, 'images_2', frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        # load flow
        fwd_flow_path = os.path.join(path, 'flow', frame_name.replace('.png', '_fwd.npz'))
        bwd_flow_path = os.path.join(path, 'flow', frame_name.replace('.png', '_bwd.npz'))

        if idx == 0:
            fwd_flow, fwd_mask = read_flow(fwd_flow_path, sh)
            bwd_flow, bwd_mask = np.zeros_like(fwd_flow), np.zeros_like(fwd_mask)
        elif idx == max_time:
            bwd_flow, bwd_mask = read_flow(bwd_flow_path, sh)
            fwd_flow, fwd_mask = np.zeros_like(bwd_flow), np.zeros_like(bwd_mask)
        else:
            fwd_flow, fwd_mask = read_flow(fwd_flow_path, sh)
            bwd_flow, bwd_mask = read_flow(bwd_flow_path, sh)

        norm_disp_path = os.path.join(disp_dir, frame_name)
        if not os.path.exists(norm_disp_path):
            disp_path = os.path.join(path, 'disp', frame_name.replace('.png', '.npy'))
            disp = np.load(disp_path)

            disp_min = disp.min()
            disp_max = disp.max()

            max_val = (2 ** (8 * 2)) - 1

            if disp_max - disp_min > np.finfo("float").eps:
                norm_disp = max_val * (disp - disp_min) / (disp_max - disp_min)
            else:
                norm_disp = np.zeros(disp.shape, dtype=disp.dtype)
            cv2.imwrite(norm_disp_path, norm_disp.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        norm_disp = Image.open(norm_disp_path)
        norm_disp = torch.clamp(PILtoTorch(norm_disp, (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1), 0, 255)
        # depth_from_disp = 255.0 - norm_disp.numpy()
        # depth_from_disp = cv2.resize(depth_from_disp, dsize=(int(sh[1]), int(sh[0])), interpolation=cv2.INTER_CUBIC)[..., None]
        
        disp_path = os.path.join(path, 'disp', frame_name.replace('.png', '.npy'))
        disp = np.load(disp_path)
        depth_from_disp = 1.0 / (disp + 1e-6)
        depth_from_disp = cv2.resize(depth_from_disp, dsize=(int(sh[1]), int(sh[0])), interpolation=cv2.INTER_CUBIC)[..., None]
        
        # load gt mask
        # mask_path = os.path.join(path, 'motion_masks_manual', frame_name)
        # mask = PILtoTorch(Image.open(mask_path), (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1).numpy()
        # mask[mask > 0] = 1

        curr_c2w = c2ws[idx]
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        depth_path = os.path.join(path, 'align_uni_depth_noclip', frame_name.replace('.png', '.npy'))
        uni_depth = np.load(depth_path)[..., None] / mean_depth
        
        # depth_from_disp_mean, depth_from_disp_std = depth_from_disp.mean(), depth_from_disp.std()
        # uni_depth_mean, uni_depth_std = uni_depth.mean(), uni_depth.std()
        # aligned_depth_from_disp = np.clip(((depth_from_disp - depth_from_disp_mean) / depth_from_disp_std) * uni_depth_std + uni_depth_mean, 0, None)
        # depth = np.min(np.concatenate([aligned_depth_from_disp, uni_depth], axis=-1), axis=-1)[..., None]
        # depth = np.zeros_like(uni_depth)
        depth = uni_depth
        
        # cv2.imwrite("debug.png", (depth / depth.max() * 255).astype(np.uint8))

        normal_path = os.path.join(path, 'uni_normal', frame_name.replace('.png', '.npy'))
        if not os.path.exists(normal_path):
            normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], metadata)
            normal = normal.squeeze(0).cpu().numpy().transpose(1,2,0)
            np.save(normal_path, normal)
        else:
            normal = np.load(normal_path)
        if np.isnan(normal).any():
            breakpoint()
            
        # apply avg pooling to normal
        normal = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()

        # if idx == 6:
        #     normal_1 = (normal + 1) / 2
        #     cv2.imwrite("debug.png", (normal_1 * 255).astype(np.uint8))
        #     # apply avg pooling to normal
        #     normal_2 = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()
        #     normal_2 = (normal_2 + 1) / 2
        #     cv2.imwrite("debug_2.png", (normal_2 * 255).astype(np.uint8))
        #     breakpoint()
        tracklet_path_fwd = os.path.join(path, 'forward_tracks_dynamic.npy')
        tracklet_fwd = np.load(tracklet_path_fwd)
        
        # tracklet_path_bwd = os.path.join(path, 'backward_tracks_dynamic.npy')
        # tracklet_bwd = np.load(tracklet_path_bwd)
        # tracklet = np.load(tracklet_path)
        # tracklet = np.concatenate([tracklet_fwd, tracklet_bwd], axis=1)
        tracklet = tracklet_fwd

        # # read tpair
        # # (P, 2)
        # num_targets_per_frame = 3
        # num_frames = 12
        # tracks_dir = os.path.join(path, 'bootstapir')
        # query_tracks = load_target_tracks(tracks_dir, idx, [idx])[:, 0, :2]
        # target_inds = torch.from_numpy(
        #     np.random.choice(
        #         num_frames, (num_targets_per_frame,), replace=False
        #     )
        # )

        # # inds = np.arange(num_frames).tolist()
        # # target_inds = torch.from_numpy(np.array([i for i in inds if i != idx]))

        # # (N, P, 4)
        # target_tracks = load_target_tracks(tracks_dir, idx, target_inds.tolist(), dim=0)
        # # (N, P).
        # (
        #     target_visibles,
        #     target_invisibles,
        #     target_confidences
        # ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # # (N, H, W)

        # target_depths = torch.stack([torch.from_numpy(np.load(os.path.join(path, 'uni_depth', f"{str(i).zfill(3)}.npy"))) / mean_depth for i in target_inds.tolist()], dim=0)
        # H, W = target_depths.shape[-2:]
        # target_track_depths = F.grid_sample(
        #     target_depths[:, None],
        #     normalize_coords(target_tracks[..., None, :2], H, W),
        #     align_corners=True,
        #     padding_mode="border",
        # )[:, 0, :, 0]

        # target_inds_all = torch.from_numpy(np.arange(num_frames))

        # # (N, P, 4)
        # target_tracks_all = load_target_tracks(tracks_dir, idx, target_inds_all.tolist(), dim=0)

        # # (N, P)
        # (
        #     target_visibles_all ,
        #     target_invisibles_all ,
        #     target_confidences_all
        # ) = parse_tapir_track_info(target_tracks_all[..., 2], target_tracks_all[..., 3])
        # # (N, H, W)

        # target_depths_all = torch.stack([torch.from_numpy(np.load(os.path.join(path, 'uni_depth', f"{str(i).zfill(3)}.npy"))) / mean_depth for i in target_inds_all.tolist()], dim=0)
        # H, W = target_depths_all.shape[-2:]
        # target_track_depths_all = F.grid_sample(
        #     target_depths_all[:, None],
        #     normalize_coords(target_tracks_all[..., None, :2], H, W),
        #     align_corners=True,
        #     padding_mode="border",
        # )[:, 0, :, 0]

        # read instance mask
        instance_path = os.path.join(path, 'instance_mask_manual', frame_name.split('.')[0] + '/*.png')
        instance_mask_list = []
        for mask_path in sorted(glob.glob(instance_path)):
            instance_mask = PILtoTorch(Image.open(mask_path), (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1).numpy()
            instance_mask[instance_mask > 0] = 1
            instance_mask_list.append(instance_mask)
        instance_mask_list = np.stack(instance_mask_list, 0)

        new_mask = np.zeros_like(instance_mask_list[0])
        for instance in instance_mask_list:
            new_mask = np.maximum(new_mask, instance)
            
        # for instance_i, instance in enumerate(instance_mask_list):
        #     if len(prior_list[instance_i]) == 0 or instance_i != 0:
        #         continue
        #     prior_instance = prior_list[instance_i][idx][0]
        #     prior_ground = prior_list[instance_i][idx][1]
        #     # scale_factor = depth[prior_ground[1], prior_ground[0]] / depth[prior_instance[1], prior_instance[0]]
        #     scale_factor = depth[prior_ground[1], prior_ground[0]] / depth[instance == 1].mean()
        #     # if idx == 11:
        #     #     breakpoint()
        #     depth[instance == 1] *= scale_factor
        #     print(scale_factor)
        
        # depth[new_mask == 1] = aligned_depth_from_disp[new_mask == 1]
        # depth[new_mask == 0] = uni_depth[new_mask == 0]
        # cv2.imwrite("debug_2.png", (depth / depth.max() * 255).astype(np.uint8))
        
        # video_disp = cv2.resize(np.load(os.path.join(path, 'depthcrafter.npz'))['depth'][idx], (int(sh[1]), int(sh[0])))[..., None]
        # if idx == 0:
        #     ca_video_disp = cv2.resize(np.load(os.path.join(path, 'depthcrafter.npz'))['depth'][idx], (int(sh[1]), int(sh[0])))[..., None]
        #     ca_video_disp_factor = np.mean(ca_video_disp[instance_mask_list[0] == 1])
        #     ca_depth_factor = np.mean(depth[instance_mask_list[0] == 1])
            
        
        # video_scale_factor = (ca_video_disp_factor / np.mean(video_disp[instance_mask_list[0] == 1]))
        # depth_scale_factor = (np.mean(depth[instance_mask_list[0] == 1]) / ca_depth_factor)
        # scale_factor = video_scale_factor / depth_scale_factor
        # # scale_factor = 1.0
        # print(video_scale_factor, depth_scale_factor, scale_factor)
        # depth[instance_mask_list[0] == 1] *= scale_factor
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=new_mask, metadata=metadata, normal=normal, depth=depth, sem_mask=None,
                              fwd_flow=fwd_flow, bwd_flow=bwd_flow, fwd_flow_mask=fwd_mask[...,None], bwd_flow_mask=bwd_mask[...,None],
                              instance_mask=instance_mask_list, tracklet=tracklet, 
                            #   query_tracks_2d=query_tracks, target_ts=target_inds,
                            #   target_tracks_2d=target_tracks[..., :2], target_visibles=target_visibles,
                            #   target_invisibles=target_invisibles, target_confidences=target_confidences, target_track_depths=target_track_depths,
                            #   target_ts_all=target_inds_all,
                            #   target_tracks_2d_all=target_tracks_all[..., :2], target_visibles_all=target_visibles_all,
                            #   target_invisibles_all=target_invisibles_all, target_confidences_all=target_confidences_all, target_track_depths_all=target_track_depths_all
                              )
        train_cam_infos.append(cam_info)

    for idx in range(max_time + 1):
        frame_name = f"v000_t{idx:03d}.png"
        img_path = os.path.join(path, 'gt', frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        curr_c2w = c2ws[0]
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=None, metadata=metadata, normal=None, depth=None, sem_mask=None)
        test_cam_infos.append(cam_info)
    return train_cam_infos, test_cam_infos, max_time

def readNvidiaInfo(path):
    print("Reading Nvidia Info")
    train_cam_infos, test_cam_infos, max_time = readStereoCameras(path)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")

    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")

    totalxyz = []
    totalrgb = []
    totaltime = []
    
    start_time = 0

    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    for i in range(start_time, start_time+max_time):        
        totalxyz.append(xyz)
        totalrgb.append(rgb)
        totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)
    
    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)    
    assert xyz.shape[0] == rgb.shape[0]
    xyzt = np.concatenate((xyz, totaltime), axis=1)     
    storePly(ply_path, xyzt, rgb)    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, maxtime=max_time)
    return scene_info

def readNvidiaInfo_stg(path):
    print("Reading Nvidia Info")
    train_cam_infos, test_cam_infos, max_time = readNvidiaCameras(path)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    
    
    totalxyz = []
    totalrgb = []
    totaltime = []
    
    start_time = 0

    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    
    # for i in range(start_time, start_time+max_time):        
    #     totalxyz.append(xyz)
    #     totalrgb.append(rgb)
    #     totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)

    # random init
    num_pts = xyz.shape[0]
    colmap_std = np.std(xyz, axis=0)
    colmap_mean = np.mean(xyz, axis=0)

    
    for i in range(start_time, start_time+max_time):
        xyz = np.random.normal(size=(num_pts, 3)) * colmap_std + colmap_mean
        shs = np.random.random((num_pts, 3)) / 255.0
        rgb = SH2RGB(shs) * 255
        
        totalxyz.append(xyz)
        totalrgb.append(rgb)
        totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)


    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)    
    assert xyz.shape[0] == rgb.shape[0]
    xyzt = np.concatenate((xyz, totaltime), axis=1)     
    storePly(ply_path, xyzt, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, maxtime=max_time)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    # "iphone": readIphoneInfo
    "iphone": readIphoneInfo_record3D,
    "nvidia": readNvidiaInfo,
}
