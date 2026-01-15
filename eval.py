import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
from PIL import Image
import models

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, blceParams
from argparse import ArgumentParser
from gaussian_renderer import render
import cv2

from tqdm import tqdm
import imageio
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import inspect
from scene.blce import blceKernel

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


torch.manual_seed(0)
def get_pixels(image_size_x, image_size_y, use_center = None):
    """Return the pixel at center or corner."""
    xx, yy = np.meshgrid(
        np.arange(image_size_x, dtype=np.float32),
        np.arange(image_size_y, dtype=np.float32),
    )
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset


def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]

def normalize_image(img):
    return (2.0 * img - 1.0)[None, ...]

def render_test_tto(
    H,
    W,
    scene,
    test_cams,
    save_dir,
    gt_rgb_dir,
    ##
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    ###
    # dbg
    use_sgd=False,
    loss_type="psnr",
    # boost
    initialize_from_previous_camera=True,
    initialize_from_previous_step_factor=10,
    initialize_from_previous_lr_factor=0.1,
    fg_mask_th=0.1,
    local_viewdirs=None,
    batch_shape=None,
    renderArgs=None
):

    device = scene.stat_gaussians.get_xyz.device
    solved_pose_list = []
    total_psnr = 0.0
    total_lpips = 0.0
    total_ssim = 0.0

    lpips_fn = models.PerceptualLoss(model='net-lin',net='alex',
                                    use_gpu=True,version=0.1)
    for i in tqdm(range(len(test_cams))):
        if initialize_from_previous_camera and i == 0:
            step_factor = initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = initialize_from_previous_lr_factor


        current_camera = test_cams[i]
        # load gt rgb and mask
        gt_rgb = imageio.imread(os.path.join(gt_rgb_dir, f"{current_camera.image_name}.png")) / 255.0
        gt_rgb = cv2.resize(gt_rgb, (W, H))
        gt_rgb = gt_rgb[..., :3]

        gt_rgb = torch.tensor(gt_rgb, device=device).float()


        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        w2c = current_camera.world_view_transform.transpose(0,1).to(device) # this is row format
        
        t_init = torch.nn.Parameter(w2c[:3, 3].clone().detach(),  requires_grad=True)
        q_init = torch.nn.Parameter(matrix_to_quaternion(w2c[:3, :3]).clone().detach(), requires_grad=True)
        if use_sgd:
            optimizer_type = torch.optim.SGD
        else:
            optimizer_type = torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p * lr_factor},
                {"params": q_init, "lr": lr_q * lr_factor},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tto_steps * step_factor - decay_start,
            eta_min=lr_final * lr_factor,
        )

        loss_list = []
        

        for _step in range(tto_steps * step_factor):
            optimizer.zero_grad()
            curr_w2c = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            curr_w2c = torch.cat([curr_w2c, T_bottom[None]], 0)
            
            render_pkg = render(current_camera, scene.stat_gaussians, scene.dyn_gaussians, stage="fine", get_static = False, get_dynamic=False,
                cam_type=scene.dataset_type, *renderArgs, w2c=curr_w2c)
            
            pred_rgb = render_pkg["render"].permute(1, 2, 0)
            # rendered_mask = render_pkg["d_alpha"].squeeze() == 0.0

            if loss_type == "abs":
                raise RuntimeError("Should not use this")
                rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None]
                rgb_loss = rgb_loss_i.sum() / gt_mask_sum
            elif loss_type == "psnr":
                # mse = ((pred_rgb - gt_rgb) ** 2)[rendered_mask].mean()
                mse = ((pred_rgb - gt_rgb) ** 2).mean()
                psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
                rgb_loss = -psnr_value

            else:
                raise ValueError(f"Unknown loss tyoe {loss_type}")

            loss = rgb_loss
            loss.backward()
            optimizer.step()
            if _step >= decay_start:
                scheduler.step()

            loss_list.append(loss.item())

        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        solved_pose_list.append(solved_T_cw.detach().cpu().numpy())
        with torch.no_grad():
            render_pkg = render(current_camera, scene.stat_gaussians, scene.dyn_gaussians, stage="fine", get_static = False, get_dynamic=True,
                cam_type=scene.dataset_type, *renderArgs, w2c=solved_T_cw)
            
            image = render_pkg["render"]
            image = torch.clamp(image, 0.0, 1.0)     

            img = Image.fromarray((np.clip(image.permute(1, 2, 0).detach().cpu().numpy(),0,1) * 255).astype('uint8'))
            os.makedirs(save_dir + '/test_refined', exist_ok=True)
            img.save(save_dir + '/test_refined/img_{}.png'.format(f"{current_camera.image_name}.png"))

    np.save(os.path.join(save_dir, "solved_poses.npy"), np.stack(solved_pose_list, 0))

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    cp = blceParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[300] + [1000 * i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 20000, 30_000, 45000,
                                 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-render_process", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default="output/spline_stereo_blurfeat_ode/street/point_cloud/iteration_10000")
    parser.add_argument("--expname", type=str, default="spline_stereo_blurfeat_ode/street")
    parser.add_argument("--configs", type=str, default="arguments/stereo/street.py")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)
    opt = op.extract(args)
    blceopt = cp.extract(args)

    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)  # for other datasets rather than iPhone dataset

    
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams]
    viewpoint_stack = [i for i in train_cams]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, 'point_cloud.ply'))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, 'point_cloud_static.ply'))

    dyn_gaussians.load_model(args.checkpoint)
    blcekernel = blceKernel(num_views=len(viewpoint_stack),
                            view_dim=blceopt.view_dim,
                            num_warp=blceopt.num_warp,
                            method=blceopt.method,
                            adjoint=blceopt.adjoint,
                            iteration=opt.iterations).cuda()
    blcekernel.model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'blce.pth')))
    
    # Compute view_dir
    pixels = get_pixels(scene.train_camera.dataset[0].metadata.image_size_x, scene.train_camera.dataset[0].metadata.image_size_y, use_center=True)
    if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y) / scene.train_camera.dataset[0].metadata.focal_length
    x = (pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x) / scene.train_camera.dataset[0].metadata.focal_length
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs =  viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True) 
    
    
    # freeze gaussians attributes
    for attr in inspect.getmembers(dyn_gaussians):
        try:
            attr[1].requires_grad = False
        except:
            pass
    for attr in inspect.getmembers(stat_gaussians):
        try:
            attr[1].requires_grad = False
        except:
            pass
        
    frames = render_test_tto(
        gt_rgb_dir=os.path.join(dataset.source_path, "inference_images"),
        tto_steps=100,
        decay_start=30,
        lr_p=0.0003,
        lr_q=0.0003,
        lr_final=0.000001,
        use_sgd=False,
        #
        H=test_cams[0].image_height,
        W=test_cams[0].image_width,
        scene=scene,
        save_dir=os.path.join("output", args.expname),
        test_cams=my_test_cams,
        #
        initialize_from_previous_camera=False,
        initialize_from_previous_step_factor=1,
        initialize_from_previous_lr_factor=1.0,
        fg_mask_th=0.1,
        local_viewdirs=local_viewdirs,
        batch_shape=batch_shape,
        renderArgs=[pipe, background],
    )