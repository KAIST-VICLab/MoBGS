import torch
import numpy as np
from matplotlib import cm
from PIL import Image
import os

__all__ = ('save_debug_imgs', 'get_normals', 'sw_cams', 'sw_depth_normalization', 'error_to_prob', 'get_rays')


def error_to_prob(error, mask=None, mean_prob=0.5):
    if mask is None:
        mean_err = torch.mean(error, dim=(3, 2, 1)) + 1e-7
    else:
        mean_err = torch.sum(mask * error, dim=(3, 2)) / (torch.sum(mask, dim=(3, 2)) + 1e-7) + 1e-7
    prob = mean_prob * (error / mean_err.view(error.shape[0], 1, 1, 1))
    prob[prob > 1] = 1
    prob = 1 - prob
    return prob


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().type_as(K)
    j = j.t().type_as(K)
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
    flow_map[:, (flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[0, :, :] += normalized_flow_map[0, :, :]
    rgb_map[1, :, :] -= 0.5 * (normalized_flow_map[0, :, :] + normalized_flow_map[1, :, :])
    rgb_map[2, :, :] += normalized_flow_map[1, :, :]
    return rgb_map.clip(0, 1)


def save_debug_imgs(debug_dict, idx, epoch=0, deb_path=None, ext='jpg'):
    new_outputs = {}
    for key in debug_dict.keys():
        out_tensor = debug_dict[key].detach()
        if out_tensor.shape[1] == 3:
            if 'normal' in key:
                p_im = (out_tensor[idx].squeeze().cpu().numpy() + 1) / 2
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                p_im = np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8)
            else:
                p_im = out_tensor[idx].squeeze().cpu().numpy()
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                p_im = np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8)
        elif out_tensor.shape[1] == 2:
            p_im = flow2rgb(out_tensor[idx].squeeze().cpu().numpy(), None)
            p_im = np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8)
        elif out_tensor.shape[1] == 1:
            if 'disp' in key:
                nmap = out_tensor[idx].squeeze().cpu().numpy()
                nmap = np.clip(nmap / (np.percentile(nmap, 99) + 1e-6), 0, 1)
                p_im = (255 * cm.plasma(nmap)).astype(np.uint8)
            elif 'error' in key:
                nmap = out_tensor[idx].squeeze().cpu().numpy()
                nmap = np.clip(nmap, 0, 1)
                p_im = (255 * cm.jet(nmap)).astype(np.uint8)
            else:
                B, C, H, W = out_tensor.shape
                the_max = torch.max_pool2d(out_tensor, kernel_size=(H, W))
                nmap = out_tensor / the_max
                p_im = nmap[idx].squeeze().cpu().numpy()
                p_im = np.rint(255 * p_im).astype(np.uint8)

        # Save or return normalized image
        if deb_path is not None:
            if len(p_im.shape) == 3: p_im = p_im[:,:,0:3]
            im = Image.fromarray(p_im)
            im.save(os.path.join(deb_path, 'e{}_{}.{}'.format(epoch, key, ext)))
        else:
            new_outputs[key] = p_im

    if deb_path is None:
        return new_outputs


def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
                pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
        ) / camera_metadata.scale_factor_x
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    viewdirs = torch.from_numpy(viewdirs).to(z.device)

    coords = viewdirs[None] * z[..., None]
    
    # coords = coords.permute(0, 3, 1, 2)

    # dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    # dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    # dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    # dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    # dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    # dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    # dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
    # dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
    # dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

    # dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    # dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    # dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    # n_x = dydv * dzdu - dydu * dzdv
    # n_y = dzdv * dxdu - dzdu * dxdv
    # n_z = dxdv * dydu - dxdu * dydv

    # pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
    # pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)
    
    coords = coords.squeeze(0)
    hd, wd, _ = coords.shape
    bottom_point = coords[..., 2:hd, 1 : wd - 1, :]
    top_point = coords[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = coords[..., 1 : hd - 1, 2:wd, :]
    left_point = coords[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    pred_normal = torch.nn.functional.pad(xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant")
    return pred_normal[None]


def sw_cams(viewpoint_stack, cam_id, sw_size=2):
    viewpoint_cams_window = [viewpoint_stack[cam_id]]
    for sw in range(1, sw_size + 1):
        if cam_id - sw >= 0:
            viewpoint_cams_window.append(viewpoint_stack[cam_id - sw])
        if cam_id + sw < len(viewpoint_stack):
            viewpoint_cams_window.append(viewpoint_stack[cam_id + sw])
    return viewpoint_cams_window


def sw_depth_normalization(viewpoint_cams_window_list, depth_tensor, batch_size):
    for n_batch in range(batch_size):
        depth_window = []
        for viewpoint_cams_window in viewpoint_cams_window_list[n_batch]:
            depth_window.append(viewpoint_cams_window.depth[None].cuda())
        depth_window = torch.cat(depth_window, 0)
        depth_window_min = torch.min(depth_window).cuda()
        depth_window_max = torch.max(depth_window).cuda()
        depth_tensor[n_batch] = (depth_tensor[n_batch] - depth_window_min) / (depth_window_max - depth_window_min)
    return depth_tensor
