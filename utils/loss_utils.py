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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
from pytorch3d.ops import ball_query
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pytorch3d.transforms import matrix_to_quaternion

def quaternion_distance(q1, q2):
    """
    Computes the geodesic distance between two unit quaternions.
    :param q1: (B, 4) First batch of quaternions.
    :param q2: (B, 4) Second batch of quaternions.
    :return: (B,) Rotation distance in radians.
    """
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    dot_product = torch.sum(q1 * q2, dim=-1).abs()  # Ensure shortest path
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp to avoid NaNs
    theta = 2 * torch.acos(dot_product)  # Convert to radians
    return theta

def quaternion_slerp(q1, q2, t):
    """
    두 개의 quaternion을 SLERP 방식으로 보간.

    입력:
        q1: (4,) 첫 번째 quaternion
        q2: (4,) 두 번째 quaternion
        t: 보간 비율 (0 ~ 1)

    출력:
        보간된 quaternion (4,)
    """
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = torch.sum(q1 * q2, dim=-1)  # 내적 계산
    if dot < 0.0:  # 반대 방향을 방지하기 위해 부호 반전
        q2 = -q2
        dot = -dot

    dot = torch.clamp(dot, -1.0, 1.0)  # acos 안정성을 위해 클램핑
    theta = torch.acos(dot)  # 두 quaternion 간의 각도
    sin_theta = torch.sin(theta)

    # sin(θ) 값이 너무 작을 경우 (두 quaternion이 거의 같을 경우) 선형 보간 수행
    if sin_theta < 1e-6:
        return (1 - t) * q1 + t * q2

    coeff1 = torch.sin((1 - t) * theta) / sin_theta
    coeff2 = torch.sin(t * theta) / sin_theta
    return coeff1 * q1 + coeff2 * q2


def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def compute_tv_loss(pred):
    """
    Args:
        pred: [batch, H, W, 3]

    Returns:
        tv_loss: [batch]
    """
    h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
    w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

## som losses
def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        channel = gt.shape[1]
        mask = mask.expand(-1, channel, -1, -1)
        return torch.abs((network_output - gt) * mask).sum() / (mask.sum() + 1e-8)
    else:
        return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt, mask=None):
    if mask is not None:
        channel = gt.shape[1]
        mask = mask.expand(-1, channel, -1, -1)
        return torch.square((network_output - gt) * mask).sum() / (mask.sum() + 1e-8)
    else:
        return torch.square((network_output - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window



def entropy_loss(alpha):
    """
    Entropy loss to encourage alpha values to be either 0 or 1.
    Lower entropy means alpha is more certain (closer to 0 or 1).
    
    Args:
        alpha (torch.Tensor): Tensor of alpha values (N,)
        
    Returns:
        torch.Tensor: Entropy loss value
    """
    epsilon = 1e-6  # Small value to avoid log(0)
    return -torch.sum(alpha * torch.log(alpha + epsilon) + (1 - alpha) * torch.log(1 - alpha + epsilon))

def entropy_loss_logit(alpha):
    """Entropy loss with logit transformation to improve gradient flow."""
    epsilon = 1e-6  # Avoid log(0)
    logit_alpha = torch.log(alpha + epsilon) - torch.log(1 - alpha + epsilon)  # Logit transformation
    return -torch.sum(torch.sigmoid(logit_alpha) * logit_alpha)


def sparsity_loss(alpha):
    """
    Sparsity loss to encourage minimal alpha values in non-motion areas.
    
    Args:
        alpha (torch.Tensor): Tensor of alpha values (N,)
        
    Returns:
        torch.Tensor: Sparsity loss value
    """
    return torch.sum(alpha**2)  # L2 regularization on alpha

def sparsity_loss_boost(alpha):
    """Encourages α to increase slightly in the early stages of training."""
    return torch.sum((alpha - 0.1)**2)  # Push α to move toward 0.1 instead of 0


def motion_consistency_loss(positions_t, positions_t_prev, alpha):
    """
    Motion consistency loss to penalize Gaussian splats with low motion but high alpha.
    
    Args:
        positions_t (torch.Tensor): Current Gaussian positions (N, 3)
        positions_t_prev (torch.Tensor): Previous frame Gaussian positions (N, 3)
        alpha (torch.Tensor): Alpha values (N,)
        
    Returns:
        torch.Tensor: Motion consistency loss value
    """
    motion = torch.norm(positions_t - positions_t_prev, dim=1)  # Euclidean distance
    return torch.sum(alpha * motion)  # Reduce alpha for low-motion Gaussians


class SSIM_tensor(nn.Module):
    """Layer to compute the SSIM loss between a pair of images, returns non-reduced tensor error
    """
    def __init__(self):
        super(SSIM_tensor, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class BinaryDiceLoss(nn.Module):

    def __init__(
            self,
            batch_dice: bool = False,
            from_logits: bool = True,
            log_loss: bool = False,
            smooth: float = 0.0,
            eps: float = 1e-7,
    ):
        """Implementation of Dice loss for binary image segmentation tasks

        Args:
            batch_dice: dice per sample and average or treat batch as a single volumetric sample (default)
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(BinaryDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)

        y_true = y_true.view(bs, -1)  # bs x num_elems
        y_pred = y_pred.view(bs, -1)  # bs x num_elems

        if self.batch_dice == True:
            intersection = torch.sum(y_pred * y_true)  # float
            cardinality = torch.sum(y_pred + y_true)  # float
        else:
            intersection = torch.sum(y_pred * y_true, dim=-1)  # bs x float
            cardinality = torch.sum(y_pred + y_true, dim=-1)  # bs x float

        dice_scores = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        if self.log_loss:
            losses = -torch.log(dice_scores.clamp_min(self.eps))
        else:
            losses = 1.0 - dice_scores
        return losses.mean()
    
def sgt_smoothness(dyn_pc, time, fwd_time, bwd_time):
    
    pointtimes = torch.ones((dyn_pc.get_xyz.shape[0],1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    
    # Calculate Polynomial trajectory    
    basicfunction = trbfunction
    trbfcenter = dyn_pc.get_trbfcenter
    trbfscale = dyn_pc.get_trbfscale
    trbfdistanceoffset = time * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    trbfdistanceoffset_prev = bwd_time * pointtimes - trbfcenter
    trbfdistance_prev =  trbfdistanceoffset_prev / torch.exp(trbfscale) 
    trbfoutput_prev = basicfunction(trbfdistance_prev)
    
    trbfdistanceoffset_next = fwd_time * pointtimes - trbfcenter
    trbfdistance_next =  trbfdistanceoffset_next / torch.exp(trbfscale) 
    trbfoutput_next = basicfunction(trbfdistance_next)
    
    return 0


# class KnnConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, neighborhood_size=4, relax_size=16):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size
#         self.relax_size = relax_size

#     def forward(self, xyz, rand_xyz):
#         idx = pointops.knn(xyz, xyz, self.neighborhood_size)[0]
#         with torch.no_grad():
#             max_rand_dist = pointops.knn(rand_xyz, rand_xyz, self.neighborhood_size + self.relax_size)[1][..., -1:]
#             rand_neighborhood = pointops.index_points(rand_xyz, idx)

#         rand_neighborhood_dist = (rand_xyz[...,None,:] - rand_neighborhood).norm(dim=-1)

#         rand_dist_diff = torch.clamp((rand_neighborhood_dist - max_rand_dist), min=0)
#         return rand_dist_diff

# class KnnConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, neighborhood_size=4, relax_size=16):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size
#         self.relax_size = relax_size

#     def forward(self, xyz, rand_xyz):
#         with torch.no_grad():
#             idx, dist = pointops.knn(xyz, xyz, self.neighborhood_size)
#             rand_neighborhood = pointops.index_points(rand_xyz, idx)

#         rand_neighborhood_dist = (rand_xyz[...,None,:] - rand_neighborhood).norm(dim=-1)
#         rand_dist_diff = torch.square(dist - rand_neighborhood_dist)
#         return rand_dist_diff

class KnnConstraint(nn.Module):
    """
    The Normal Consistency Constraint
    """
    def __init__(self, neighborhood_size=20):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.temperature = 0.1


    def forward(self, xyz, canno_xyz, radius):
        batch_size, nsample, _ = xyz.shape
        neighbor_inds = ball_query(xyz, xyz, K=self.neighborhood_size, radius=radius)[1][..., 1:] # remove first element
        neighbor_inds_mask = neighbor_inds != -1

        neighbor_inds[~neighbor_inds_mask] = 0
        neighbor_inds = neighbor_inds.reshape(batch_size, -1).long()

        # get the neighborhood points
        neighborhood = torch.gather(
            xyz, 1, neighbor_inds[:, :, None].expand(-1, -1, 3)
        ).reshape(
            batch_size, nsample, self.neighborhood_size - 1, 3
        ) # B, N, K, 3
        current_dist = (xyz[..., None, :] - neighborhood).norm(dim=-1) # B, N, K
    
        # get cannocal neighborhood points
        canno_neighborhood = torch.gather(
            canno_xyz[None].expand(batch_size,-1,-1), 1, neighbor_inds[:, :, None].expand(-1, -1, 3)
        ).reshape(
            batch_size, nsample, self.neighborhood_size - 1, 3
        )
        canno_dist = (canno_xyz[..., None, :] - canno_neighborhood).norm(dim=-1).detach() # B, N, K

        weight = torch.exp(-torch.square(canno_dist)*self.temperature).detach()
        weight[~neighbor_inds_mask] = 0

        return weighted_l2_loss_v1(current_dist, canno_dist, weight)
    

def compute_cluster_cohesion_loss(means3D, labels, centroids):
    """
    같은 클러스터 내 Gaussian들의 Mean 값이 클러스터 중심과 가까워지도록 하는 Loss 계산.
    - means3D: (N, 3) 각 Gaussian의 Mean 위치
    - labels: (N,) 각 Gaussian이 속한 클러스터 ID
    - centroids: (num_clusters, 3) 각 클러스터의 평균 위치
    Returns:
    - loss_cluster_cohesion: 클러스터 내 Gaussian들의 밀집도를 증가시키는 Loss
    """
    num_clusters = centroids.shape[0]  # 클러스터 개수

    # ✅ 클러스터별 Loss를 계산할 변수
    loss = torch.zeros(num_clusters, device=means3D.device)

    # ✅ 각 클러스터별로 Gaussian들의 Mean 값을 가져와서 Loss 계산
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id  # 현재 클러스터에 속한 Gaussian들의 인덱스 (Boolean Mask)
        if mask.sum() > 0:  # 해당 클러스터에 Gaussian이 존재하는 경우만 처리
            means_clustered = means3D[mask]  # 현재 클러스터의 Gaussian 위치들 (subset)
            cluster_center = centroids[cluster_id]  # 해당 클러스터의 중심 좌표

            # ✅ L2 Distance (MSE) Loss 계산
            loss[cluster_id] = torch.mean((means_clustered - cluster_center) ** 2)

    # ✅ 전체 클러스터의 평균 Loss 계산
    loss_cluster_cohesion = torch.mean(loss)  # 클러스터별 Loss의 평균을 최종 Loss로 사용

    return loss_cluster_cohesion


def path_distance_loss_separate(ref_Rs, warped_Rs, num_samples=100):
    """
    `warped` 카메라가 `R1 → R2` 및 `R2 → R3` 경로를 독립적으로 따르도록 하는 Loss.
    단, `R1 == R2` 또는 `R2 == R3`일 경우 해당 경로에 대한 Loss는 제외.

    입력:
        ref_Rs: (3, 3, 3) → 인접한 3개 카메라 회전 행렬 (R1, R2, R3)
        warped_Rs: (N, 3, 3) → N개의 `warped` 카메라 회전 행렬
        num_samples: 경로를 몇 개의 지점으로 샘플링할지 (default=100)

    출력:
        loss: warped 카메라가 두 경로를 따르도록 하는 Loss
    """
    # ✅ 회전 행렬을 Quaternion으로 변환
    ref_quats = matrix_to_quaternion(ref_Rs)  # (3, 4)
    warped_quats = matrix_to_quaternion(warped_Rs)  # (N, 4)

    device = ref_quats.device

    # ✅ SLERP 경로 샘플링 (t=0과 t=1 제외)
    t_vals = torch.linspace(0, 1, num_samples, device=device)[1:-1]  # t=0, t=1 제거

    # ✅ `R1 → R2` 및 `R2 → R3`이 같은 경우 확인
    is_r1_r2_same = torch.allclose(ref_quats[0], ref_quats[1], atol=1e-6)
    is_r2_r3_same = torch.allclose(ref_quats[1], ref_quats[2], atol=1e-6)

    # ✅ `R1 → R2` SLERP 경로 생성 (같지 않은 경우만)
    if not is_r1_r2_same:
        path_12_quats = torch.stack([quaternion_slerp(ref_quats[0], ref_quats[1], t) for t in t_vals], dim=0)  # (num_samples-2, 4)

    # ✅ `R2 → R3` SLERP 경로 생성 (같지 않은 경우만)
    if not is_r2_r3_same:
        path_23_quats = torch.stack([quaternion_slerp(ref_quats[1], ref_quats[2], t) for t in t_vals], dim=0)  # (num_samples-2, 4)

    # ✅ 각 `warped` 카메라가 두 경로에서 가장 가까운 지점에 대해 Loss 계산
    loss = 0
    for warped_q in warped_quats:
        min_distance = None

        # ✅ `R1 → R2` 경로에 대한 최소 거리 계산 (적용 가능할 경우)
        if not is_r1_r2_same:
            distances_12 = quaternion_distance(warped_q.unsqueeze(0), path_12_quats)  # (num_samples-2,)
            min_distance_12 = torch.min(distances_12)
            min_distance = min_distance_12 if min_distance is None else torch.min(torch.stack([min_distance, min_distance_12]))

        # ✅ `R2 → R3` 경로에 대한 최소 거리 계산 (적용 가능할 경우)
        if not is_r2_r3_same:
            distances_23 = quaternion_distance(warped_q.unsqueeze(0), path_23_quats)  # (num_samples-2,)
            min_distance_23 = torch.min(distances_23)
            min_distance = min_distance_23 if min_distance is None else torch.min(torch.stack([min_distance, min_distance_23]))

        loss += min_distance  # 가장 작은 거리만 Loss에 포함

    return loss / len(warped_quats)  # 평균 Loss 반환


def path_distance_loss_rotation(ref_Rs, warped_Rs, num_samples=50):
    """
    `warped_Rs`가 `R1 → R2`, `R2 → R3` 회전 경로를 따라가도록 유도하는 Loss.
    두 개의 SLERP 곡선을 50개씩 샘플링하여 더 정밀한 경로를 생성.
    
    입력:
        ref_Rs: (3, 3, 3) → 인접한 3개 카메라 회전 행렬 (R1, R2, R3)
        warped_Rs: (N, 3, 3) → N개의 `warped` 카메라 회전 행렬
        num_samples: 각 구간의 샘플 개수 (default=50)

    출력:
        loss: warped_Rs가 `R1 → R2 → R3` 회전 경로를 따르도록 하는 Loss
    """
    device = ref_Rs.device

    # ✅ 기준 벡터 [1,0,0]을 R1, R2, R3에 변환하여 단위구(S²) 위의 호를 정의
    base_vec = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(-1)  # (3,1)
    base_vec = base_vec / torch.norm(base_vec)
    
    v1 = torch.matmul(ref_Rs[0], base_vec).squeeze(-1)  # R1 변환 벡터
    v2 = torch.matmul(ref_Rs[1], base_vec).squeeze(-1)  # R2 변환 벡터
    v3 = torch.matmul(ref_Rs[2], base_vec).squeeze(-1)  # R3 변환 벡터

    # ✅ `warped_Rs`도 `[1,0,0]`을 변환하여 비교
    warped_points = torch.matmul(warped_Rs, base_vec).squeeze(-1)  # (N, 3)

    loss = 0

    # ✅ R1과 R2가 같으면 `R1 → R2` 경로를 만들 필요 없음
    if not torch.allclose(v1, v2, atol=1e-6):
        theta_12 = torch.acos(torch.clamp(torch.dot(v1, v2), -1.0, 1.0))  # 두 벡터의 각도
        t_vals_12 = torch.linspace(0, 1, num_samples, device=device)  # t ∈ [0,1]

        arc_12_samples = torch.stack([
            (torch.sin((1 - t) * theta_12) / torch.sin(theta_12)) * v1 +
            (torch.sin(t * theta_12) / torch.sin(theta_12)) * v2
            for t in t_vals_12
        ], dim=0)  # (num_samples, 3)

        for warped_p in warped_points:
            distances_12 = torch.norm(warped_p.unsqueeze(0) - arc_12_samples, dim=-1)  # (num_samples,)
            min_distance_12 = torch.min(distances_12)  # 가장 가까운 거리
            loss += min_distance_12

    # ✅ R2와 R3가 같으면 `R2 → R3` 경로를 만들 필요 없음
    if not torch.allclose(v2, v3, atol=1e-6):
        theta_23 = torch.acos(torch.clamp(torch.dot(v2, v3), -1.0, 1.0))  # 두 벡터의 각도
        t_vals_23 = torch.linspace(0, 1, num_samples, device=device)  # t ∈ [0,1]

        arc_23_samples = torch.stack([
            (torch.sin((1 - t) * theta_23) / torch.sin(theta_23)) * v2 +
            (torch.sin(t * theta_23) / torch.sin(theta_23)) * v3
            for t in t_vals_23
        ], dim=0)  # (num_samples, 3)

        for warped_p in warped_points:
            distances_23 = torch.norm(warped_p.unsqueeze(0) - arc_23_samples, dim=-1)  # (num_samples,)
            min_distance_23 = torch.min(distances_23)  # 가장 가까운 거리
            loss += min_distance_23

    return loss / len(warped_Rs)  # 평균 Loss 반환

def path_distance_loss_translation(ref_Ts, warped_Ts, num_samples=50):
    """
    `warped_Ts`가 `T1 → T2`, `T2 → T3` 이동 경로를 따라가도록 하는 Loss.
    각 구간을 50개씩 샘플링하여 더 정밀한 이동 경로를 생성.

    입력:
        ref_Ts: (3, 3) → 인접한 3개 카메라 위치 (T1, T2, T3)
        warped_Ts: (N, 3) → N개의 `warped` 카메라 위치
        num_samples: 각 구간의 샘플 개수 (default=50)

    출력:
        loss: warped_Ts가 `T1 → T2 → T3` 이동 경로를 따르도록 하는 Loss
    """
    device = ref_Ts.device

    T1, T2, T3 = ref_Ts  # (3,) 벡터

    loss = 0

    # ✅ T1과 T2가 같으면 `T1 → T2` 경로를 만들 필요 없음
    if not torch.allclose(T1, T2, atol=1e-6):
        t_vals_12 = torch.linspace(0, 1, num_samples, device=device).unsqueeze(-1)  # (num_samples, 1)
        trans_12_samples = (1 - t_vals_12) * T1 + t_vals_12 * T2  # (num_samples, 3)

        for warped_T in warped_Ts:
            distances_12 = torch.norm(warped_T.unsqueeze(0) - trans_12_samples, dim=-1)  # (num_samples,)
            min_distance_12 = torch.min(distances_12)  # 가장 가까운 거리
            loss += min_distance_12

    # ✅ T2와 T3가 같으면 `T2 → T3` 경로를 만들 필요 없음
    if not torch.allclose(T2, T3, atol=1e-6):
        t_vals_23 = torch.linspace(0, 1, num_samples, device=device).unsqueeze(-1)  # (num_samples, 1)
        trans_23_samples = (1 - t_vals_23) * T2 + t_vals_23 * T3  # (num_samples, 3)

        for warped_T in warped_Ts:
            distances_23 = torch.norm(warped_T.unsqueeze(0) - trans_23_samples, dim=-1)  # (num_samples,)
            min_distance_23 = torch.min(distances_23)  # 가장 가까운 거리
            loss += min_distance_23

    return loss / len(warped_Ts)  # 평균 Loss 반환