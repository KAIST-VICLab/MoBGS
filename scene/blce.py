import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from torchdiffeq import odeint, odeint_adjoint
from scene.cameras import Camera
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getWorld2View2_torch
import math

def rgb_to_grayscale(image):
    """
    (3, H, W) 또는 (H, W, 3) RGB 이미지를 그레이스케일로 변환
    """
    if image.ndimension() == 3 and image.shape[-1] == 3:  # (H, W, 3)인 경우
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
    elif image.ndimension() == 3 and image.shape[0] == 3:  # (3, H, W)인 경우
        r, g, b = image[0], image[1], image[2]
    else:
        raise ValueError("Input image must be (H, W, 3) or (3, H, W)")

    grayscale = 0.299 * r + 0.587 * g + 0.114 * b  # 표준 Y 변환식
    return grayscale

def compute_frequency_blur_feature(image):
    """
    RGB 이미지를 입력받아 고주파 성분 비율로 블러 특성 추출
    Args:
        image (torch.Tensor): (3, H, W) 또는 (H, W, 3) 형태 RGB 이미지
    Returns:
        torch.Tensor: 고주파 성분 비율 (Blur Feature)
    """
    gray_image = rgb_to_grayscale(image)

    f = torch.fft.fft2(gray_image)
    fshift = torch.fft.fftshift(f)
    magnitude_spectrum = torch.abs(fshift)

    h, w = magnitude_spectrum.shape
    center_size = 20  # 중앙 저주파 영역 크기

    center = (slice(h//2 - center_size//2, h//2 + center_size//2),
              slice(w//2 - center_size//2, w//2 + center_size//2))

    total_energy = magnitude_spectrum.sum()
    low_freq_energy = magnitude_spectrum[center].sum()
    high_freq_energy = total_energy - low_freq_energy

    high_freq_ratio = high_freq_energy / total_energy
    return 1 - high_freq_ratio

def rotation_matrix_distance(R1, R2):
    """
    Computes the geodesic distance (angular distance) between two rotation matrices in SO(3).
    :param R1: (B, 3, 3) First batch of rotation matrices.
    :param R2: (B, 3, 3) Second batch of rotation matrices.
    :return: (B,) Rotation distance in radians.
    """
    trace_value = torch.einsum('bii->b', torch.matmul(R1.transpose(1, 2), R2))  # Trace of (R1^T R2)
    trace_value = torch.clamp((trace_value - 1) / 2, -1.0, 1.0)  # Clamp to avoid NaNs
    theta = torch.acos(trace_value)  # Convert to radians
    return theta

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

class blceKernel(nn.Module):
    def __init__(self, 
                 num_views: int = None,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 iteration: int = None,
                 ) -> None:
        super(blceKernel, self).__init__()

        self.num_warp = num_warp
        self.model = BLCE(num_views=num_views,
                          view_dim=view_dim,
                          num_warp=num_warp,
                          method=method,
                          adjoint=adjoint)
        
        l = []
        l.append({'params': list(self.model.get_params()),
                'lr': 1e-4, "name": "posenet"})
        l.append({'params': [self.model.exposure_time_expo], 'lr': 1e-1, "name": "exposure_time_expo"})

        self.optimizer = torch.optim.Adam(l, lr=1e-4)
        self.lr_factor = 0.01 ** (1 / iteration)
    
    def get_warped_cams(self,
                        cam : Camera = None,
                        fwd_cam : Camera = None,
                        bwd_cam : Camera = None,
                        ):
        
        Rt = self.get_Rt_c2w(cam)
        idx_view = cam.uid
        time = cam.time
        
        blur_feature = compute_frequency_blur_feature(cam.image.cuda())
        
        warped_Rt_c2w, exposure_time = self.model(Rt, blur_feature, idx_view)
        warped_Rt_w2c = torch.inverse(warped_Rt_c2w)
        warped_R = warped_Rt_c2w[:, :3, :3]
        warped_t = warped_Rt_w2c[:, :3, 3]
        warped_Rt = torch.cat([warped_R, warped_t[..., None]], dim=-1)
        warped_Rt = torch.cat([warped_Rt, torch.tensor([0, 0, 0, 1])[None].repeat(warped_Rt.size(0), 1, 1).to(warped_Rt)], dim=1)
        warped_cams = [self.get_cam(cam, warped_Rt[i]) for i in range(self.num_warp)]

        return warped_cams, exposure_time
    
    def get_cam(self,
                cam: Camera = None,
                Rt: torch.Tensor = None
                ) -> Camera:
        
        return Camera(colmap_id=cam.colmap_id,R=Rt[:3, :3], T=Rt[:3, 3],
                            FoVx=cam.FoVx, FoVy=cam.FoVy,
                            image=cam.image, gt_alpha_mask=cam.gt_alpha_mask, image_name=cam.image_name,uid=cam.uid,
                            data_device=cam.data_device,time=cam.time, mask=cam.mask, metadata=cam.metadata,
                            normal=cam.normal,
                            depth=cam.depth,
                            max_time=cam.max_time,
                            sem_mask=cam.sem_mask,
                            fwd_flow=cam.fwd_flow,
                            bwd_flow=cam.bwd_flow,
                            fwd_flow_mask=cam.fwd_flow_mask,
                            bwd_flow_mask=cam.bwd_flow_mask,
                            instance_mask=cam.instance_mask, 
                            tracklet=cam.tracklet,
                            query_tracks_2d = cam.query_tracks_2d,
                            target_ts = cam.target_ts,
                            target_tracks_2d = cam.target_tracks_2d,
                            target_visibles = cam.target_visibles,
                            target_invisibles = cam.target_invisibles,
                            target_confidences = cam.target_confidences,
                            target_track_depths = cam.target_track_depths,
                            target_ts_all = cam.target_ts_all,
                            target_tracks_2d_all = cam.target_tracks_2d_all,
                            target_visibles_all = cam.target_visibles_all,
                            target_invisibles_all = cam.target_invisibles_all,
                            target_confidences_all = cam.target_confidences_all,
                            target_track_depths_all = cam.target_track_depths_all,
                            K = cam.K,
                            covisible=cam.covisible,
                            depth_mask=cam.depth_mask,
                            sharp_img=cam.sharp_img,
                            )

    def get_Rt(self, 
               cam: Camera = None
               ) -> torch.Tensor:
        R, T = cam.R, cam.T # R:c2w, T:w2c
        Rt = np.concatenate([R, T[:, None]], axis=-1)
        Rt_fill = np.array([0, 0, 0, 1])[None]
        Rt = np.concatenate([Rt, Rt_fill], axis=0)
        Rt = torch.tensor(Rt, dtype=torch.float32).cuda()
        return Rt

    def get_Rt_c2w(self,
                   cam: Camera = None
                   ) -> torch.Tensor:
        if type(cam.R) == torch.Tensor:
            Rt_w2c = getWorld2View2_torch(cam.R, cam.T)
            Rt_w2c = Rt_w2c.detach().cpu().numpy()
        else:
            Rt_w2c = getWorld2View2(cam.R, cam.T)
        Rt_c2w = np.linalg.inv(Rt_w2c)
        Rt_c2w = torch.tensor(Rt_c2w, dtype=torch.float32).cuda()
        return Rt_c2w

    def get_weight_and_mask(self,
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        weight, mask = self.model.get_weight_and_mask(img, idx_view)
        return weight, mask

    def adjust_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_factor


class WV_Derivative(nn.Module):
    def __init__(self,
                 view_dim: int = 32,
                 num_views: int = 29,
                 num_warp: int = 5,
                 time_dim: int = 8,
                 blur_feat_dim: int = 32,
                 ) -> None:
        super(WV_Derivative, self).__init__()

        self.view_dim = view_dim
        self.num_views = num_views
        self.num_warp = num_warp
        self.blur_feature = None

        self.time_embedder = nn.Parameter(
            torch.zeros(num_warp, time_dim).type(torch.float32), 
            requires_grad=True
        )
        self.w_linear = nn.Linear(view_dim // 2 + time_dim + blur_feat_dim, view_dim // 2)
        self.v_linear = nn.Linear(view_dim // 2 + time_dim + blur_feat_dim, view_dim // 2)

        self.relu = nn.ReLU()

    def set_blur_feature(self, blur_feature: torch.Tensor):
        self.blur_feature = blur_feature

    def forward(self,
                t: int = 0,
                x: torch.Tensor = None,
                ) -> torch.Tensor:
        
        t_embed = self.time_embedder[int(t)]
        x = self.relu(x)

        w, v = torch.chunk(x, 2, dim=-1)

        w = torch.cat([w, t_embed, self.blur_feature], dim=-1)
        v = torch.cat([v, t_embed, self.blur_feature], dim=-1)

        w, v = self.w_linear(w), self.v_linear(v)

        return torch.cat([w, v], dim=-1)
    

class DiffEqSolver(nn.Module):
    def __init__(self, 
                 odefunc: nn.Module = None,
                 method: str = 'euler',
                 odeint_rtol: float = 1e-4,
                 odeint_atol: float = 1e-5,
                 num_warp: int = 5,
                 adjoint: bool = False,
                 ) -> None:
        super(DiffEqSolver, self).__init__()
        
        self.ode_func = odefunc
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.integration_time = torch.arange(0, num_warp, dtype=torch.long)
        self.solver = odeint_adjoint if adjoint else odeint
            
    def forward(self, 
                x: torch.Tensor = None,
                blur_feature: torch.Tensor = None,
                ) -> torch.Tensor:
        '''
        x                     : [ view_dim ]
        out                   : [ num_warp, view_dim ]
        '''
        self.integration_time = self.integration_time.type_as(x)
        if blur_feature is not None:
            self.ode_func.set_blur_feature(blur_feature)
        out = self.solver(self.ode_func, x, self.integration_time.cuda(x.get_device()),
                     rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        return out


class BLCE(nn.Module):
    def __init__(self,
                 num_views: int = 29,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 ) -> None:
        super(BLCE, self).__init__()

        self.num_warp = num_warp
        self.num_views = num_views

        self.view_embedder = nn.Parameter(
            torch.zeros(num_views, view_dim).type(torch.float32), 
            requires_grad=True
        )
        
        self.exposure_time_expo = nn.Parameter(
            torch.ones(num_views).type(torch.float32) * 0.4, 
            requires_grad=False
        ) 

        self.num_freqs = 10

        self.view_encoder = nn.ModuleList()
        self.Rt_encoder = nn.ModuleList()
        self.wv_derivative = nn.ModuleList()
        self.diffeq_solver = nn.ModuleList()
        self.rot_decoder = nn.ModuleList()
        self.trans_decoder = nn.ModuleList()
        self.theta_decoder = nn.ModuleList()
        self.blur_feature_encoder = nn.ModuleList()

        for i in range(num_views):
            self.blur_feature_encoder.append(nn.Sequential(
            nn.Linear(2 * self.num_freqs + 1, view_dim),
            nn.ReLU(),
            nn.Linear(view_dim, view_dim),
            nn.ReLU(),  
            nn.Linear(view_dim, view_dim),
            ))
            self.Rt_encoder.append(nn.Linear(12, view_dim))
            self.view_encoder.append(nn.Linear(view_dim * 2, view_dim))
            self.wv_derivative.append(WV_Derivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp))
            self.diffeq_solver.append(DiffEqSolver(odefunc=self.wv_derivative[i], method=method, num_warp=num_warp, adjoint=adjoint))
            self.rot_decoder.append(nn.Linear(view_dim // 2, 3))
            self.trans_decoder.append(nn.Linear(view_dim // 2, 3))
            self.theta_decoder.append(nn.Linear(view_dim // 2, 1))

            gain = 0.00001 / (math.sqrt((view_dim // 2 + 3) / 6))
            torch.nn.init.xavier_uniform_(self.rot_decoder[i].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.trans_decoder[i].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.theta_decoder[i].weight, gain=gain)
            self.rot_decoder[i].bias.data.fill_(0)
            self.trans_decoder[i].bias.data.fill_(0)
            self.theta_decoder[i].bias.data.fill_(0)


    def update_exposure_time(self,idx_view, value):
        self.exposure_time_expo[idx_view] = value
        
    def forward(self,
                Rt: torch.Tensor = None,
                blur_feature: float = None,
                idx_view: int = None,
                ) -> torch.Tensor:
        
        '''
        new_Rt, w_loss, kernel_weights, kernel_mask = kernelnet(Rt, viewpoint_cam.uid, image.detach(), depth.detach())
        Input:
            Rt           : [4, 4]
            idx_view     : scalar
            image        : [3, 400, 600]
            depth        : [1, 400, 600]
        Output:
            new_Rt        : [num_warp, 4, 4]
            w_loss        : scalar
            kernel_weights: [num_warp, 1, 400, 600]
            kernel_mask   : [1, 400, 600]
        '''

        freqs = (2 ** torch.arange(self.num_freqs)).cuda()
        angles = blur_feature * freqs * np.pi
        blur_feature_embed = torch.cat([blur_feature.unsqueeze(0), torch.sin(angles), torch.cos(angles)], dim=-1).to(Rt.device)
        blur_feature_embed = self.blur_feature_encoder[idx_view](blur_feature_embed)

        view_embed = self.view_embedder[idx_view]
        Rt_encoded = self.Rt_encoder[idx_view](Rt[:3, :].reshape(-1))
        view_embed = torch.cat([view_embed, Rt_encoded], dim=-1)
        view_encoded = self.view_encoder[idx_view](view_embed)
        latent_wv = self.diffeq_solver[idx_view](view_encoded, blur_feature_embed)

        latent_w, latent_v = torch.chunk(latent_wv, 2, dim=-1)
        w_rigid = self.rot_decoder[idx_view](latent_w)
        theta = self.theta_decoder[idx_view](latent_w)[..., None]

        v_rigid = self.trans_decoder[idx_view](latent_v)

        w_norm, _ = self.exp_map(w_rigid)
        w_skew = self.skew_symmetric(w_norm)
        R_exp = self.rodrigues_formula(w_skew, theta)
        G = self.G_formula(w_skew, theta)
        p = torch.matmul(G, v_rigid[..., None])
        Rt_rigid = self.transform_SE3(R_exp, p)
        
        Rt_transform = Rt_rigid
        Rt_new = torch.einsum('ij, tjk -> tik', Rt, Rt_transform)

        
        exposure_time = torch.linspace(-1, 1, self.num_warp).to(self.exposure_time_expo[idx_view].device) * self.exposure_time_expo[idx_view]

        return Rt_new, exposure_time
    
    def get_params(self):
        exclude_params = ['exposure_time_expo']
        filtered_params = [param for name, param in self.named_parameters() if name not in exclude_params]
        filtered_params_generator = (param for param in filtered_params)
        return filtered_params_generator
    
    def transform_SE3(self, 
                      exp_w_skew: torch.Tensor, 
                      p: torch.Tensor
                      ) -> torch.Tensor:
        
        delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
        delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
        delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
        return delta_Rt
    
    def rodrigues_formula(self, 
                          w: torch.Tensor, 
                          theta: torch.Tensor,
                          ) -> torch.Tensor:
        
        term1 = torch.eye(3).to(w)
        term2 = torch.sin(theta) * w
        term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def G_formula(self,
                  w: torch.Tensor, 
                  theta: torch.Tensor,
                  ) -> torch.Tensor:
        term1 = torch.eye(3)[None].to(w) * theta
        term2 = (1 - torch.cos(theta)) * w
        term3 = (theta - torch.sin(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3

    def exp_map(self, 
                w: torch.Tensor,
                ) -> torch.Tensor:
        norm = torch.norm(w, dim=-1)[..., None] + 1e-10
        w = w / norm
        return w, norm[..., None]

    def skew_symmetric(self, 
                       w : torch.Tensor,
                       ) -> torch.Tensor:
        
        w1, w2, w3 = torch.chunk(w, 3, dim=-1)

        w_skew =  torch.cat([torch.zeros_like(w1), -w3, w2,
                             w3, torch.zeros_like(w1), -w1,
                             -w2, w1, torch.zeros_like(w1)], dim=-1)
        w_skew = w_skew.reshape(-1, 3, 3)
        return w_skew
    
