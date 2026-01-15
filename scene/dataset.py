from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = PILtoTorch(caminfo.image, (caminfo.image.width, caminfo.image.height))
                if caminfo.sharp_img is not None:
                    sharp_img = PILtoTorch(caminfo.sharp_img,(caminfo.image.width, caminfo.image.height))
                else:
                    sharp_img = None
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

            return Camera(colmap_id=index,R=R,T=T,
                            FoVx=FovX,FoVy=FovY,
                            image=image,gt_alpha_mask=None,
                            image_name=f"{caminfo.image_name}",uid=index,
                            data_device=torch.device("cuda"),time=time, mask=mask, metadata=caminfo.metadata,
                            normal=caminfo.normal,
                            depth=caminfo.depth,
                            max_time=caminfo.max_time,
                            sem_mask=caminfo.sem_mask,
                            fwd_flow=caminfo.fwd_flow,
                            bwd_flow=caminfo.bwd_flow,
                            fwd_flow_mask=caminfo.fwd_flow_mask,
                            bwd_flow_mask=caminfo.bwd_flow_mask,
                            instance_mask=caminfo.instance_mask, 
                            tracklet=caminfo.tracklet,
                            query_tracks_2d = caminfo.query_tracks_2d,
                            target_ts = caminfo.target_ts,
                            target_tracks_2d = caminfo.target_tracks_2d,
                            target_visibles = caminfo.target_visibles,
                            target_invisibles = caminfo.target_invisibles,
                            target_confidences = caminfo.target_confidences,
                            target_track_depths = caminfo.target_track_depths,
                            target_ts_all = caminfo.target_ts_all,
                            target_tracks_2d_all = caminfo.target_tracks_2d_all,
                            target_visibles_all = caminfo.target_visibles_all,
                            target_invisibles_all = caminfo.target_invisibles_all,
                            target_confidences_all = caminfo.target_confidences_all,
                            target_track_depths_all = caminfo.target_track_depths_all,
                            K = caminfo.K,
                            covisible=caminfo.covisible,
                            depth_mask=caminfo.depth_mask,
                            sharp_img=sharp_img,
                            )
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
