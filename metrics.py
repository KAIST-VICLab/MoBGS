import torch
import numpy as np
import os
import time
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2
from argparse import ArgumentParser
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)

def get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey, mask=None):
    target_OF = cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    output_OF = cv2.calcOpticalFlowFarneback(pre_output_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if mask is not None:
        mask, _, _ = crop_8x8(mask.squeeze())
    target_OF, ofy, ofx = crop_8x8(target_OF)
    output_OF, ofy, ofx = crop_8x8(output_OF)

    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm

    if mask is not None:
        return (OF_diff*mask).sum() / mask.sum()

    return OF_diff.mean()


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y + h, x:x + w]
    return crop_img, y, x

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8
    return np.sum(ssim_map * mask) / num_valid

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)

def evaluation(args):    
    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.

        count = 0.
        tofs = []
        pre_gt_grey, pre_output_grey = None, None
        for i in range(0, 24):

            # # ours
            pred_img_path = os.path.join(f'{args.output_dir}/{args.scene_name}/test_refined/img_{str(i).zfill(5)}.png.png')

            pred_img = cv2.imread(pred_img_path)[:, :, ::-1]
            pred_img = np.float32(pred_img) / 255

            gt_img_path = os.path.join(args.datadir, 'inference_images', '%05d.png'%i)
            gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
            gt_img = cv2.resize(gt_img, dsize=(pred_img.shape[1], pred_img.shape[0]), 
                                interpolation=cv2.INTER_AREA)
            gt_img = np.float32(gt_img) / 255
            
            ###### tOF ##########
            gt_grey = cv2.cvtColor((gt_img*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            output_grey = cv2.cvtColor((pred_img*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            if pre_gt_grey is not None:
                tOF = get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey)
            else:
                tOF = -1.0
            tofs.append(tOF)
            if i < 23:
                pre_gt_grey = gt_grey
                pre_output_grey = output_grey
            ###### tOF ##########

            ###### Full region metrics ######
            psnr = peak_signal_noise_ratio(gt_img, pred_img)
            ssim = structural_similarity(gt_img, pred_img, 
                                                multichannel=True)

            gt_img_0 = im2tensor(gt_img).cuda()
            rgb_0 = im2tensor(pred_img).cuda()

            lpips = model.forward(gt_img_0, rgb_0)
            lpips = lpips.item()

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            count += 1

        mean_psnr = total_psnr / count
        mean_ssim = total_ssim / count
        mean_lpips = total_lpips / count
        tofs = np.array(tofs)
        mean_tof = float(tofs[tofs >= 0.0].mean())


        print('mean_psnr ', mean_psnr)
        print('mean_ssim ', mean_ssim)
        print('mean_lpips ', mean_lpips)
        print('mean_tof ', mean_tof)




if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = ArgumentParser(description="Evaluation params")
    parser.add_argument("--datadir", type=str, required=True,
                        help='input data directory')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='output data directory')
    parser.add_argument("--scene_name", type=str,
                        help='scene name')
    args = parser.parse_args()
    evaluation(args)