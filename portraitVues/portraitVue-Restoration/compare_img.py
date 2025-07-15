import torch
from torchvision import utils
import argparse
import os
from models import FullGenerator_t
import cv2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from operation import copy_G_params, load_params
from PIL import Image, ImageDraw, ImageFont
from piq import LPIPS
from piq import fsim 
import matplotlib.pyplot as plt
import re



def find_best_checkpoint(args):


    lpips_loss_fn = LPIPS().to(args.device)



        
    psnr_metric = PeakSignalNoiseRatio().to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
        
    image = cv2.imread('output_py.png')
    image_gt = cv2.imread('output_c.png')
    image = cv2.resize(image, (args.im_size, args.im_size))
    image_gt = cv2.resize(image_gt, (args.im_size, args.im_size))

    # Convert to tensor
    img_lq = torch.from_numpy(image).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_lq = (img_lq/255.-0.5)/0.5
    img_lq = torch.flip(img_lq, [1])
    img_gt = torch.from_numpy(image_gt).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_gt = (img_gt/255.-0.5)/0.5
    img_gt = torch.flip(img_gt, [1])

    psnr = psnr_metric(img_lq, img_gt).item()
    ssim = ssim_metric(img_lq, img_gt).item()
    
    
    # LPIPS 계산 (입력 범위 [-1, 1] 유지)
    lpips = lpips_loss_fn(img_lq, img_gt).item()
    
    # FSIM 계산 (입력 범위 [0, 1]로 변환)
    out_fsim = (img_lq + 1) / 2  # [-1, 1] -> [0, 1]
    img_gt_fsim = (img_gt + 1) / 2        # [-1, 1] -> [0, 1]
    fsim_f = fsim(out_fsim, img_gt_fsim, data_range=1.0).item()
                
            

                    
        
    metrics = {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lpips, 'FSIM': fsim_f}


    
    
    return metrics

def main(args):
    
    print(find_best_checkpoint(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    
    parser.add_argument("--in_path",        type=str,   default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/')
    parser.add_argument("--gt_path",        type=str,   default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/gt/' )
    parser.add_argument('--batch_size',     type=int,   default=1)
    parser.add_argument('--im_size',        type=int,   default=512)
    parser.add_argument('--device',         type=str,   default='cuda:0') #cuda:0 사용시 메모리 할당 오류(초과과) -> 모든 이미지를 하나의 배치 이미지로 묶어서 메모리 초과 문제가 발생한듯듯
    parser.add_argument('--ngf',            type=int,   default=64)
    parser.add_argument('--nz',             type=int,   default=256)
    parser.add_argument('--nc',             type=int,   default=3)
    parser.add_argument('--pretrain',       type=str,   default='D:/obj_data/portraitVue-Restoration-code/assets/ckpt/')
    parser.add_argument('--gen_script',     type=str,   default=True)

    args = parser.parse_args()
    
    
    main(args)