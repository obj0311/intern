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
import json

#pth파일 하나 불러와서 성능 평가 지표 수치 계산한후 튜플로 출력
def find_best_checkpoint(args, pt_file_path, val_image_list):

    lpips_loss_fn = LPIPS().to(args.device)
    model = torch.jit.load(pt_file_path, map_location=args.device)

    # 정규화 기준값 설정 (PSNR의 경우)
    PSNR_MIN = 20.0  # 최소 기준
    PSNR_MAX = 40.0  # 최대 기준    
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_fsim = 0
    count = 0
    

    
    with torch.no_grad():
        psnr_metric = PeakSignalNoiseRatio().to(args.device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
        
        for image_name in val_image_list:
            # Your existing image loading and preprocessing logic
            image = cv2.imread(args.in_path + image_name)
            image_gt = cv2.imread(args.gt_path + image_name)
            image = cv2.resize(image, (args.im_size, args.im_size))
            image_gt = cv2.resize(image_gt, (args.im_size, args.im_size))

            # Convert to tensor
            img_lq = torch.from_numpy(image).to(args.device).permute(2, 0, 1).unsqueeze(0)
            img_lq = (img_lq/255.-0.5)/0.5
            img_lq = torch.flip(img_lq, [1])
            img_gt = torch.from_numpy(image_gt).to(args.device).permute(2, 0, 1).unsqueeze(0)
            img_gt = (img_gt/255.-0.5)/0.5
            img_gt = torch.flip(img_gt, [1])
            
            out = model(img_lq)
            total_psnr += psnr_metric(out, img_gt).item()
            total_ssim += ssim_metric(out, img_gt).item()
            count += 1
            
            # LPIPS 계산 (입력 범위 [-1, 1] 유지)
            total_lpips += lpips_loss_fn(out, img_gt).item()
            
                # FSIM 계산 (입력 범위 [0, 1]로 변환)
            out_fsim = (out + 1) / 2  # [-1, 1] -> [0, 1]
            img_gt_fsim = (img_gt + 1) / 2        # [-1, 1] -> [0, 1]
            total_fsim += fsim(out_fsim, img_gt_fsim, data_range=1.0).item()
            
    
    

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count
    avg_fsim = total_fsim / count
    
    # 종합 점수 계산
    # PSNR 정규화
    normalized_psnr = min(max((avg_psnr - PSNR_MIN) / (PSNR_MAX - PSNR_MIN), 0), 1)
    # SSIM은 이미 0~1 범위
    normalized_ssim = avg_ssim
    # LPIPS는 낮을수록 좋으므로 1에서 뺌 (0~1 범위 가정)
    normalized_lpips = 1 - min(max(avg_lpips, 0), 1)
    # FSIM은 이미 0~1 범위
    normalized_fsim = avg_fsim
    
    # 가중치 적용하여 종합 점수 계산
    composite_score = ( normalized_psnr + normalized_ssim + normalized_lpips + normalized_fsim)

    total = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips, 'fsim': avg_fsim, 'composite': composite_score}
    
    return total

def main(args):
    image_list = os.listdir(args.in_path)
    s3_perform = find_best_checkpoint(args,'s3.pt',image_list)
    train_perform = find_best_checkpoint(args, 'all_24000_newcomposite.pt',image_list)
    t_perform = find_best_checkpoint(args, "D:/obj_data/portraitVue-Restoration-code/(lr = 1e-6, batch = 2)/all_312000_newcomposite.pt", image_list)
    tt_perform = find_best_checkpoint(args, "D:/obj_data/portraitVue-Restoration-code/(lr = 1e-5, batch = 4)/all_476000_newcomposite.pt", image_list)
    performance_data={'s3.pt': s3_perform, '(lr = 1e-4, batch=4)composite.pt': train_perform, '(lr = 1e-5, batch=4)composite.pt': tt_perform, '(lr = 1e-6, batch=2)composite.pt': t_perform}
    with open('performance_contrast.json', 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=4)
        
    
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

    args = parser.parse_args()



    main(args)
    
    

    
    