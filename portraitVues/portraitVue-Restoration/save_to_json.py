import json
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from piq import LPIPS, fsim
import torch
from torchvision import utils
import argparse
import os
from models import FullGenerator_t
import cv2
import numpy as np
from operation import copy_G_params, load_params
from PIL import Image, ImageDraw, ImageFont
from piq import LPIPS
from piq import fsim 
import re


# 특정 pth 파일 하나 불러와서 psnr, ssim 등 여러 성능 평가 지표 수치 측정하고 이를 json 파일로 저장하는 코드드



# 2. 모델 로드
pth_path = "D:/obj_data/portraitVue-Restoration-code/assets/ckpt/all_24000.pth"
model = FullGenerator_t(size=512, ngf=64, nz=256, nc=3, device='cuda:0').to('cuda:0')

ckpt = torch.load(pth_path)
model.load_state_dict(ckpt['g'])
model.eval()
image_list = os.listdir('D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/')
lpips_loss_fn = LPIPS().to('cuda:0')

total_psnr = 0
total_ssim = 0
total_lpips = 0
total_fsim = 0
count = 0


with torch.no_grad():
            psnr_metric = PeakSignalNoiseRatio().to('cuda:0')
            ssim_metric = StructuralSimilarityIndexMeasure().to('cuda:0')
            
            for image_name in image_list:
                # Your existing image loading and preprocessing logic
                image = cv2.imread('D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/' + image_name)
                image_gt = cv2.imread('D:/obj_data/portraitVue-Restoration/1_Final/auto_test/gt/' + image_name)
                image = cv2.resize(image, (512,512))
                image_gt = cv2.resize(image_gt, (512,512))

                # Convert to tensor
                img_lq = torch.from_numpy(image).to("cuda:0").permute(2, 0, 1).unsqueeze(0)
                img_lq = (img_lq/255.-0.5)/0.5
                img_lq = torch.flip(img_lq, [1])
                img_gt = torch.from_numpy(image_gt).to('cuda:0').permute(2, 0, 1).unsqueeze(0)
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
normalized_psnr = min(max((avg_psnr - 20) / 20, 0), 1)
# SSIM은 이미 0~1 범위
normalized_ssim = avg_ssim
# LPIPS는 낮을수록 좋으므로 1에서 뺌 (0~1 범위 가정)
normalized_lpips = 1 - min(max(avg_lpips, 0), 1)
# FSIM은 이미 0~1 범위
normalized_fsim = avg_fsim

# 가중치 적용하여 종합 점수 계산
composite_score = normalized_psnr + normalized_ssim + normalized_lpips + normalized_fsim


# 6. 결과 저장
results = {
    'PSNR': avg_psnr,
    'SSIM': avg_ssim,
    'LPIPS': avg_lpips,
    'FSIM': avg_fsim,
    'composite': composite_score
}
json_path = 'all_24000_g.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=5)

print('Saved metrics to', json_path)
