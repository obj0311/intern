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



def find_best_checkpoint(args, model_class, val_image_list):
    checkpoint_dir = os.path.dirname(args.pretrain)
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_lpips = float('inf')  # LPIPS는 낮을수록 좋음
    best_fsim = -float('inf')  # FSIM은 높을수록 좋음
    best_composite_score = -float('inf')
    
    #psnr 수치는 픽셀값이 살짝만 달라져도 급격히 떨어질 수 있으므로 성능 평가에 있어서 참고만 해야하는 지수
    # 1. PSNR (Peak Signal-to-Noise Ratio)
    # 20 dB 이하: 화질 저하가 눈에 띄는 수준, 복원 품질이 낮음

    # 20~30 dB: 중간 정도 품질, 노이즈가 보일 수 있음

    # 30 dB 이상: 일반적으로 "좋은" 복원 품질로 간주됨

    # 최신 SOTA 복원 논문(예: SIDD, CDD-11 등)에서는 PSNR 27~30 dB 이상이 흔함

    # 2. SSIM (Structural Similarity Index)
    # 0.9 이상: 구조적 유사도가 매우 높음, 시각적으로도 원본과 유사함

    # 0.8~0.9: 품질이 양호하지만, 미세한 차이 존재

    # 0.7 이하: 품질 저하가 명확하게 보임
    
    # 3. LPIPS (Learned Perceptual Image Patch Similarity)
    # LPIPS는 딥러닝 기반으로 사전 학습된 신경망(예: VGG, AlexNet)을 사용하여 두 이미지 간의 perceptual 유사성을 측정하는 지표입니다. 
    # 픽셀 단위 차이가 아닌 고수준(high-level) 특징 차이를 비교하므로, 인간의 시각적 판단과 더 잘 일치하는 결과를 제공합니다. 
    # 값이 낮을수록 두 이미지가 시각적으로 더 유사하다는 것을 의미합니다.

    # 0.0~0.1: 매우 유사한 이미지, 인간의 시각으로 거의 차이를 느낄 수 없는 수준. 복원 품질이 매우 높음.

    # 0.1~0.3: 품질이 양호하지만, 미세한 차이가 존재. 일반적으로 "좋은" 복원 품질로 간주됨.

    # 0.3~0.5: 중간 정도 품질, 차이가 눈에 띄기 시작함. 복원 품질이 다소 낮음.

    # 0.5 이상: 품질 저하가 명확하게 보임, 시각적으로 큰 차이가 있음.

    # 참고: 최신 SOTA 복원 논문이나 GAN, 스타일 전이 모델 평가에서 LPIPS는 종종 0.1~0.3 수준을 목표로 하며, 
    # 0.2 이하의 값은 대체로 우수한 복원 품질로 간주됩니다. LPIPS는 구조적 변화나 perceptual 차이를 민감하게 반영하므로, 
    # PSNR이나 SSIM과 달리 픽셀값의 작은 변화에 크게 영향을 받지 않습니다.

    # 4. FSIM (Feature Similarity Index Measure)
    # FSIM은 SSIM을 확장한 지표로, 이미지의 구조적 유사성뿐만 아니라 특징적 유사성(예: 경사 크기 및 방향 차이)을 고려하여 품질을 평가합니다. 
    # 인간 시각 시스템(HVS)에 더 가까운 결과를 제공하며, 값이 높을수록 두 이미지가 더 유사하다는 것을 의미합니다. 값은 0에서 1 사이로 정규화되어 있습니다.

    # 0.9 이상: 특징적 유사도가 매우 높음, 시각적으로 원본과 거의 동일한 수준. 복원 품질이 매우 높음.

    # 0.8~0.9: 품질이 양호하지만, 미세한 차이가 존재. 일반적으로 "좋은" 복원 품질로 간주됨.

    # 0.7~0.8: 중간 정도 품질, 차이가 눈에 띄기 시작함. 복원 품질이 다소 낮음.

    # 0.7 이하: 품질 저하가 명확하게 보임, 특징적 차이가 크게 나타남.

    # 참고: FSIM은 SSIM과 마찬가지로 정규화된 값을 제공하며, 인간의 시각적 인식과 saliency 기반 오류를 반영합니다. 
    # 연구에 따르면 FSIM은 종종 SSIM보다 인간 판단과 더 일치하는 결과를 제공하며, 최신 이미지 복원 연구에서는 0.85 이상의 값이 우수한 품질로 간주됩니다. 
    # FSIM은 픽셀 단위 절대 오류(MSE, PSNR)보다 구조적, 특징적 차이를 더 잘 반영합니다.

    lpips_loss_fn = LPIPS().to(args.device)
    psnr_lst = []
    ssim_lst = []
    lpips_lst = []
    fsim_lst = []
    composite_lst = []
    step_lst = []
    
    for ckpt_file in checkpoint_files:
        ckpt_path = checkpoint_dir + '/' + ckpt_file
        model = model_class(size=args.im_size, ngf=args.ngf, nz=args.nz, 
                          nc=args.nc, device=args.device).to(args.device)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['g'])
        model.eval()

    # 정규화 기준값 설정 (PSNR의 경우)
        PSNR_MIN = 20.0  # 최소 기준
        PSNR_MAX = 40.0  # 최대 기준    
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_fsim = 0
        count = 0
        
        
        step = ckpt_file.split('_')[1].split('.')[0]
        step_lst.append(int(step))
        
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
        
        psnr_lst.append(avg_psnr)
        ssim_lst.append(avg_ssim)
        lpips_lst.append(avg_lpips)
        fsim_lst.append(avg_fsim)
        
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_checkpoint_ssim = ckpt_path
            best_checkpoint_ssim_file = ckpt_file
            
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_checkpoint_psnr = ckpt_path
            best_checkpoint_psnr_file = ckpt_file
            
        if avg_lpips < best_lpips:  # LPIPS는 낮을수록 좋음
            best_lpips = avg_lpips
            best_checkpoint_lpips = ckpt_path
            best_checkpoint_lpips_file = ckpt_file
             
        if avg_fsim > best_fsim:
            best_fsim = avg_fsim
            best_checkpoint_fsim = ckpt_path
            best_checkpoint_fsim_file = ckpt_file
            
            
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
        
        composite_lst.append(composite_score)
        
        # 종합 점수가 가장 높은 모델 저장
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_checkpoint_composite = ckpt_path
            best_checkpoint_composite_file = ckpt_file
            
        print(f"Checkpoint: {ckpt_path} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FSIM: {avg_fsim:.4f} | Composite Score: {composite_score:.4f}")
        
        
    metrics = {'PSNR': psnr_lst, 'SSIM': ssim_lst, 'LPIPS': lpips_lst, 'FSIM': fsim_lst, 'COMPOSITE': composite_lst}
    for name, values in metrics.items():
        plt.figure()
        plt.plot(step_lst, values)
        plt.title(f'{name} Trend')
        plt.xlabel('Steps')
        plt.ylabel(name)
        plt.savefig(f'{name.lower()}_trend.png')
        plt.close()    

    
    
    return best_checkpoint_psnr, best_checkpoint_ssim, best_checkpoint_lpips, best_checkpoint_fsim, best_checkpoint_composite

def main(args):
    
    image_list = os.listdir(args.in_path)
    a,b,c,d,e = find_best_checkpoint(args, FullGenerator_t, image_list)


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