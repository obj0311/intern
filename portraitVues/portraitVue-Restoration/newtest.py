import os
import cv2
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from piq import fsim, LPIPS
import numpy as np
from models import FullGenerator_t


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """모델 파일을 로드하는 함수"""
    try:
        model_data = torch.load(model_path, map_location=device)
        if isinstance(model_data, dict) and 'g' in model_data:
            # 상태 딕셔너리인 경우
            model = FullGenerator_t(size=512, ngf=64, nz=100, nc=3, device=device).to(device)  # 모델 클래스와 파라미터는 실제 코드에 맞게 조정
            model.load_state_dict(model_data['g'])
        elif isinstance(model_data, dict):
            model = FullGenerator_t(size=512, ngf=64, nz=100, nc=3, device=device).to(device)
            model.load_state_dict(model_data)
        else:
            model = model_data.to(device)
        model.eval()
        print(f"Loaded {model_path} successfully.")
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def evaluate_model(model, image_list, in_path, gt_path, im_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """모델 성능을 평가하는 함수"""
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_fsim = 0.0
    count = len(image_list)
    
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 메트릭 초기화
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    lpips_metric = LPIPS().to(device)
    
    with torch.no_grad():
        for image_name in image_list:
            image = cv2.imread(os.path.join(in_path, image_name))
            image_gt = cv2.imread(os.path.join(gt_path, image_name))
            
            if image is None or image_gt is None:
                print(f"Skipping {image_name}: Image not found.")
                continue
                
            image = cv2.resize(image, (im_size, im_size))
            image_gt = cv2.resize(image_gt, (im_size, im_size))
            
            # 이미지 전처리 (모델 입력 형식에 맞게)
            img_lq = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0).float()
            img_lq = (img_lq / 255.0 - 0.5) / 0.5
            img_lq = torch.flip(img_lq, [1])
            
            img_gt = torch.from_numpy(image_gt).to(device).permute(2, 0, 1).unsqueeze(0).float()
            img_gt = (img_gt / 255.0 - 0.5) / 0.5
            img_gt = torch.flip(img_gt, [1])
            
            # 모델 출력
            batch_out = model(img_lq)
            
            # 메트릭 계산
            total_psnr += psnr_metric(batch_out, img_gt).item()
            total_ssim += ssim_metric(batch_out, img_gt).item()
            total_lpips += lpips_metric(batch_out, img_gt).item()
            
            # FSIM 계산 (입력 범위 [0, 1]로 변환)
            batch_out_fsim = (batch_out + 1) / 2
            img_gt_fsim = (img_gt + 1) / 2
            total_fsim += fsim(batch_out_fsim, img_gt_fsim, data_range=1.0).item()
            

    
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count
    avg_fsim = total_fsim / count
    
    return avg_psnr, avg_ssim, avg_lpips, avg_fsim

def main(args):
    # 테스트 이미지 경로 (args에서 가져오거나 하드코딩)
    image_list = os.listdir(args.in_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 모델 파일 경로
    current_dir = '.'
    s3_model_path = './s3.pt'
    
    # 학습한 모델 파일 (newpsnr.pt, newssim.pt 등 패턴으로 검색)
    patterns = ['newpsnr.pt', 'newssim.pt', 'newlpips.pt', 'newfsim.pt', 'newcomposite.pt']
    model_files = {}
    for f in os.listdir(current_dir):
        for pattern in patterns:
            if f.endswith(pattern):
                model_name = pattern.split('new')[1].split('.')[0]
                model_files[model_name] = os.path.join(current_dir, f)
                break
    
    # s3.pt 모델 로드
    s3_model = load_model(s3_model_path, device)
    if s3_model is None:
        print("Failed to load s3.pt. Exiting...")
        return
    
    # s3.pt 모델 평가
    print("\nEvaluating s3.pt...")
    s3_metrics = evaluate_model(s3_model, image_list, args.in_path, args.gt_path, args.im_size, device)
    print(f"s3.pt Metrics - PSNR: {s3_metrics[0]:.2f} dB | SSIM: {s3_metrics[1]:.4f} | LPIPS: {s3_metrics[2]:.4f} | FSIM: {s3_metrics[3]:.4f}")

    
    # 학습한 모델 평가
    for model_name, model_path in model_files.items():
        print(f"\nEvaluating {model_name} model ({model_path})...")
        user_model = load_model(model_path, device)
        if user_model is None:
            print(f"Failed to load {model_path}. Skipping...")
            continue
        
        user_metrics = evaluate_model(user_model, image_list, args.in_path, args.gt_path, args.im_size, device)
        print(f"{model_name} Metrics - PSNR: {user_metrics[0]:.2f} dB | SSIM: {user_metrics[1]:.4f} | LPIPS: {user_metrics[2]:.4f} | FSIM: {user_metrics[3]:.4f}")
        
        # s3.pt와 비교
        print(f"Comparison with s3.pt:")
        print(f"  - PSNR Diff: {user_metrics[0] - s3_metrics[0]:.2f} dB (Higher is better)")
        print(f"  - SSIM Diff: {user_metrics[1] - s3_metrics[1]:.4f} (Higher is better)")
        print(f"  - LPIPS Diff: {user_metrics[2] - s3_metrics[2]:.4f} (Lower is better)")
        print(f"  - FSIM Diff: {user_metrics[3] - s3_metrics[3]:.4f} (Higher is better)")
        
        del user_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare models using performance metrics.")
    parser.add_argument('--in_path', type=str, default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/')
    parser.add_argument('--gt_path', type=str, default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/gt/')
    parser.add_argument('--im_size', type=int, default=512)
    args = parser.parse_args()
    main(args)

#s3.pt와 직접 학습시킨 모델의 체크포인트들을 비교해보았을때 476000_pth의 성능이 제일 좋게 나온것을 알 수 있다. 