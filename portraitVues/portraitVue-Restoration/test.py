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


# Add this function to find the best checkpoint
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
        
        step_lst = []
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
        
        
        # 종합 점수가 가장 높은 모델 저장
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_checkpoint_composite = ckpt_path
            best_checkpoint_composite_file = ckpt_file
            
        print(f"Checkpoint: {ckpt_path} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FSIM: {avg_fsim:.4f} | Composite Score: {composite_score:.4f}")
        
        
        


    print(f"Best checkpoint (PSNR): {best_checkpoint_psnr_file} | Best checkpoint (SSIM): {best_checkpoint_ssim_file}")
    print(f"Best checkpoint (LPIPS): {best_checkpoint_lpips_file} | Best checkpoint (FSIM): {best_checkpoint_fsim_file}")
    print(f"Best checkpoint (Composite): {best_checkpoint_composite_file}")

    print(f"Validation PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f} | LPIPS: {best_lpips:.4f} | FSIM: {best_fsim:.4f} | Best Composite Score: {best_composite_score:.4f}")
    return best_checkpoint_psnr, best_checkpoint_ssim, best_checkpoint_lpips, best_checkpoint_fsim, best_checkpoint_composite


def save_comparison_image(input_img, inference_img, gt_img, save_path, image_name, psnr, ssim, l_pips, f_sim, model_name, titles=['Input', 'Inference', 'GT']):
    # 텐서 → PIL 이미지로 변환
    def tensor_to_pil(tensor):
        tensor = tensor.squeeze(0).cpu()  # [C, H, W]
        tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
        tensor = tensor.clamp(0, 1)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    pil_imgs = [tensor_to_pil(img) for img in [input_img, inference_img, gt_img]]

    # 가로로 이어 붙일 새 이미지 생성 (제목 공간 30픽셀 추가)
    widths, heights = zip(*(im.size for im in pil_imgs))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height + 90), (255, 255, 255))

    # 이미지 붙이기
    x_offset = 0
    for im in pil_imgs:
        new_img.paste(im, (x_offset, 30))
        x_offset += im.size[0]

    # 제목 쓰기
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    #제목
    x_offset = 0
    for im, title in zip(pil_imgs, titles):
        bbox = draw.textbbox((0, 0), title, font=font)
        w = bbox[2] - bbox[0]
        draw.text((x_offset + (im.size[0] - w) / 2, 5), title, fill=(0, 0, 0), font=font)
        x_offset += im.size[0]
    
    
    
    #성능 수치
    metrics_text = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {l_pips:.4f} | FSIM: {f_sim:.4f}"
    bbox = draw.textbbox((0, 0), metrics_text, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((total_width - w) / 2, 35 + max_height), metrics_text, fill=(0, 0, 0), font=font)        
    
    bbox = draw.textbbox((0,0), model_name, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((total_width - w) / 2, 65 + max_height), model_name, fill=(0, 0, 0), font=font)
        

    # 저장
    os.makedirs(save_path, exist_ok = True)
    new_img.save(os.path.join(save_path ,model_name + '_' + image_name ))
    print(f"Saved comparison image to {save_path}")

def find_model_files_by_pattern(directory, patterns):
    """
    디렉토리 내에서 파일 이름이 특정 패턴으로 끝나는 파일을 찾는 함수
    directory: 검색할 디렉토리 경로
    patterns: 찾고자 하는 파일명 패턴 리스트 (예: ['newpsnr.pt', 'newssim.pt'])
    return: 패턴별로 찾은 파일 경로 딕셔너리 (찾지 못하면 None)
    """
    found_files = {pattern: None for pattern in patterns}
    files = os.listdir(directory)
    for pattern in patterns:
        # 파일 이름이 패턴으로 끝나는지 확인하는 정규 표현식
        regex = re.compile(f'.*{re.escape(pattern)}$')
        for f in files:
            if regex.match(f):
                found_files[pattern] = os.path.join(directory, f)
                break
    return found_files

def test(args, model, img_lq, img_gt, image_name, model_name):
    img_lq       = torch.from_numpy(img_lq).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_lq       = (img_lq/255.-0.5)/0.5
    img_lq       = torch.flip(img_lq, [1])

    img_gt       = torch.from_numpy(img_gt).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_gt       = (img_gt/255.-0.5)/0.5
    img_gt       = torch.flip(img_gt, [1])
    
    lpips_loss_fn = LPIPS().to(args.device)



    with torch.no_grad():
        img_out = model(img_lq)
        img_out = img_out.to(args.device)
        img_gt = img_gt.to(args.device)

        # 크기 다르면 보정
        if img_out.shape != img_gt.shape:
            img_gt = torch.nn.functional.interpolate(img_gt, size=img_out.shape[2:], mode='bilinear')

        # 메트릭 초기화
        psnr_metric = PeakSignalNoiseRatio().to(args.device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)

        psnr = psnr_metric(img_out, img_gt).item()
        ssim = ssim_metric(img_out, img_gt).item()
        l_pips = lpips_loss_fn(img_out, img_gt).item()
                
        out_fsim = (img_out + 1) / 2  
        img_gt_fsim = (img_gt + 1) / 2        
        f_sim = fsim(out_fsim, img_gt_fsim, data_range=1.0).item()
        print(f"[{image_name}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {l_pips:.4f} | FSIM: {f_sim:.4f}") 
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
        
      
        save_comparison_image(img_lq, img_out, img_gt, args.merge_path , image_name, psnr, ssim, l_pips,f_sim, model_name)
        utils.save_image(img_out,   os.path.join(args.out_path, model_name + '_' + image_name), nrow=1, normalize=True) #range=(-1,1) 삭제
        # utils.save_image(img_out,   os.path.join(args.out_path, image_name.replace('jpg', 'png')), nrow=1, normalize=True,  range=(-1, 1))


def main(args):
    image_list = os.listdir(args.in_path)

    

    if args.gen_script == True:
        best_ckpt_psnr, best_ckpt_ssim, best_ckpt_lpips, best_ckpt_fsim, best_ckpt_composite = find_best_checkpoint(args, FullGenerator_t, image_list)
        

        checkpoints = {
            'psnr': best_ckpt_psnr,
            'ssim': best_ckpt_ssim,
            'lpips': best_ckpt_lpips,
            'fsim': best_ckpt_fsim,
            'composite': best_ckpt_composite
        }
        
        # 새 파일 이름 생성
        saved_names = {
            name: ckpt.split('/')[-1].split('.')[0] + f'_new{name}.pt'
            for name, ckpt in checkpoints.items()
        }
        
        print("Best Checkpoints:", checkpoints)
        traced_modules = {}
        for name, ckpt_path in checkpoints.items():
            # 모델 인스턴스 생성
            model = FullGenerator_t(size=args.im_size, ngf=args.ngf, nz=args.nz, nc=args.nc, device=args.device).to(args.device)
            
            # 체크포인트 로드
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['g'])
            model.eval()
            
            # TorchScript로 모델 트레이싱
            traced_module = torch.jit.trace(model, torch.rand([1, 3, 512, 512]).to(args.device))
            traced_modules[name] = traced_module
            
            # 저장
            traced_module.save(saved_names[name])

    else:
        
        model_dir = './'  # 현재 디렉토리 또는 저장된 모델 디렉토리 경로로 변경 필요
        patterns = ['newpsnr.pt', 'newssim.pt', 'newlpips.pt', 'newfsim.pt', 'newcomposite.pt']
        model_files = find_model_files_by_pattern(model_dir, patterns)
        
        model_files_mapped = {
            pattern.split('new')[1].split('.')[0]: file_path
            for pattern, file_path in model_files.items()
            if file_path is not None
        }

        if not model_files_mapped:
            print("No model files found with the specified patterns. Please check the directory.")
            return

        cnt = 0
        for image_name in image_list:
            cnt = cnt + 1
            print(cnt, '>>', image_name)
            image = cv2.imread(os.path.join(args.in_path, image_name))
            image_gt = cv2.imread(os.path.join(args.gt_path, image_name))
            
            image = cv2.resize(image, (args.im_size, args.im_size))
            image_gt = cv2.resize(image_gt, (args.im_size, args.im_size))
            
            # 각 모델 파일에 대해 로드 및 테스트 수행
            for model_name, file_path in model_files_mapped.items():
                
                try:
                    model = torch.jit.load(file_path, map_location=args.device)
                    save_prefix = file_path.split('_')[2].split('.')[0]
                    test(args, model, image, image_gt, image_name, save_prefix)
                except FileNotFoundError:
                    print(f"Model file {file_path} not found for {model_name}. Skipping...")
                except Exception as e:
                    print(f"Error loading model {file_path} for {model_name}: {e}. Skipping...")
                    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    
    parser.add_argument("--in_path",        type=str,   default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/')
    parser.add_argument("--out_path",       type=str,   default='D:/obj_data/portraitVue-Restoration/new_Final/auto_test/out/')
    parser.add_argument("--merge_path",       type=str,   default='D:/obj_data/portraitVue-Restoration/new_Final/auto_test/merge/')
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

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.merge_path, exist_ok=True)

    main(args)

# test된 이미지 저장시 input 이미지 + inference 후 생성된 이미지 이렇게 2개를 한꺼번에 보여주는 게 좋을듯
# 그리고 gt랑 inference 후 이미지랑 비교해서 성능 측정해보는것도 좋을듯듯
# find_best_checkpoint -> 모든 체크포인트에 대해서 pretrained로 load한뒤 가장 성능이 좋은 pth파일을 찾아서 출력한다. 
# psnr, ssim 두 가지 수치에 대해서 각각 가장 좋은 수치를 나타낸 pth 파일을 튜플 쌍으로 return한다.
# save_comparison_image input, inference, gt 이미지 이렇게 세 개의 이미지를 한 번에 나타내며 밑에 성능이랑 사용한 pth 파일 이름까지 나타내준다.
# 9000 -> psnr 최고 성능 ckpt, 327000 -> ssim 최고 성능 ckpt