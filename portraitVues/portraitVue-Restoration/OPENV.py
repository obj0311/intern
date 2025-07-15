import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm
from torch import nn, autograd, optim
from torch.utils import data
from models import FullGenerator, Generator
from skimage.metrics import structural_similarity as ssim
from piq import LPIPS
from piq import fsim 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


import openvino as ov
from openvino.runtime import get_version

import psutil  
import time    
import json    


#변경해야할 사항 pt파일 경로 사용할 pt 파일 경로로 설정 필요요

def preprocess(img_path):
    
    #image load 및 resize
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    
    # normalization(training 과 동일)
    img = img.astype(np.float32) / 255.0 #[0,1] 범위
    img = (img - 0.5) / 0.5 # [-1,1] 범위 변환
    
    #tensor 변환 (nchw)
    img = np.transpose(img, (2,0,1))[np.newaxis,:]
    return img

def postprocess(output_tensor, save_path):
    # 1. 텐서 → NumPy 변환 및 차원 조정
    print(type(output_tensor))
    if isinstance(output_tensor, torch.Tensor):
        print('aaadffg')
        output_tensor = output_tensor.numpy()
    output_np = output_tensor.squeeze(0)  # (1,3,512,512) → (3,512,512)
    output_np = np.transpose(output_np, (1,2,0))  # CHW → HWC

    # 2. [-1,1] → [0,255] 범위 변환
    output_np = (output_np * 0.5 + 0.5) * 255  # [-1,1] → [0,255]
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    # 3. BGR → RGB 변환 (OpenCV는 BGR 포맷 사용)
    output_rgb = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)

    # 4. 이미지 저장
    cv2.imwrite(save_path, output_rgb)
    print(f"Image saved to {save_path}")
    
def measure_inference_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

# --- OpenVINO 추론 함수 ---
def openvino_inference(compiled_model, inputs, output_layer):
    return compiled_model(inputs)[output_layer]

# --- PyTorch 추론 함수 ---
def pytorch_inference(model, input_img):
    with torch.no_grad():
        return model(input_img=input_img)
    
    
    
    
    
#torchscript load해서 모델 구성
pt_path = "D:/obj_data/portraitVue-Restoration-code/(lr = 1e-4, batch = 4)/all_24000_newcomposite.pt"
pt_model = torch.jit.load(pt_path, map_location='cpu')

#input image 전처리
a = time.time()
example_input = preprocess("D:/obj_data/portraitVue-Restoration-code/(lr = 1e-4, batch = 4)/input.jpg")
b = time.time()

pre = b-a

# openvino 변환
# 변경점 원래는 pth파일로 generator에 파라미터를 불러오는 식으로 모델을 구성했는데 그냥 pt파일을 불러와서 바로 모델을 구성하도록 바꿨다.
ov_model = ov.convert_model(pt_model, example_input = [example_input], input=[(1, 3, 512, 512)])

#openvino 모델을 xml과 bin 파일로 저장
#xml: 모델내의 layer 순서, layer 특성등을 표현
#bin: 모델의 weight와 bias값을 binary 형태로 가짐
ov.save_model(ov_model, 'netG.xml') # .xml + .bin 생성

# openvino 런타임 코어 객체 생성
core = ov.Core()

# 저장된 xml, bin 파일을 읽어서 network 객체로 로드
network = core.read_model(model = 'netG.xml', weights = 'netG.bin')
print(network)

#모델을 intel CPU 디바이스에 맞게 컴파일
compiled_model = core.compile_model(model = network, device_name = 'CPU')

# 컴파일된 모델의 출력 레이어(첫 번쨰 출력) 이름을 가져옴
output_layer = next(iter(compiled_model.outputs))
print("aaaaa")
print(output_layer)

# --- 성능 측정 시작 ---
metrics = {}

# OpenVINO 추론 측정

ov_result, ov_time = measure_inference_time(
    openvino_inference, compiled_model, [example_input], output_layer)

    
y = ov_result  # 기존 코드와 호환 위해 결과 저장
ovsize_bytes = os.path.getsize('netG.xml')
ovsize_bytes += os.path.getsize('netG.bin')
ovsize_mb = ovsize_bytes / (1024*1024)


#intel GPU가 존재한다면 실행할 수 있는 부분분
'''
core_gpu = ov.Core()
print(core_gpu.available_devices)

network_gpu = core_gpu.read_model(model = 'netG.xml', weights = 'netG.bin')
compiled_model_gpu = core.compile_model(model = network_gpu, device_name = 'GPU')


ov_result, ov_time_gpu = measure_inference_time(
    openvino_inference, compiled_model_gpu, [example_input], output_layer
)

metrics['openvino'] = {
    'inference_time_sec_cpu': round(ov_time, 4),
    'inference_time_sec_gpu': round(ov_time_gpu, 4),
    'model_size_MB': ovsize_mb
}
'''

# PyTorch 추론 측정
i = 0
while(i<3):
    pt_result, pt_time = measure_inference_time(pytorch_inference, pt_model, torch.from_numpy(example_input).to('cpu'))
    i += 1    


pt_output = pt_result  # 기존 코드와 호환 위해 결과 저장
ptsize_bytes = os.path.getsize(pt_path)
ptsize_mb = ptsize_bytes / (1024*1024)


pt_gpu_model = torch.jit.load(pt_path, map_location='cuda:0')

i = 0
while(i<3):
    pt_gpu_result, pt_gpu_time = measure_inference_time(
        pytorch_inference, pt_gpu_model, torch.from_numpy(example_input).to('cuda:0')
    )
    i += 1








e = time.time()
postprocess(y, "output_openvino.png")
f = time.time()
post_ov = f-e

c = time.time()
postprocess(pt_output, "output_python.png")
d = time.time()
post_py = d-c

metrics['openvino'] = {
    'preprocess_time_sec_cpu': round(pre, 4),
    'inference_time_sec_cpu': round(ov_time, 4),
    'postprocess_time_sec_cpu': round(post_ov, 4),

    'model_size_MB': ovsize_mb
}

metrics['pytorch'] = {
    'preprocess_time_sec_cpu': round(pre, 4),
    'inference_time_sec_cpu': round(pt_time, 4),
    'postprocess_time_sec_cpu': round(post_py, 4),
    
    'preprocess_time_sec_gpu': round(pre, 4),
    'inference_time_sec_gpu': round(pt_gpu_time, 4),
    'postprocess_time_sec_gpu': round(post_py, 4),
    
    'model_size_MB': ptsize_mb
}

with open('inference_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)
#pt 파일과 pth파일을 

print(y)
print(y.shape)


    
    
#openvino inference

# 오차 검증
pt_np = pt_output[0].cpu().detach().numpy()  # (1,3,512,512)
ov_np = y[0].squeeze()   # OpenVINO 출력 (1,3,512,512)

# 차원 조정 없이 직접 Tensor 변환
pt_np = torch.from_numpy(pt_np).unsqueeze(0).float()  # (1,3,512,512)
ov_np = torch.from_numpy(ov_np).unsqueeze(0).float()  # (1,3,512,512)

psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to('cpu')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to('cpu')

psnr = psnr_metric(pt_np, ov_np).item()
s_sim = ssim_metric(pt_np, ov_np).item()

pt_lpips = (pt_np + 1)/2
ov_lpips = (ov_np + 1)/2
lpips_loss_fn = LPIPS().to('cpu')
l_pips = lpips_loss_fn(pt_np, ov_np).item()

pt_fsim = (pt_np + 1) / 2  
ov_fsim = (ov_np + 1) / 2        
f_sim = fsim(pt_fsim, ov_fsim, data_range=1.0).item()

total= {'psnr': psnr, 'ssim': s_sim, 'lpips': l_pips, 'fsim': f_sim}

with open('compare_pt_ov.json', 'w', encoding='utf-8') as f:
    json.dump(total, f, ensure_ascii=False, indent=4)

#그냥 모델이랑 openvino 모델이랑 이미지로 비교해서 정성적 평가도 들어가면 좋을듯