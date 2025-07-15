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


def test(args, model, img_lq, image_name):
    img_lq       = torch.from_numpy(img_lq).to(args.device).permute(2, 0, 1).unsqueeze(0)
    img_lq       = (img_lq/255.-0.5)/0.5
    img_lq       = torch.flip(img_lq, [1])

    with torch.no_grad():
        img_out = model(img_lq)
        utils.save_image(img_out,'./restore_480/' + image_name , nrow=1, normalize=True,  value_range=(-1, 1))    
      



def main(args):

    inpath = 'D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/'

    model = torch.jit.load('all_24000.pt', map_location=args.device)
    image_list = os.listdir(inpath)
    cnt = 0
    for image_name in image_list:
        cnt = cnt+1
        print(cnt, '>>', image_name)
        image = cv2.imread(inpath + image_name)
        image = cv2.resize(image, (args.im_size, args.im_size))
        
        test(args, model, image, image_name.split('/')[-1])

                    

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



    main(args)