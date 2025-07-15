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

from PIL import Image
from . import degradations
# import lmdb
# import imageio
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class GFPGAN_degradation(object):
    def __init__(self):
        self.kernel_list            = ['iso', 'aniso']
        self.kernel_prob            = [0.5, 0.5]
        self.blur_kernel_size       = 41
        self.blur_sigma             = [0.1, 20]
        self.downsample_range       = [0.8, 8]
        self.noise_range            = [0, 20]
        self.jpeg_range             = [60, 100]
        self.gray_prob              = 0.2
        self.color_jitter_prob      = 0.2
        self.color_jitter_pt_prob   = 0.0
        self.shift                  = 20/255.
    
    # def brush_stroke_mask(self, img, color=(166,197,245)):
    def brush_stroke_mask(self, img, color=(255,255,255)):
        min_num_vertex  = 5
        max_num_vertex  = 28
        mean_angle      = 2*math.pi / 5
        angle_range     = 2*math.pi / 15
        min_width       = 12
        max_width       = 100
        def generate_mask(H, W, img=None):
            average_radius  = math.sqrt(H*H+W*W) / 20
            mask            = Image.new('RGB', (W, H), 0)
            mask            = img

            for _ in range(np.random.randint(1, 4)):
                num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
                angle_min = mean_angle - np.random.uniform(0, angle_range)
                angle_max = mean_angle + np.random.uniform(0, angle_range)
                angles = []
                vertex = []
                for i in range(num_vertex):
                    if i % 2 == 0:
                        angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                    else:
                        angles.append(np.random.uniform(angle_min, angle_max))

                vertex.append((int(np.random.randint(100, W-100)), int(np.random.randint(100, H-100))))
                for i in range(num_vertex):
                    r       = np.clip(np.random.normal(loc=average_radius, scale=average_radius//2),0, 2*average_radius)
                    new_x   = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 100, W-100)
                    new_y   = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 100, H-100)
                    vertex.append((int(new_x), int(new_y)))

                draw        = ImageDraw.Draw(mask)
                width       = int(np.random.uniform(min_width, max_width))
                draw.line(vertex, fill=color, width=width)
                for v in vertex:
                    draw.ellipse((v[0] - width//2,v[1] - width//2,v[0] + width//2,v[1] + width//2),fill=color)
            return mask

        height          = np.shape(img)[0]
        width           = np.shape(img)[1]
        mask            = generate_mask(height, width, img)
        return mask

    def image_fitting(self, img, resolution):
        img         = np.squeeze(img)
        center_x    = np.sum(img[int(resolution/2),:,:],1)
        center_y    = np.sum(img[:,int(resolution/2),:],1)
        not_zero_ind_x= []
        for i in range(len(center_x)):
            data    = center_x[i]
            if data != 0:
                not_zero_ind_x.append(i)

        not_zero_ind_y= []
        for i in range(len(center_y)):
            data    = center_y[i]
            if data != 0:
                not_zero_ind_y.append(i)

        resize_x        = 280
        resize_y        = 400
        black_img       = np.zeros((resolution, resolution, 3))
        cropped_img     = img[not_zero_ind_y[0]:not_zero_ind_y[-1], not_zero_ind_x[0]:not_zero_ind_x[-1], :]
        resize_img      = cv2.resize(cropped_img, (resize_x, resize_y), interpolation=cv2.INTER_LINEAR)

        st_y            = int(resolution/2-resize_y/2)
        st_x            = int(resolution/2-resize_x/2)
        black_img[st_y:st_y+resize_y, st_x:st_x+resize_x, :] = resize_img
        return black_img

    def degrade_process(self, img_gt, resolution, task, phase):
        h, w    = img_gt.shape[:2]
        if phase == 'train':
            if random.random() > 0.5:
                img_gt      = cv2.flip(img_gt, 1)

            # random color jitter 
            if np.random.uniform() < self.color_jitter_prob:
                jitter_val  = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
                img_gt      = img_gt + jitter_val
                img_gt      = np.clip(img_gt, 0, 1)    

            # random grayscale
            if np.random.uniform() < self.gray_prob:
                img_gt      = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt      = np.tile(img_gt[:, :, None], [1, 1, 3])
            
            # blur
            kernel          = degradations.random_mixed_kernels(
                                self.kernel_list,
                                self.kernel_prob,
                                self.blur_kernel_size,
                                self.blur_sigma,
                                self.blur_sigma, [-math.pi, math.pi],
                                noise_range=None)
            img_lq          = cv2.filter2D(img_gt, -1, kernel)
            
            # downsample
            scale           = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq          = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                img_lq  = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
            # jpeg compression
            if self.jpeg_range is not None:
                img_lq  = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

            # round and clip
            img_lq      = np.clip((img_lq * 255.0).round(), 0, 255)

            # resize to original size
            img_lq      = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        else : 
            img_lq      = np.clip((img_gt * 255.0).round(), 0, 255)            
            # img_lq      = self.image_fitting(img_lq, resolution)
            img_lq      = cv2.resize(img_lq, (h,w))
        if task =='FaceInpainting':
            img_lq      = np.asarray(self.brush_stroke_mask(Image.fromarray(img_lq.astype(np.uint8))))
        return img_gt, img_lq

class FaceDataset(Dataset):
    def __init__(self, path, resolution=512, task='FaceInpainting', phase='train'):
        self.resolution = resolution
        self.HQ_imgs    = glob.glob(os.path.join(path, '*.*'))
        self.task       = task
        self.degrader   = GFPGAN_degradation()
        self.phase      = phase
    def __len__(self):
        return len(self.HQ_imgs)

    def __getitem__(self, index):
        img_gt          = cv2.imread(self.HQ_imgs[index], cv2.IMREAD_UNCHANGED)
        img_gt          = cv2.resize(img_gt, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        
        img_gt          = img_gt.astype(np.float32)/255.
        img_gt, img_lq  = self.degrader.degrade_process(img_gt, self.resolution, self.task, self.phase)
        img_lq          = img_lq.astype(np.float32)/255.

        img_gt          =  (torch.from_numpy(img_gt) - 0.5) / 0.5
        img_lq          =  (torch.from_numpy(img_lq) - 0.5) / 0.5

        img_gt          = img_gt.permute(2, 0, 1).flip(0) # BGR->RGB
        img_lq          = img_lq.permute(2, 0, 1).flip(0) # BGR->RGB
        return img_lq, img_gt

    