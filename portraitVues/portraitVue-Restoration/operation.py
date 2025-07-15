import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import shutil
import json
import cv2
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def image_fitting(img, resolution):
    img         = np.squeeze(img)
    center_x    = np.sum(np.sum(img,2),0)
    center_y    = np.sum(np.sum(img,2),1)
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

    cropped_img     = img[not_zero_ind_y[0]:not_zero_ind_y[-1], not_zero_ind_x[0]:not_zero_ind_x[-1], :]
    resize_img      = cv2.resize(cropped_img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return resize_img

class GFPGAN_degradation(object):
  
    def brush_stroke_mask(self, im_size, color=(1,1,1)):
        min_num_vertex  = 5
        max_num_vertex  = 28
        mean_angle      = 2*math.pi / 5
        angle_range     = 2*math.pi / 15
        min_width       = 20
        max_width       = 60
        def generate_mask(im_size):
            H               = im_size
            W               = im_size
            average_radius  = math.sqrt(H*H+W*W) / 20
            mask            = Image.new('RGB', (W, H), (0,0,0))
            gap             = 128

            for _ in range(np.random.randint(1, 4)):
                num_vertex  = np.random.randint(min_num_vertex, max_num_vertex)
                angle_min   = mean_angle - np.random.uniform(0, angle_range)
                angle_max   = mean_angle + np.random.uniform(0, angle_range)
                angles      = []
                vertex      = []
                for i in range(num_vertex):
                    if i % 2 == 0:
                        angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                    else:
                        angles.append(np.random.uniform(angle_min, angle_max))

                vertex.append((int(np.random.randint(gap, W-gap)), int(np.random.randint(gap, H-gap))))
                for i in range(num_vertex):
                    r       = np.clip(np.random.normal(loc=average_radius, scale=average_radius//2),0, 2*average_radius)
                    new_x   = np.clip(vertex[-1][0] + r * math.cos(angles[i]), gap, W-gap)
                    new_y   = np.clip(vertex[-1][1] + r * math.sin(angles[i]), gap, H-gap)
                    vertex.append((int(new_x), int(new_y)))

                draw            = ImageDraw.Draw(mask)
                width           = int(np.random.uniform(min_width, max_width))
                draw.line(vertex, fill=color, width=width)
                for v in vertex:
                    draw.ellipse((v[0] - width//2,
                                  v[1] - width//2,
                                  v[0] + width//2,
                                  v[1] + width//2),
                                 fill=color)
            return mask
        mask            = np.array(generate_mask(im_size))
        mask            = np.transpose(mask,[2,0,1])
        return mask

    def degrade_process(self, img_gt, im_size):
        mask        = np.asarray(self.brush_stroke_mask(im_size))
        img_lq      = img_gt.copy()
        img_lq[np.where(mask==1.0)] = 1.0
        return img_lq

class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, im_size, phase):
        super(ImageFolder, self).__init__()
        self.root       = root
        self.frame      = self._parse_frame()
        self.degrader   = GFPGAN_degradation()
        self.im_size    = im_size
        self.phase      = phase
        self.train_transform  = transforms.Compose([
                            transforms.Resize((int(im_size),int(im_size))),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ])
        self.valid_transform  = transforms.Compose([
                            transforms.Resize((int(im_size),int(im_size))),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ])       

    def _parse_frame(self):
        frame       = []
        img_names   = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.bmp' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_lq          = Image.open(self.frame[idx]).convert('RGB')

        if self.phase == 'train':
            img_gt          = Image.open(self.frame[idx].replace('ori', 'gt')).convert('RGB')
            img_lq          = self.train_transform(img_lq).numpy()
            img_gt          = self.train_transform(img_gt).numpy()
        else:
            img_gt          = Image.open(self.frame[idx]).convert('RGB')
            img_lq          = self.valid_transform(img_lq).numpy()
            img_gt          = self.valid_transform(img_gt).numpy()

        # img_lq = self.degrader.degrade_process(img_lq, self.im_size)

        return torch.from_numpy(img_lq), torch.from_numpy(img_gt)

