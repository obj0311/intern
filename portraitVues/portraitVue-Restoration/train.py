import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm
import os
import numpy as np

from models import weights_init, Discriminator, FullGenerator, Generator
from operation import copy_G_params, load_params
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from loss.id_loss import IDLoss

policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[0])

#torch.backends.cudnn.benchmark = True
def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part    = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err_ad  = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean()
        err_re  = percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err     = err_ad + err_re
        err.backward()
        return pred.mean().item(), err_ad, err_re, rec_all, rec_small, rec_part
    else:
        pred    = net(data, label)
        err_ad  = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err     = err_ad
        err.backward()
        return pred.mean().item(), err
        

def train(args):
    # # fix the seed for reproducibility
    # seed                = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    writer          = SummaryWriter(args.summary)

    dataset         = ImageFolder(root=args.train_path, im_size=args.im_size, phase='train')
    dataloader      = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=True))
    
    dataset         = ImageFolder(root=args.valid_path, im_size=args.im_size, phase='valid')
    validloader     = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=True))

    generator       = Generator(ngf=args.ngf, nz=args.nz, nc=args.nc, im_size=args.im_size)
    # model_ir_se50.pth load 하는 부분
    if args.gen_pretrained != 'None':
        ckpt                = torch.load(args.gen_pretrained)
        print(type(ckpt))
        print(ckpt.keys())
        # Case 1: 단일 모델 state_dict
        if 'g' not in ckpt.keys():
            generator.load_state_dict(
                {k.replace('module.', ''): v for k, v in ckpt.items()},
                strict=False
            )
        # Case 2: 'g' 키 존재 (기존 코드)
        else:
            generator.load_state_dict(
                {k.replace('module.', ''): v for k, v in ckpt['g'].items()},
                strict=False
            )
        print ("Load Success!!")
        del ckpt
    netG            = FullGenerator(Gen=generator, size=args.im_size, ngf=args.ngf, nz=args.nz, nc=args.nc)
    netD            = Discriminator(ndf=args.ndf, im_size=args.im_size)
    netG.apply(weights_init)
    netD.apply(weights_init)

    netG.to(args.device)
    netD.to(args.device)
    

    avg_param_G     = copy_G_params(netG)
    fixed_noise     = torch.FloatTensor(args.batch_size, args.nz).normal_(0, 1).to(args.device)
    
    optimizerG      = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD      = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    smooth_l1_loss  = torch.nn.SmoothL1Loss().to(args.device)
    id_loss         = IDLoss(device=args.device, ckpt_dict=None)
    
    if args.all_pretrained != None:
        ckpt                = torch.load(args.all_pretrained)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G         = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        # args.start_iter     = int(args.ckpt.split('_')[-1].split('.')[0])
        del ckpt
    
    for iteration in tqdm(range(args.start_iter, args.iter+1)):
        degraded_img, gt_img    = next(dataloader)
        degraded_img            = degraded_img.to(args.device)
        gt_img                  = gt_img.to(args.device)

        noise                   = torch.Tensor(args.batch_size, args.nz).normal_(0, 1).to(args.device)
        fake_images             = netG(noise=noise, input_img=degraded_img)
        gt_img_aug              = DiffAugment(gt_img, policy=policy)
        fake_images_aug         = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()
        d_real_pred, err_d_real_ad, err_d_re, rec_img_all, rec_img_small, rec_img_part = train_d(netD, gt_img_aug, label="real")
        d_fake_pred, err_d_fake_ad = train_d(netD, [fi.detach() for fi in fake_images_aug], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        g_fake_pred         = netD(fake_images_aug, "fake")
        err_g_ad            = -g_fake_pred.mean()
        err_g_lpips         = percept(fake_images[0], gt_img).sum()
        err_g_recon         = smooth_l1_loss(fake_images[0], gt_img).sum()
        err_g_id, __, __    = id_loss(fake_images[0], gt_img, degraded_img)

        err_g               = err_g_ad + args.lambda_recon * err_g_recon + args.lambda_lpips * err_g_lpips + args.lambda_id * err_g_id
        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % args.save_interval == 0:
            backup_para     = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                sample      = netG(noise=fixed_noise, input_img=degraded_img)[0]
                sample      = torch.cat((degraded_img, sample, gt_img), 3).add(1).mul(0.5)
                vutils.save_image(sample, args.sample+'/%d.jpg'%iteration, nrow=args.batch_size)
                vutils.save_image( torch.cat([
                                F.interpolate(degraded_img, 128),
                                rec_img_all, rec_img_small,
                                rec_img_part]).add(1).mul(0.5), args.sample+'/rec_%d.jpg'%iteration, nrow=args.batch_size)
                valid_sample = validation(netG, validloader, iteration, args)

            # torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, args.ckpt+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, args.ckpt+'/all_%d.pth'%iteration)
            random_idx = random.randint(0,1)
            writer.add_image("Image/Train1", torch.squeeze(sample[0,:,:,:]), iteration)
            writer.add_image("Image/Train2", torch.squeeze(sample[1,:,:,:]), iteration)
            writer.add_image("Image/Train3", torch.squeeze(sample[2,:,:,:]), iteration)
            writer.add_image("Image/Train4", torch.squeeze(sample[3,:,:,:]), iteration)
            writer.add_image("Image/Valid1", torch.squeeze(valid_sample[0,:,:,:]), iteration)
            writer.add_image("Image/Valid2", torch.squeeze(valid_sample[1,:,:,:]), iteration)
            writer.add_image("Image/Valid3", torch.squeeze(valid_sample[2,:,:,:]), iteration)
            writer.add_image("Image/Valid4", torch.squeeze(valid_sample[3,:,:,:]), iteration)

        writer.add_scalar("Generator/err_g_ad", err_g_ad.mean().item(), iteration)
        writer.add_scalar("Generator/err_g_recon", err_g_recon.mean().item(), iteration)
        writer.add_scalar("Generator/err_g_lpips", err_g_lpips.mean().item(), iteration)
        writer.add_scalar("Generator/err_g_id", err_g_id.mean().item(), iteration)
        writer.add_scalar("Discriminator/err_d_real_ad", err_d_real_ad.mean().item(), iteration)
        writer.add_scalar("Discriminator/err_d_fake_ad", err_d_fake_ad.mean().item(), iteration)
        writer.add_scalar("Discriminator/d_real_pred", d_real_pred, iteration)
        writer.add_scalar("Discriminator/d_fake_pred", d_fake_pred, iteration)
        writer.add_scalar("Discriminator/err_d_re", err_d_re.mean().item(), iteration)

def validation(model, validloader, step, args):
    fixed_noise     = torch.FloatTensor(args.batch_size, args.nz).normal_(0, 1).to(args.device)
    model.eval()
    with torch.no_grad():
        degraded_img, gt_img  = next(validloader)
        degraded_img          = degraded_img.to(args.device)
        gt_img                = gt_img.to(args.device)

        img_out               = model(noise=fixed_noise, input_img=degraded_img)[0]
        sample                = torch.cat((degraded_img, img_out, gt_img), 3).add(1).mul(0.5)
        vutils.save_image(sample,
                    os.path.join(args.sample, 'valid_'+str(step)+'.jpg'),
                    nrow=args.batch_size
                )
        return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    gen_pre = 'D:/obj_data/portraitVue-Restoration-code/weights/model_ir_se50.pth'

    parser.add_argument('--train_path',     type=str,   default='../portraitVue-Restoration/1_Final/auto_ori/')
    parser.add_argument('--valid_path',     type=str,   default='../portraitVue-Restoration/1_Final/auto_ori/')
    parser.add_argument('--iter',           type=int,   default=500000)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--start_iter',     type=int,   default=0)
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--im_size',        type=int,   default=512)
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--ndf',            type=int,   default=64)
    parser.add_argument('--ngf',            type=int,   default=64)
    parser.add_argument('--nz',             type=int,   default=256)
    parser.add_argument('--nc',             type=int,   default=3)
    parser.add_argument('--save_interval',  type=int,   default=4000)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--lambda_recon',   type=float, default=1.0)
    parser.add_argument('--lambda_lpips',   type=float, default=1.0)
    parser.add_argument('--lambda_id',      type=float, default=1.0)
# lr을 1e-4에서 1e-3로 변경 save interval를 1000에서 4000으로 변경 gen_pretrained로 model_ir_se50.pth 사용
    parser.add_argument('--asset_dir',      type=str,   default='./assets/')
    parser.add_argument('--all_pretrained', type=str,   default=None)
    parser.add_argument('--gen_pretrained', type=str,   default=gen_pre)
    args               = parser.parse_args()
    args.ckpt          =  args.asset_dir + 'ckpt/'
    args.sample        =  args.asset_dir + 'sample/'
    args.summary       =  args.asset_dir + 'summary/'

    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.sample, exist_ok=True)
    os.makedirs(args.summary, exist_ok=True)
    train(args)
