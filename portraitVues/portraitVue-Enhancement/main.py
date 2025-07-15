
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model import FullGenerator, Discriminator
from training.loss.id_loss import IDLoss


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss       = F.softplus(-real_pred)
    fake_loss       = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real,      = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty    = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None):
    smooth_l1_loss, id_loss = loss_funcs
    loss                    = F.softplus(-fake_pred).mean()
    loss_l1                 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __         = id_loss(fake_img, real_img, input_img)
    loss                    += 1.0*loss_l1 + 1.0*loss_id
    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise           = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad,           = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths    = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean       = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty    = (path_lengths - path_mean).pow(2).mean()
    return path_penalty, path_mean.detach()


def train(args, train_loader, valid_loader, generator, discriminator, losses, g_optim, d_optim, g_ema):
    train_loader            = sample_data(train_loader)
    writer                  = SummaryWriter(args.summary_dir)
    pbar                    = range(0, args.iter)
    pbar                    = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length        = 0
    g_module                = generator
    d_module                = discriminator
 
    accum                   = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        degraded_img, real_img  = next(train_loader)
        degraded_img            = degraded_img.to(args.device)
        real_img                = real_img.to(args.device)

        generator.requires_grad     = False
        discriminator.requires_grad =  True

        fake_img, _     = generator(degraded_img)
        fake_pred       = discriminator(fake_img)
        real_pred       = discriminator(real_img)
        d_loss          = d_logistic_loss(real_pred, fake_pred)
      
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if i % args.d_reg_every == 0:
            real_img.requires_grad = True
            real_pred   = discriminator(real_img)
            r1_loss     = args.r1 / 2 * d_r1_loss(real_pred, real_img) * args.d_reg_every

            discriminator.zero_grad()
            r1_loss.backward()
            d_optim.step()

        generator.requires_grad     = True
        discriminator.requires_grad = False

        fake_img, _     = generator(degraded_img)
        fake_pred       = discriminator(fake_img)
        g_loss          = g_nonsaturating_loss(fake_pred, losses, fake_img, real_img, degraded_img)

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        if i % args.g_reg_every == 0:

            fake_img, latents           = generator(degraded_img)
            path_loss, mean_path_length = g_path_regularize(
                                            fake_img, latents, mean_path_length
                                        )
            weighted_path_loss          = args.path_regularize * args.g_reg_every * path_loss
            
            generator.zero_grad()
            weighted_path_loss.backward()
            g_optim.step()


        accumulate(g_ema, g_module, accum)

        writer.add_scalar("Generator/g_loss_val", g_loss.mean().item(), i)
        writer.add_scalar("Generator/weighted_path_val", weighted_path_loss.mean().item(), i)
        writer.add_scalar("Discriminator/d_loss_val", d_loss.mean().item(), i)
        writer.add_scalar("Discriminator/r1_loss_val", r1_loss.mean().item(), i)

        pbar.set_description((f'd: {d_loss.mean().item():.4f}; g: {g_loss.mean().item():.4f}; r1: {r1_loss.mean().item():.4f}; '))
        
        if i % args.save_freq == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _   = g_ema(degraded_img)
                sample      = torch.cat((degraded_img, sample, real_img), 0) 
                utils.save_image(
                    sample,      os.path.join(args.sample, 'train_'+str(i)+'.png'),
                    nrow=args.batch,
                    normalize=True,
                    value_range=(-1, 1))
            validation(g_ema, valid_loader, args, i)
            print(f'{i}/{args.iter}')
            
        if i and i % args.save_freq == 0:
            torch.save(
                {
                    'g': g_module.state_dict(),
                    'd': d_module.state_dict(),
                    'g_ema': g_ema.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(),
                },
                f'{args.ckpt}/{str(i).zfill(6)}.pth',
            )

def validation(model, valid_loader, args, step):
    valid_loader    = sample_data(valid_loader)
    model.eval()
    with torch.no_grad():
        degraded_img, real_img  = next(valid_loader)
        degraded_img            = degraded_img.to(args.device)
        real_img                = real_img.to(args.device)

        img_out, __             = model(degraded_img)
        sample                  = torch.cat((degraded_img, img_out, real_img), 0) 
        utils.save_image(
                    sample,   os.path.join(args.sample, 'valid_'+str(step)+'.png'),
                    nrow=args.batch,
                    normalize=True,
                    value_range=(-1, 1),
                )

def test(args, model, valid_loader):
    print (valid_loader)
    data_num        = len(valid_loader)
    test_loader     = sample_data(valid_loader)
    model.eval()
    with torch.no_grad():
        for da in range(data_num):
            print ("Data index : ", da)
            degraded_img, real_img  = next(test_loader)
            degraded_img            = degraded_img.to(args.device)
            real_img                = real_img.to(args.device)

            img_out, __             = model(degraded_img)
            sample                  = torch.cat((degraded_img, img_out, real_img), 0)
            # utils.save_image(
            #             sample,   os.path.join(args.result_dir, 'test_'+str(da)+'.png'),
            #             nrow=args.batch,
            #             normalize=True,
            #             range=(-1, 1),
            #         )
            utils.save_image(
                        img_out,   os.path.join(args.result_dir, 'test_'+str(da)+'.bmp'),
                        nrow=args.batch,
                        normalize=True,
                        value_range=(-1, 1),
                    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path',               type=str,   default='../portraitVue-Enhancement/2_Final/')
    parser.add_argument('--test_path',          type=str,   default='D:/obj_data/portraitVue-Restoration/1_Final/auto_test/ori/')
    parser.add_argument('--task',               type=str,   default='FaceInpainting')
    parser.add_argument('--phase',              type=str,   default='train')
    parser.add_argument('--base_dir',           type=str,   default='./')
    parser.add_argument('--device',             type=str,   default='cuda:0')

    parser.add_argument('--iter',               type=int,   default=500001)
    parser.add_argument('--batch',              type=int,   default=1)
    parser.add_argument('--size',               type=int,   default=512)
    parser.add_argument('--channel_multiplier', type=int,   default=2)
    parser.add_argument('--narrow',             type=float, default=1.0)
    parser.add_argument('--latent',             type=int,   default=512)
    parser.add_argument('--n_mlp',              type=int,   default=8)
    parser.add_argument('--r1',                 type=float, default=10)
    parser.add_argument('--path_regularize',    type=float, default=2)
    parser.add_argument('--d_reg_every',        type=int,   default=16)
    parser.add_argument('--g_reg_every',        type=int,   default=4)
    parser.add_argument('--lr',                 type=float, default=0.001)

    parser.add_argument('--save_freq',          type=int,   default=4000)
    parser.add_argument('--ckpt',               type=str,   default='./assets/ckpt/')
    parser.add_argument('--sample',             type=str,   default='./assets/sample/')
    parser.add_argument('--result_dir',         type=str,   default='./assets/result/')
    parser.add_argument('--summary_dir',        type=str,   default='./assets/summary/')
    # parser.add_argument('--pretrain',           type=str,   default='./assets/1st/best_model.pth')
    parser.add_argument('--pretrain',           type=str,   default=None)
    parser.add_argument('--start_iter',         type=str,   default=0)

    args = parser.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.sample, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.summary_dir, exist_ok=True)

    generator       = FullGenerator(args.size, args.latent, args.n_mlp, 
                        channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=args.device).to(args.device)
    discriminator   = Discriminator(args.size, channel_multiplier=args.channel_multiplier, 
                        narrow=args.narrow, device=args.device).to(args.device)
    g_ema           = FullGenerator(args.size, args.latent, args.n_mlp, 
                        channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=args.device).to(args.device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio     = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio     = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim         = optim.Adam(
                        generator.parameters(),
                        lr      = args.lr * g_reg_ratio,
                        betas   = (0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
                        )

    d_optim         = optim.Adam(
                        discriminator.parameters(),
                        lr=args.lr * d_reg_ratio,
                        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
                        )

    if args.pretrain is not None:
        print('load model:', args.pretrain)
        
        ckpt    = torch.load(args.pretrain)
        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
            
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
 
    smooth_l1_loss  = torch.nn.SmoothL1Loss().to(args.device)
    id_loss         = IDLoss(args.base_dir, args.device, ckpt_dict=None)
    
    train_dataset   = FaceDataset(args.path, args.size, args.task, 'train')
    train_loader    = data.DataLoader(
                            train_dataset,
                            batch_size=args.batch,
                            sampler=data_sampler(train_dataset, shuffle=True),
                            drop_last=True)
    valid_dataset   = FaceDataset(args.test_path, args.size, args.task, 'valid')
    valid_loader    = data.DataLoader(
                            valid_dataset,
                            batch_size  =args.batch,
                            sampler     =data_sampler(valid_dataset, shuffle=False),
                            drop_last   =True)
    if args.phase == 'train':
        train(args, train_loader, valid_loader, generator, discriminator, [smooth_l1_loss, id_loss], g_optim, d_optim, g_ema)
    elif args.phase == 'test':
        test(args, g_ema, valid_loader)
