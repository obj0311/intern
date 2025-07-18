import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random
import math
import numpy as np
seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes,
    ):
        super().__init__()
        self.up1         = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1       = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.activate1   = batchNorm2d(out_planes*2)
        self.GLU1        = GLU()

    def forward(self, input):
        out     = self.up1(input)
        out     = self.conv1(out)
        out     = self.activate1(out)
        out     = self.GLU1(out)
        return out


class UpBlockComp(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes,
    ):
        super().__init__()
        self.up1         = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1       = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise1      = NoiseInjection()
        self.activate1   = batchNorm2d(out_planes*2)
        self.GLU1        = GLU()
        self.conv2       = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise2      = NoiseInjection()
        self.activate2   = batchNorm2d(out_planes*2)
        self.GLU2        = GLU()

    def forward(self, input, noise=None):
        out     = self.up1(input)
        out     = self.conv1(out)
        out     = self.noise1(out, noise)
        out     = self.activate1(out)
        out     = self.GLU1(out)
        out     = self.conv2(out)
        out     = self.noise2(out, noise)
        out     = self.activate2(out)
        out     = self.GLU2(out)
        return out

class FullGenerator(nn.Module):
    def __init__(
        self,
        Gen,
        size,
        ngf,
        nz,
        nc,
        device='cpu'
    ):
        super().__init__()

        self.channels = {
            4:      int(ngf*32),
            8:      int(ngf*16),
            16:     int(ngf*8),
            32:     int(ngf*4),
            64:     int(ngf*4),
            128:    int(ngf*2),
            256:    int(ngf*1),
            512:    int(ngf*0.5),
            1024:   int(ngf*0.25),
        }
        self.log_size   = int(math.log(size, 2))
        self.generator  = Gen
        
        conv            = [conv2d(nc, self.channels[size], 3, 1, 1, bias=False)]
        self.ecd0       = nn.Sequential(*conv)
        in_channel      = self.channels[size]

        self.names      = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            conv        = [conv2d(in_channel, out_channel, 3, 2, 1, bias=False)]
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel  = out_channel

    def forward(self, noise, input_img):
        features = []
        for i in range(self.log_size-1):
            ecd         = getattr(self, self.names[i])
            input_img   = ecd(input_img)
            features.append(input_img)
        features    = features[::-1]
        outs        = self.generator(noise=noise, features=features)
        return outs

class FullGenerator_t(nn.Module):
    def __init__(
            self,
            # Gen,
            size,
            ngf,
            nz,
            nc,
            device='cpu'
    ):
        super().__init__()

        self.channels = {
            4: int(ngf * 32),
            8: int(ngf * 16),
            16: int(ngf * 8),
            32: int(ngf * 4),
            64: int(ngf * 4),
            128: int(ngf * 2),
            256: int(ngf * 1),
            512: int(ngf * 0.5),
            1024: int(ngf * 0.25),
        }
        self.log_size   = int(math.log(size, 2))
        self.generator  = Generator(ngf=ngf, nz=nz, nc=nc, im_size=size)

        conv            = [conv2d(nc, self.channels[size], 3, 1, 1, bias=False)]
        self.ecd0       = nn.Sequential(*conv)
        in_channel      = self.channels[size]

        self.noise = torch.FloatTensor(1, nz).normal_(0, 1).to('cuda:0')

        self.names      = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            conv        = [conv2d(in_channel, out_channel, 3, 2, 1, bias=False)]
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel  = out_channel

    def forward(self, input_img):
        features = []

        for i in range(self.log_size-1):
            ecd         = getattr(self, self.names[i])
            input_img   = ecd(input_img)
            features.append(input_img)
        features    = features[::-1]
        outs        = self.generator(noise=self.noise, features=features)
        return outs[0]

class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=512):
        super(Generator, self).__init__()

        self.channels = {
            4:      int(ngf*16),
            8:      int(ngf*8),
            16:     int(ngf*4),
            32:     int(ngf*2),
            64:     int(ngf*2),
            128:    int(ngf*1),
            256:    int(ngf*0.5),
            512:    int(ngf*0.25),
            1024:   int(ngf*0.125),
        }
        self.im_size    = im_size
        self.nz         = nz

        self.init       = InitLayer(nz, channel=self.channels[4])
        
        self.feat_8     = UpBlockComp(self.channels[4], self.channels[8])
        self.feat_16    = UpBlock(self.channels[8], self.channels[16])
        self.feat_32    = UpBlockComp(self.channels[16], self.channels[32])
        self.feat_64    = UpBlock(self.channels[32], self.channels[64])
        self.feat_128   = UpBlockComp(self.channels[64], self.channels[128])  
        self.feat_256   = UpBlock(self.channels[128], self.channels[256]) 

        self.se_64      = SEBlock(self.channels[4], self.channels[64])
        self.se_128     = SEBlock(self.channels[8], self.channels[128])
        self.se_256     = SEBlock(self.channels[16], self.channels[256])

        self.to_128     = conv2d(self.channels[128], nc, 1, 1, 0, bias=False) 
        self.to_big     = conv2d(self.channels[im_size], nc, 3, 1, 1, bias=False) 
        
        if im_size > 256:
            self.feat_512   = UpBlockComp(self.channels[256], self.channels[512]) 
            self.se_512     = SEBlock(self.channels[32], self.channels[512])
        if im_size > 512:
            self.feat_1024  = UpBlock(self.channels[512], self.channels[1024])  
        
    def forward(self, noise, features):
        if features == None:
            features    = []
            for i in range(9):
                noise_gen   = None
                features.append(noise_gen)

        feat_4      = self.init(noise)
        feat_8      = self.feat_8(feat_4, noise=features[1])
        feat_16     = self.feat_16(feat_8)
        feat_32     = self.feat_32(feat_16, noise=features[3])

        feat_64     = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128    = self.se_128(feat_8, self.feat_128(feat_64, noise=features[5]))
        feat_256    = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            return [torch.tanh(self.to_big(feat_256)), torch.tanh(self.to_128(feat_128))]
        
        feat_512    = self.se_512(feat_32, self.feat_512(feat_256, noise=features[7]))
        if self.im_size == 512:
            return [torch.tanh(self.to_big(feat_512)), torch.tanh(self.to_128(feat_128))]

        feat_1024   = self.feat_1024(feat_512)
        if self.im_size == 1024:
            return [torch.tanh(self.to_big(feat_1024)), torch.tanh(self.to_128(feat_128))]

        
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])
        
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big    = SimpleDecoder(nfc[16], nc)
        self.decoder_part   = SimpleDecoder(nfc[32], nc)
        self.decoder_small  = SimpleDecoder(nfc[32], nc)
        
    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2      = self.down_from_big(imgs[0])        
        feat_4      = self.down_4(feat_2)
        feat_8      = self.down_8(feat_4)
        
        feat_16     = self.down_16(feat_8)
        feat_16     = self.se_2_16(feat_2, feat_16)

        feat_32     = self.down_32(feat_16)
        feat_32     = self.se_4_32(feat_4, feat_32)
        
        feat_last   = self.down_64(feat_32)
        feat_last   = self.se_8_64(feat_8, feat_last)

        #rf_0       = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        #rff_big    = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small  = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1        = self.rf_small(feat_small).view(-1)

        if label=='real':    
            rec_img_big     = self.decoder_big(feat_last)
            rec_img_small   = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1]) 


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf