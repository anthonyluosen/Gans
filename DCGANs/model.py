#-*-coding =utf-8 -*-
#@time :2021/10/10 17:11
#@Author: Anthony
import torch
import torch.nn as nn

# define the discriminator
class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(channels_img,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            #_block(self,in_channels,out_channels,kernel_size,stride,padding)
            self._block(features_d,features_d * 2, 4, 2, 1),
            self._block(features_d*2, features_d * 4, 4, 2, 1),
            self._block(features_d*4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid(),


        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.dis(x)

#define the generator
class Generator(nn.Module):
    def __init__(self,noise_dim,channels_img,features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim,features_g * 16,4,1,0),             #imag 4*4
            self._block(features_g * 16,features_g * 8,4,2,1),          #imag 8*8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),   #imag 16*16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),   #imag 32*32
            nn.ConvTranspose2d(features_g * 2,channels_img,kernel_size=4,stride=2,padding=1),#imag 64*64
            nn.Tanh()

        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,

            ),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)

def initialize_weight(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.2)


# here we test model
def test():
    n, in_channels, H, W = 8, 3, 64, 64
    x = torch.randn((n,in_channels,H,W))
    disc = Discriminator(in_channels,8)
    noise_dim = 100
    assert disc.shape == (n,1,1,1), 'discriminator failed'
    gen = Generator(noise_dim=noise_dim,features_g=8,channels_img=in_channels)
    Z = torch.randn((n,noise_dim,1,1))
    assert gen(Z).shape == (n,in_channels,H,W) ,'generator failed'