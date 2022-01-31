import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# define a residual block model with batch normalization as described in 
# Photo-REalistic Single Image Super-Resolution Using Generative Adversarial Network
class res_block(nn.Module):
  # each residual block consists a 2 [convolutional layer 
  # with stride 1, kernel size 3, and 64 kernels, followed by 
  # a batch normalization], between a Parametric relu
  def __init__(self):
    super(res_block, self).__init__()
    self.conv1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.bn1 = nn.BatchNorm2d(64)
    self.PR = nn.PReLU()
    self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.bn2 = nn.BatchNorm2d(64)

  # forward function for the residual block
  def forward(self, x):
    idt = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.PR(x)
    x = self.conv2(x)
    x = self.bn2(x)
    return torch.add(x, idt)

# upsampleblock to upsample by a factor of 2
class Upsample(nn.Module):
  def __init__(self):
    super(Upsample, self).__init__()
    self.conv = nn.Conv2d(64, 256, kernel_size = 3, stride = 1, padding = 1)
    self.PS = nn.PixelShuffle(2)
    self.PR = nn.PReLU()
  def forward(self, x):
    x = self.conv(x)
    x = self.PS(x)
    return self.PR(x)


# Generator class
class Gen(nn.Module):
  def __init__(self, B = 4, N_UP = 2):
    super(Gen, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size = 9, stride = 1, padding = 4),
        nn.PReLU(),
    )

    self.res_blocks = [res_block() for _ in range(B)]

    self.resid_end = nn.Sequential(
        nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(64),
    )

    self.up_blocks = [Upsample() for _ in range(N_UP)]
    self.final_conv = nn.Conv2d(64,3,kernel_size = 9, stride = 1, padding = 4)

  def forward(self, x):
    # pass through starting block
    x = self.block1(x)
    # hold the skip 
    skip = x
    # for all residual blocks, propagate
    for i in range(len(self.res_blocks)):
      x = self.res_blocks[i](x)
    x = self.resid_end(x)
    x = torch.add(x, skip)
    for i in range(len(self.up_blocks)):
      x = self.up_blocks[i](x)
    #x = self.enhance(x)
    x = self.final_conv(x)
    return x

class Disc(nn.Module):
  def __init__(self):
    super(Disc, self).__init__()
    self.maps = nn.Sequential(
        nn.Conv2d(3,64, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(64,64, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(64,128, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(128,128, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(128,256, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(256,256, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(256,512, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(512,512, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, True),
    )
    self.head = nn.Sequential(
        nn.Linear(512*6*6, 1024),
        nn.LeakyReLU(0.2, True),
        nn.Linear(1024, 1),
    )
  def forward(self, x):
    x = self.maps(x)
    x = torch.flatten(x, 1)
    out = self.head(x)
    return out

if __name__ == "__main__":
  hq = torch.randn(1,3,4*96,4*96)
  a = torch.randn((1,3,96,96))
  print(a.shape)
  m = Gen()
  out = m(a)
  print(out.shape)

  m = ContentLoss()
  loss = m(out, hq)
  print(loss)

  a = torch.randn((1,3,96,96))
  print(a.shape)
  m = Disc()
  out = m(a)
  print(out.shape)




