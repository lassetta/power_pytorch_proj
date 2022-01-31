import models
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np



BATCH_SIZE = 1
CHANNELS = 3
downsample = 96
xformHR = transforms.Compose([transforms.Scale((downsample*4,downsample*4))])
xformLR = transforms.Compose([transforms.Scale((downsample,downsample))])


if __name__ == "__main__":
  G = models.Gen()
  D = models.Disc()
  print(G, D)
  img = PIL.Image.open("cawcaw.jpg")
  img = np.array(img)
  img = torch.tensor(img)
  img = img.unsqueeze(-1)
  img = torch.permute(img, (3,2,0,1))
  print(img.shape)

  HR = xformHR(img)
  LR = xformLR(HR)

  plt.figure(1)
  plt.imshow(torch.permute(HR, (0,2,3,1))[0])
  plt.show()

  plt.figure(1)
  plt.imshow(torch.permute(LR, (0,2,3,1))[0])
  plt.show()
  
  
