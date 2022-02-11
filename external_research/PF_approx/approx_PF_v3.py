import sys
sys.path.insert(1, '../../lib')
import numpy as np
import pandas as pd
from dotted_dict import DottedDict
from LoadCase import *
from auxFCNs import *
from NRPF import newtonPF
import copy
from torch.autograd import Variable
import itertools
import torch
import torch.nn as nn

class model(nn.Module):
  def __init__(self, in_size, out_updates):
    super(model, self).__init__() 
    self.Lin1 = nn.Linear(in_size, 512)
    self.lr1 = nn.LeakyReLU(0, True)
    self.Lin2 = nn.Linear(512, 512)
    self.lr2 = nn.LeakyReLU(0, True)
    self.Lin3 = nn.Linear(512, 512)
    self.lr3 = nn.LeakyReLU(0, True)
    self.Lin4 = nn.Linear(512, out_updates)
  def forward(self, x):
    x = torch.sigmoid(x)
    x = self.lr1(self.Lin1(x))
    x = self.lr2(self.Lin2(x))
    x = self.lr3(self.Lin3(x))
    return torch.tanh(self.Lin4(x))
    

def f(x,A,B):
  B2 = torch.matmul(B, B.T)
  res = torch.matmul(B2, x) + torch.matmul(B, x) + torch.matmul(A, x)
  return res




if __name__ == "__main__":
  x = torch.randn((10,1)) * 21
  A = torch.randn((10,10)) * 20
  B = torch.randn((10,10)) * 21

  fx = f(x,A,B)

  m = model(10,10)
  opt = torch.optim.Adam(m.parameters(), lr = 1e-4)
  for i in range(1000):
    opt.zero_grad()
    out = torch.unsqueeze(m(x.flatten()), dim = 1)
    fout = f(out, A, B)
    norm_fout = torch.linalg.norm(fout)
    print(norm_fout)
    x += fout.data
    norm_fout.backward()
    opt.step()


  
  







