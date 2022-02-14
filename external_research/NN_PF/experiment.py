import sys
import time
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
import torch.nn.functional as F
from tqdm import tqdm
import gc

class MLP(nn.Module):
  def __init__(self, insize):
    super(MLP, self).__init__()
    self.lin1 = nn.Linear(insize, 1024)
    self.lin2 = nn.Linear(1024, 1024)
    self.lin3 = nn.Linear(1024, 1024)
    self.lin4 = nn.Linear(1024, 1)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = F.relu(self.lin3(x))
    x = torch.sigmoid(self.lin4(x))
    return x



if __name__ == "__main__":
  epsilon = 1e-5
  mpc = load_case("../../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  gt_mpcd = convert2dict(mpc, device = d)

  data_in = np.loadtxt("input.csv", delimiter = ',')
  print(data_in.shape)

  max_din = data_in.max(axis = 0)
  min_din = data_in.min(axis = 0)

  m = torch.load("model.pth")
  LOAD_LOOKUP = [3250,3500,4000,4500,5000,5500,6000,6500,7000,7500,7750]
  res = np.zeros((len(LOAD_LOOKUP), 6))
  zzz = 0
  for LOAD in LOAD_LOOKUP:
    TT = 0
    TF = 0
    FT = 0 
    FF = 0
    for q in tqdm(range(100)):
      iter_PF = 0
      iter_NN = 0
      mpcd = copy.deepcopy(gt_mpcd)
      center_load = gt_mpcd.bus.Pd.numpy()
      center_load = center_load / (center_load.sum()/LOAD)
      random_load = np.random.normal(center_load, center_load * 0.15, size = center_load.shape)
      random_load[random_load < 0] = 0
      mpcd.bus.Pd = torch.tensor(random_load, dtype=torch.float64)

      iter_PF = newtonPF(mpcd)

      mpcd_Va = mpcd.bus.Va
      mpcd_Vm = mpcd.bus.Vm
      mpcd_Pg = mpcd.gen.Pg
      mpcd_Qg = mpcd.gen.Qg
      mpcd_Pd = mpcd.bus.Pd
      mpcd_Qd = mpcd.bus.Qd

      inputNN = torch.hstack([mpcd_Va, mpcd_Vm, mpcd_Pg, mpcd_Qg, mpcd_Pd, mpcd_Qd])
      inputNN = (inputNN - min_din) / (max_din - min_din)
      inputNN = np.nan_to_num(inputNN, 0.5)
      inputNN = torch.tensor(inputNN, dtype = torch.float64).to("cuda:0")
      out = m(inputNN)
      if out > 0.5:
        iter_NN = 1

      if iter_NN == 1 and iter_PF == 1:
        TT = TT + 1
      if iter_NN == 0 and iter_PF == 1:
        FT = FT + 1
      if iter_NN == 1 and iter_PF == 0:
        TF = TF + 1
      if iter_NN == 0 and iter_PF == 0:
        FF = FF + 1


      mpcd = None
    
    total = TT + FT + TF + FF
    res[zzz,0] = LOAD
    res[zzz,1] = total
    res[zzz,2] = TT 
    res[zzz,3] = FT
    res[zzz,4] = TF
    res[zzz,5] = FF
    zzz = zzz+1

    print("""accuracies:\n
          True PF and True NN: {}%\n
          True PF and False NN: {}%\n
          False PF and True NN: {}%\n
          False PF and False NN: {}%""".format(
          100*float(TT/float(total)),
          100*float(FT/float(total)),
          100*float(TF/float(total)),
          100*float(FF/float(total))))
  np.savetxt("results.csv", res, delimiter = ',')


