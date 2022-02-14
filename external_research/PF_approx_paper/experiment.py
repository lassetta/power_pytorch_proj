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
  def __init__(self, insize, aux_size):
    super(MLP, self).__init__()
    self.lin1 = nn.Linear(insize+aux_size, 1024)
    self.lin2 = nn.Linear(1024, 1024)
    self.lin4 = nn.Linear(1024, insize)

  def forward(self, inp, state):
    x = torch.hstack([inp, state])

    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = torch.sigmoid(self.lin4(x))
    return x


if __name__ == "__main__":
  epsilon = 1e-5
  mpc = load_case("../../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  gt_mpcd = convert2dict(mpc, device = d)

  data_in = np.loadtxt("data_input.csv", delimiter = ',')
  data_out = np.loadtxt("data_output.csv", delimiter = ',')
  state_in = np.loadtxt("data_state_l.csv", delimiter = ',')
  
  max_din = data_in.max(axis = 0)
  max_dout = data_out.max(axis = 0)
  max_all = np.vstack([max_din, max_dout])
  max_d = max_all.max(axis = 0)

  min_din = data_in.min(axis = 0)
  min_dout = data_out.min(axis = 0)
  min_all = np.vstack([min_din, max_dout])
  min_d = min_all.min(axis = 0)


  state_max = state_in.max(axis = 0)
  state_min = state_in.min(axis = 0)

  state_norm = (state_in - state_min) / (state_max - state_min)
  print(state_norm.shape)

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

      
      step_Va = mpcd.bus.Va
      step_Vm = mpcd.bus.Vm
      step_Pg = mpcd.gen.Pg
      step_Qg = mpcd.gen.Qg

      data_state_Pd = mpcd.bus.Pd
      data_state_Qd = mpcd.bus.Qd

      state = torch.hstack([data_state_Pd, data_state_Qd])
      state = state.flatten().numpy()
      state = np.nan_to_num(state, 0.5) 

      data = torch.hstack([step_Va, step_Vm, step_Pg, step_Qg])
      data = data.flatten().numpy()

      VA_shape = step_Va.shape[0]
      VM_shape = step_Vm.shape[0]
      PG_shape = step_Pg.shape[0]
      QG_shape = step_Qg.shape[0]

      data_copy = data
      data = (data - min_d) / (max_d - min_d)
      state = (state - state_min) / (state_max - state_min)
      idcs = np.argwhere(np.isnan(data))
      data[idcs] = 0.5
      data = torch.tensor(data, dtype = torch.float64).cuda()
      state = np.nan_to_num(state, 0.5)
      state = torch.tensor(state, dtype = torch.float64).cuda()
      
      check = data.cpu().numpy() * (max_d - min_d) + min_d
      for i in range(250):
        out = m(data, state)
        out[list(idcs.squeeze())] = 0.5
        j = torch.norm(out-data)
        if j < epsilon:
          g = out.detach().cpu().numpy() * (max_d - min_d) + min_d
          g[idcs] = data_copy[idcs]
          g = torch.tensor(g, dtype = torch.float64)
          x0 = VA_shape
          x1 = x0 + VM_shape
          x2 = x1 + PG_shape
          x3 = x2 + QG_shape
          mpcd.bus.Va = g[0:x0]
          mpcd.bus.Vm = g[x0:x1]
          mpcd.gen.Pg = g[x1:x2]
          mpcd.gen.Qg = g[x2:x3]
          iter_NN = newtonPF(mpcd)
          break
        data = out

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


