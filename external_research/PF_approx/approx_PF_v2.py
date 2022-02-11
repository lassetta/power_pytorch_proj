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
    return torch.tanh(self.Lin4(x)) / 1000.
    




def mm_step(VA, VM, VAPV, VAPQ, VMPQ, mpcd):
  PQtype = mpcd.bus.type
  PQidcs = mpcd.bus.ibus_i
  PQidcs = PQidcs[PQtype == 1]
  PVtype = mpcd.bus.type
  PVidcs = mpcd.bus.ibus_i
  PVidcs = PVidcs[PVtype == 2]
  PVPQidcs = torch.hstack([PVidcs, PQidcs])

  j1 = 0
  j2 = len(PVidcs)
  j3 = j2
  j4 = j2 + len(PQidcs)
  j5 = j4
  j6 = j5 + len(PQidcs)

  #VA = mpcd.bus.Va
  #VM = mpcd.bus.Vm
  VA[PVidcs] = VAPV
  VA[PQidcs] = VAPQ
  VM[PQidcs] = VMPQ

  V = torch.multiply(VM, torch.exp(VA*1j))
  Sbus = MakeSbus(mpcd, device = 'cpu')
  Ybus = MakeYbus(mpcd, device = 'cpu')

  mm_eqns = torch.multiply(V, torch.conj(torch.matmul(Ybus.T, V))) - Sbus.flatten()
  real_mm = mm_eqns[PVPQidcs].real
  imag_mm = mm_eqns[PQidcs].imag
  F = torch.hstack((real_mm, imag_mm))

  return F


if __name__ == "__main__":
  mpc = load_case("../../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  mpcd = convert2dict(mpc, device = d)

  PQtype = mpcd.bus.type
  PQidcs = mpcd.bus.ibus_i
  PQidcs = PQidcs[PQtype == 1]
  PVtype = mpcd.bus.type
  PVidcs = mpcd.bus.ibus_i
  PVidcs = PVidcs[PVtype == 2]
  PVPQidcs = torch.hstack([PVidcs, PQidcs])

  mpcd.bus.Pd[2] = 500
  VA = mpcd.bus.Va
  VA = VA * torch.pi / 180
  VM = mpcd.bus.Vm
  #V = torch.multiply(VM, torch.exp(VA))

  print(VA, VM)

  VAPV = VA[PVidcs]
  VAPQ = VA[PQidcs]
  VMPQ = VM[PQidcs]

  F = mm_step(VA, VM, VAPV, VAPQ, VMPQ, mpcd)
  F2 = torch.linalg.norm(F, ord = torch.inf)

  in_shape = F.shape[0] + mpcd.bus.Vm.shape[0]*2
  out_shape = VAPV.shape[0] + VAPQ.shape[0] + VMPQ.shape[0]

  m = model(in_shape, out_shape)
  opt = torch.optim.Adam(m.parameters(), lr = 1e-3)

  for i in range(1000):
    #print(VAPV, VAPQ, VMPQ, F)
    opt.zero_grad()
    x = torch.hstack([F, VM-1, VA])

    print(F2)
    out = m(x.float())
    #mpcd.bus.Va[PVidcs] += out[0:len(PVidcs)]
    #mpcd.bus.Va[PQidcs] += out[len(PVidcs):len(PVidcs) + len(PQidcs)]
    #mpcd.bus.Vm[PQidcs] += out[len(PVidcs)+len(PQidcs):len(PQidcs) + len(PVidcs) + len(PQidcs)]
    oVAPV = VAPV.data
    oVAPQ = VAPQ.data
    oVMPQ = VMPQ.data
    VAPV += out[0:len(PVidcs)]
    VAPQ += out[len(PVidcs):len(PVidcs) + len(PQidcs)]
    VMPQ += out[len(PVidcs)+len(PQidcs):len(PQidcs) + len(PVidcs) + len(PQidcs)]
    F = mm_step(copy.deepcopy(VA), copy.deepcopy(VM), VAPV, VAPQ, VMPQ, mpcd)
    F2 = torch.linalg.norm(F, ord = torch.inf)
    F2.backward()
    opt.step()

    VA[PVidcs]= VAPV.data
    VA[PQidcs] = VAPQ.data
    VM[PQidcs] = VMPQ.data
    #VA = VA.data
    #VM = VM.data
    VAPV = VAPV.data
    VAPQ = VAPQ.data
    VMPQ = VMPQ.data
    F = F.data


  
  sys.exit(1)
  
  







