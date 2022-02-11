import sys
sys.path.insert(1, '../../lib')
import numpy as np
import pandas as pd
from dotted_dict import DottedDict
from LoadCase import *
from auxFCNs import *
from NRPF import newtonPF
from torch.autograd import Variable
import itertools
import torch
import torch.nn



def mm_step(VAPV, VAPQ, VMPQ, mpcd, opt):
  opt.zero_grad()
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

  VA = mpcd.bus.Va
  VM = mpcd.bus.Vm
  VA[PVidcs] = VAPV
  VA[PQidcs] = VAPQ
  VM[PQidcs] = VMPQ

  V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))
  Sbus = MakeSbus(mpcd, device = 'cpu')
  Ybus = MakeYbus(mpcd, device = 'cpu')

  mm_eqns = torch.multiply(V, torch.conj(torch.matmul(Ybus.T, V))) - Sbus.flatten()
  real_mm = mm_eqns[PVPQidcs].real
  imag_mm = mm_eqns[PQidcs].imag
  F = torch.hstack((real_mm, imag_mm))
  F2 = torch.linalg.norm(F, ord = torch.inf)
  F2.backward(retain_graph = True)
  print(F2)
  #VAPV.data -= 1e-6 * VAPV.grad.data 
  #VAPQ.data -= 1e-6 * VAPQ.grad.data 
  #VMPQ.data -= 1e-6 * VMPQ.grad.data 
  opt.step()
  return VAPV, VAPQ, VMPQ


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

  mpcd.bus.Vm[7] = 1.1
  VAPV = Variable(mpcd.bus.Va[PVidcs], requires_grad = True)
  VAPQ = Variable(mpcd.bus.Va[PQidcs], requires_grad = True)
  VMPQ = Variable(mpcd.bus.Vm[PQidcs], requires_grad = True)

  #VAPV = torch.randn(mpcd.bus.Va[PVidcs].shape, requires_grad = True)
  #VAPQ = torch.randn(mpcd.bus.Va[PQidcs].shape, requires_grad = True)
  #VMPQ = torch.randn(mpcd.bus.Vm[PQidcs].shape, requires_grad = True)

  opt = torch.optim.Adam([VAPV, VAPQ, VMPQ], lr = 1e-3)

  for _ in range(10000):
    VAPV, VAPQ, VMPQ = mm_step(VAPV, VAPQ, VMPQ, mpcd, opt)




