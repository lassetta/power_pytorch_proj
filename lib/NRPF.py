import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.sparse as scsp
from scipy.linalg import lu_factor, lu_solve
import torch
import time
import timeit

from LoadCase import *
from auxFCNs import *

def newtonPF(mpcd, device = 'cpu', max_iter = 100):

  # extract VA and VM
  VA = mpcd.bus.Va
  VM = mpcd.bus.Vm
  V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))
  
  # Make Sbus and Ybus
  Sbus = MakeSbus(mpcd, device = device)
  Ybus = MakeYbus(mpcd, device = device)
  # indexing for updates
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


  mm_eqns = torch.multiply(V, torch.conj(torch.matmul(Ybus.T, V))) - Sbus.flatten()
  real_mm = mm_eqns[PVPQidcs].real
  imag_mm = mm_eqns[PQidcs].imag
  F = torch.hstack((real_mm, imag_mm))

  for i in range(max_iter):
    # get the jacobian
    s1 = time.process_time()
    V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))
    VA = torch.angle(V)
    VM = torch.abs(V)
    
    J = getJacobian(mpcd, V, Ybus, device = device)
    norm_F = torch.linalg.norm(F, ord = torch.inf)
    #print(norm_F)
    s = time.process_time()
    if norm_F < 1e-5:
      print("The powerflow successfully converged in {} iterations.".format(i+1))
      break

    dx = torch.linalg.solve(J, -1*F)
    e = time.process_time()
    print("solver time: {}".format(e - s))
    dx = dx.flatten()

    # update!
    VA[PVidcs] = VA[PVidcs] + dx[j1:j2]
    VA[PQidcs] = VA[PQidcs] + dx[j3:j4]
    VM[PQidcs] = VM[PQidcs] + dx[j5:j6]
    VA = VA * 180. / torch.pi 
    V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))

    # recompute mm_eqns
    mm_eqns = torch.multiply(V, torch.conj(torch.matmul(Ybus, V))) - Sbus.flatten()
    real_mm = mm_eqns[PVPQidcs].real
    imag_mm = mm_eqns[PQidcs].imag
    F = torch.hstack((real_mm, imag_mm))
    e1 = time.process_time()
    print("loop time: {}".format(e1 - s1))
    #print(VA)



if __name__ == "__main__":
  mpc = load_case("../case39.m")
  mpc = ext2int(mpc)
  d = 'cuda'
  mpcDict = convert2dict(mpc, device = d)
  #J = getJacobian(mpcDict, device = d)
  newtonPF(mpcDict, device = d)
