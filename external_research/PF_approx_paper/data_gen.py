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
from tqdm import tqdm
import gc

def PF_step(step, conv_thresh, device = 'cpu'):
  converged = 0
  # indexing for updates
  # get the PV, PQ, and slack idcs for both the bus
  # and generator (some internal/external conversions are 
  # necessary
  PQtype = step.bus.type
  PQidcs = step.bus.ibus_i
  PQidcs = PQidcs[PQtype == 1]
  PVtype = step.bus.type
  PVidcs = step.bus.ibus_i
  PVidcs = PVidcs[PVtype == 2]
  
  slacktype  = step.bus.type
  slackidcs  = step.bus.ibus_i
  slackidcs = slackidcs[slacktype == 3]
  PVPQidcs = torch.hstack([PVidcs, PQidcs])
  j1 = 0
  j2 = len(PVidcs)
  j3 = j2
  j4 = j2 + len(PQidcs)
  j5 = j4
  j6 = j5 + len(PQidcs)

  ext_idcs = step.gen.ibus
  int_idcs = [i for i in range(ext_idcs.shape[0])]
  int2ext = dict(zip(list(ext_idcs.numpy()), int_idcs))
  
  slackidcsGen = pd.Series(slackidcs)
  slackidcsGen = slackidcsGen.map(int2ext)
  
  PVidcsGen = pd.Series(PVidcs)
  PVidcsGen = PVidcsGen.map(int2ext)


  # PF STEP
  VA = step.bus.Va
  VM = step.bus.Vm
  V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))

  # Make Sbus and Ybus
  Sbus = MakeSbus(step, device = device)
  Ybus = MakeYbus(step, device = device)

  # Calculate Mismatch eqns
  mm_eqns = torch.multiply(V, torch.conj(torch.matmul(Ybus.T, V))) - Sbus.flatten()
  real_mm = mm_eqns[PVPQidcs].real
  imag_mm = mm_eqns[PQidcs].imag
  F = torch.hstack((real_mm, imag_mm))

  V = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))
  VA = torch.angle(V)
  VM = torch.abs(V)

  # get jacobian
  J = getJacobian(step, V, Ybus, device = device)
  norm_F = torch.linalg.norm(F, ord = torch.inf)
  if norm_F < conv_thresh:
    converged = 1
    return step, converged
  # solve for change
  dx = torch.linalg.solve(J, -1*F)

  # update!
  VA[PVidcs] = VA[PVidcs] + dx[j1:j2]
  VA[PQidcs] = VA[PQidcs] + dx[j3:j4]
  VM[PQidcs] = VM[PQidcs] + dx[j5:j6]
  VA = VA * 180. / torch.pi 
  step.bus.Va = VA
  step.bus.Vm = VM

  # add or subtract??
  # get the Qg for slack and PV buses
  step.gen.Qg[PVidcsGen] += mm_eqns[PVidcs].imag*step.baseMVA
  step.gen.Qg[slackidcsGen] += mm_eqns[slackidcs].imag*step.baseMVA

  # get the Pg for slack bus
  step.gen.Pg[slackidcsGen] += mm_eqns[slackidcs].real*step.baseMVA

  # create masks for constraints
  maxmaskQ = np.argwhere(step.gen.Qmax - step.gen.Qg < 0)
  minmaskQ = np.argwhere(step.gen.Qmin - step.gen.Qg > 0)
  maxmaskP = np.argwhere(step.gen.Pmax - step.gen.Pg < 0)
  minmaskP = np.argwhere(step.gen.Pmin - step.gen.Pg > 0)

  # update if constraints violated
  step.gen.Qg[minmaskQ] = step.gen.Qmin[minmaskQ]
  step.gen.Qg[maxmaskQ] = step.gen.Qmax[maxmaskQ]
  step.gen.Pg[minmaskP] = step.gen.Pmin[minmaskP]
  step.gen.Pg[maxmaskP] = step.gen.Pmax[maxmaskP]
  '''
  print(step.bus.Va)
  print(step.bus.Vm)
  print(step.gen.Pg)
  print(step.gen.Qg)
  print("_____________")
  '''

  return step, 0


if __name__ == "__main__":
  mpc = load_case("../../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  gt_mpcd = convert2dict(mpc, device = d)
  mpcd = copy.deepcopy(gt_mpcd)

  data_k = []
  data_k1 = []
  for qq in tqdm(range(0,2000)):
    gc.collect()
    converged = 0
    i = 0
    mpcd = copy.deepcopy(gt_mpcd)
    center_load = gt_mpcd.bus.Pd.numpy()
    random_load = np.random.uniform(center_load * (2/3), center_load * 1.5, size = center_load.shape)
    mpcd.bus.Pd = torch.tensor(random_load, dtype=torch.float64)
    step = mpcd
    while True:
      # get previous data
      prev_step = step
      prev_step_Va = prev_step.bus.Va
      prev_step_Vm = prev_step.bus.Vm
      prev_step_Pg = prev_step.gen.Pg 
      prev_step_Qg = prev_step.gen.Qg 

      prev_data = torch.hstack([prev_step_Va, prev_step_Vm, prev_step_Pg, prev_step_Qg])
      prev_data = prev_data.flatten().numpy()
      step, converged = PF_step(step, conv_thresh = 1e-9)
      if converged:
        step = None
        prev_step = None
        break
      else:
        step_Va = step.bus.Va
        step_Vm = step.bus.Vm
        step_Pg = step.gen.Pg
        step_Qg = step.gen.Qg

        data = torch.hstack([step_Va, step_Vm, step_Pg, step_Qg])
        data = data.flatten().numpy()

        data_k.append(prev_data)
        data_k1.append(data)
        prev_step = None

        i = i + 1
        if i>100:
          print(i)
          break
  data_k = np.vstack(data_k)
  data_k1 = np.vstack(data_k1)
  print(data_k.shape)
  print(data_k1.shape)
  np.savetxt("data_input.csv", data_k, delimiter = ',')
  np.savetxt("data_output.csv", data_k1, delimiter = ',')



