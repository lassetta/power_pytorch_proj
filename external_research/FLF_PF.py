import sys
sys.path.insert(1, '../lib')

import numpy as np
import pandas as pd
from dotted_dict import DottedDict

from LoadCase import *
from auxFCNs import *
from NRPF import newtonPF


if __name__ == "__main__":
  mpc = load_case("../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  mpcd = convert2dict(mpc, device = d)

  # build the p vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  Sbus = MakeSbus(mpcd)

  PV_active = Sbus[mpcd.bus['type'] == 1].real
  PQ_active = Sbus[mpcd.bus['type'] == 2].real
  PQ_reactive = Sbus[mpcd.bus['type'] == 2].imag
  known_p = torch.vstack([PQ_reactive, PQ_active, PV_active])
  print(known_p.flatten())

  # build the state vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  PV_phase = mpcd.bus.Va[mpcd.bus['type'] == 1]
  PQ_phase = mpcd.bus.Va[mpcd.bus['type'] == 2]
  PQ_mag = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 2]))
  state_vec = torch.hstack([PQ_mag, PQ_phase, PV_phase])
  print(state_vec)

  # build the intermediate vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  br_t = mpcd.branch.it_bus
  br_f = mpcd.branch.if_bus

  Ui = torch.square(mpcd.bus.Vm) 
  Va_ij = mpcd.bus.Va[br_f] - mpcd.bus.Va[br_t]
  Va_ij = Va_ij * (torch.pi / 180.)
  KL_ij = torch.multiply(mpcd.bus.Vm[br_f], mpcd.bus.Vm[br_t])
  K_ij = torch.multiply(KL_ij, torch.cos(Va_ij))
  L_ij = torch.multiply(KL_ij, torch.sin(Va_ij))

  y = torch.hstack([Ui, K_ij, L_ij])

  alpha_i = torch.log(torch.square(mpcd.bus.Vm))
  alpha_ij = torch.log(torch.square(mpcd.bus.Vm[br_f])) + torch.log(torch.square(mpcd.bus.Vm[br_t]))
  theta_ij = Va_ij

  u = torch.hstack([alpha_i, alpha_ij, theta_ij])

  print(y.shape)
  print(u.shape)
  




