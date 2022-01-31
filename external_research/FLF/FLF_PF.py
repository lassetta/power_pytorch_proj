import sys
sys.path.insert(1, '../../lib')

import numpy as np
import pandas as pd
from dotted_dict import DottedDict

from LoadCase import *
from auxFCNs import *
from NRPF import newtonPF


if __name__ == "__main__":
  mpc = load_case("../../data/case39.m")
  mpc = ext2int(mpc)
  d = 'cpu'
  mpcd = convert2dict(mpc, device = d)

  # build the p vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  Sbus = MakeSbus(mpcd)

  PV_active = Sbus[mpcd.bus['type'] == 1].real.flatten()
  PQ_active = Sbus[mpcd.bus['type'] == 2].real.flatten()
  PQ_reactive = Sbus[mpcd.bus['type'] == 2].imag.flatten()
  U = torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 1]).flatten()
  p = torch.hstack([PV_active, PQ_active, PQ_reactive, U])
  print(p.shape)


  # build the state vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  PV_phase = mpcd.bus.Va[mpcd.bus['type'] == 1].flatten()
  PQ_phase = mpcd.bus.Va[mpcd.bus['type'] == 2].flatten()
  PV_alpha = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 1])).flatten()
  PQ_alpha = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 2])).flatten()
  x = torch.hstack([PV_phase, PQ_phase, PQ_alpha, PV_alpha])
  print(x.shape)
  
  sys.exit(1)

  '''
  PV_active = Sbus[mpcd.bus['type'] == 1].real
  PQ_active = Sbus[mpcd.bus['type'] == 2].real
  PV_reactive = Sbus[mpcd.bus['type'] == 1].imag
  PQ_reactive = Sbus[mpcd.bus['type'] == 2].imag
  slack_reactive = Sbus[mpcd.bus['type'] == 3].imag
  known_p = torch.vstack([PQ_reactive, PV_reactive, slack_reactive, PQ_active, PV_active])
  print(known_p.flatten())

  # build the state vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  PV_phase = mpcd.bus.Va[mpcd.bus['type'] == 1]
  PQ_phase = mpcd.bus.Va[mpcd.bus['type'] == 2]
  PV_mag = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 1]))
  PQ_mag = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 2]))
  slack_mag = torch.log(torch.square(mpcd.bus.Vm[mpcd.bus['type'] == 3]))
  state_vec = torch.hstack([PV_mag,PQ_mag, slack_mag,PQ_phase, PV_phase])
  '''

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
  




