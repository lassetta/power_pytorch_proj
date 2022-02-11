import sys
sys.path.insert(1, '../../lib')

import numpy as np
import pandas as pd
from dotted_dict import DottedDict

from LoadCase import *
from auxFCNs import *
from NRPF import newtonPF



def f_forward(y,N,B,inv):
  u_est = torch.zeros(y.shape)
  u_est[:N] = torch.log(y[:N])
  u_est[N:N+B] = torch.log(torch.square(y[N:N+B]) + torch.square(y[N+B:]))
  u_est[N+B:] = torch.arctan(y[N+B:]/y[N:N+B])
  if inv == 1:
    return 1/u_est
  else:
    return u_est

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
  

  # build the intermediate vector as described in the paper
  # Factored Solution of Infeasible Load Flow Cases by Antonio Gomez
  br_t = mpcd.branch.it_bus
  br_f = mpcd.branch.if_bus
 

  Ui = torch.square(mpcd.bus.Vm) 
  B = br_t.shape[0]
  N = Ui.shape[0]
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
  
  # Build the E and C matrices
  p = p.unsqueeze(-1)
  y = y.unsqueeze(-1)
  E1 = torch.matmul(p,y.T)
  E2 = torch.inverse(torch.matmul(y,y.T))
  E = torch.matmul(E1,E2)
  print(E1.shape)
  print(E2.shape)
  print(E.shape)

  x = x.unsqueeze(-1)
  u = u.unsqueeze(-1)
  C1 = torch.matmul(u,x.T)
  C2 = torch.inverse(torch.matmul(x,x.T))
  C = torch.matmul(C1,C2)
  print(C.shape)

  # Get the F jacobian
  F = torch.zeros((y.shape[0],y.shape[0]))

  i = 0
  for j in range(Ui.shape[0]):
    F[i,i] = 1/Ui[i]
    i += 1
  for j in range(K_ij.shape[0]):
    scalar = 1/(torch.square(K_ij[j]) + torch.square(L_ij[j]))
    F[i,i] = scalar * 2 * K_ij[j]
    F[i+1,i] = scalar * -1 * L_ij[j]
    F[i,i+1] = scalar * 2 * L_ij[j]
    F[i+1,i+1] = scalar *  K_ij[j]
    i += 2

  u_est = f_forward(y,N,B,1) 

  # update step:
  y_k = f_forward(torch.matmul(C,x),N,B,1).type(torch.DoubleTensor)
  Esq = torch.matmul(E,E.T)
  print(Esq)
  p_Eyk = p - torch.matmul(E,y_k) 
  lam = torch.linalg.solve(Esq, p_Eyk)
  print(lam)




