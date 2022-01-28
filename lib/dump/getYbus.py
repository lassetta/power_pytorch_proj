import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch
import sys
sys.path.insert(1, '~/POWER_RESEARCH/pytorch_power/lib')

from LoadCase import load_case
from LoadCase import ext2int

# Build the mpc's Ybus. Outputs a torch tensor
# for the specific device type.
def MakeYbus(mpc, device = 'cpu'):

  # get Ybus shape
  size = int(max(mpc.bus.ibus_i) + 1)

  # get internal indexing for to and from buses for branch:
  br_t = mpc.branch.it_bus.values
  br_f = mpc.branch.if_bus.values

  # get branch data
  br_x = mpc.branch.x.values
  br_r = mpc.branch.r.values
  br_b = mpc.branch.b.values
  br_shift = mpc.branch.angle.values
  br_tap = mpc.branch.ratio.values
  br_stat = mpc.branch.status.values
  
  # get internal indexing for bus
  bus_i = mpc.bus.ibus_i.values
  sparse_bus_idcs = (bus_i, bus_i)
  # get bus data
  bus_Gs = mpc.bus.Gs.values
  bus_Bs = mpc.bus.Bs.values

  # Build the branch components to the Ybus
  br_tap[br_tap == 0] = 1
  br_tap = np.multiply(br_tap, np.exp(1j * np.pi/180. * br_shift))

  y_series = br_stat/(br_r + 1j*br_x)

  y_tt = y_series + 1j*br_b/2.
  y_ff = np.divide(y_tt, np.multiply(br_tap, np.conj(br_tap)))
  y_ft = -1*np.divide(y_series, np.conj(br_tap))
  y_tf = -1*np.divide(y_series, br_tap)
  # Build the bus components to the Ybus
  Ysh = (bus_Gs + 1j * bus_Bs) * .01

  # store as sparse matrices, add and send to array
  y_tt = coo_matrix((y_tt, (br_t, br_t)), (size, size))  
  y_tf = coo_matrix((y_tf, (br_t, br_f)), (size, size))  
  y_ft = coo_matrix((y_ft, (br_f, br_t)), (size, size))  
  y_ff = coo_matrix((y_ff, (br_f, br_f)), (size, size))  
  Ysh = coo_matrix((Ysh, (bus_i, bus_i)), (size, size))
  Ybus = torch.tensor((y_tt + y_tf + y_ft + y_ff + Ysh).toarray(), device = device)
  return Ybus

# Build the mpc's Sbus. Outputs a torch tensor
# for the specific device type.
def getSbus(mpc, device = 'cpu'):
  # get dispatch power and reactive power 
  Pd = mpc.bus.Pd.values
  Qd = mpc.bus.Qd.values
  # get bus internal indices
  bus_idx = mpc.bus.ibus_i.values
  size = bus_idx.shape[0]

  # get dispatch power and reactive power 
  Pg = np.multiply(mpc.gen.status.values, mpc.gen.Pg.values)
  Qg = np.multiply(mpc.gen.status.values, mpc.gen.Qg.values)
  # get generator-bus internal indices
  gen_bus_idx = mpc.gen.ibus.values

  zd = np.zeros_like(bus_idx)
  zg = np.zeros_like(gen_bus_idx)

  # build matrices for generation and distribution as coo arrays
  Pd = coo_matrix((Pd, (bus_idx, zd)), shape = (size,1))
  Qd = coo_matrix((Qd, (bus_idx, zd)), shape = (size,1))
  Pg = coo_matrix((Pg, (gen_bus_idx, zg)), shape = (size,1))
  Qg = coo_matrix((Qg, (gen_bus_idx, zg)), shape = (size,1))

  # Determine Sbus 
  Sbus = (Pg + 1j*Qg) - (Pd + 1j*Qd) / mpc.baseMVA

  # send to array
  Sbus = torch.tensor(Sbus.toarray(), device = device)
  return Sbus

def getJacobian(Ybus, VA, VM):
    Vall = torch.multiply(VM, torch.exp( (torch.pi*1j/180) * VA))
    VA = torch.angle(Vall)
    VM = torch.abs(VM)
    Ibus = torch.matmul(Ybus, Vall)

def convert2dict(mpc, device = 'cpu'):
    busDict = {}
    for col in mpc.bus.columns:
      busDict[col] = torch.tensor(mpc.bus[col], device = device)

    genDict = {}
    for col in mpc.gen.columns:
      genDict[col] = torch.tensor(mpc.gen[col], device = device)

    branchDict = {}
    for col in mpc.branch.columns:
      branchDict[col] = torch.tensor(mpc.branch[col], device = device)

    mpcDict = {"bus": busDict, "gen": genDict, "branch": branchDict}
    return mpcDict

if __name__ == "__main__":
  mpc = load_case("../case300.m")
  mpcDict = convert2dict(mpc)
  print(mpcDict["bus"])

#if __name__ == "__main__":
#  mpc = load_case("../case300.m")
#  mpc = ext2int(mpc)
#  Sbus = getSbus(mpc, device = 'cuda')
#  print(Sbus)
