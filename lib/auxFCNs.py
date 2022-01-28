import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch

from LoadCase import load_case
from LoadCase import ext2int

from types import SimpleNamespace
from dotted_dict import DottedDict


# Build the mpc's Ybus. Outputs a torch tensor
# for the specific device type.
def MakeYbus(mpcd, device = 'cpu'):

  # get Ybus shape
  size = int(max(mpcd.bus.ibus_i) + 1)

  # get internal indexing for to and from buses for branch:
  br_t = mpcd.branch.it_bus
  br_f = mpcd.branch.if_bus

  # get branch data
  br_x = mpcd.branch.x
  br_r = mpcd.branch.r
  br_b = mpcd.branch.b
  br_shift = mpcd.branch.angle
  br_tap = mpcd.branch.ratio
  br_stat = mpcd.branch.status
  
  # get internal indexing for bus
  bus_i = mpcd.bus.ibus_i
  sparse_bus_idcs = (bus_i, bus_i)
  # get bus data
  bus_Gs = mpcd.bus.Gs
  bus_Bs = mpcd.bus.Bs

  # Build the branch components to the Ybus
  br_tap[br_tap == 0] = 1
  br_tap = torch.multiply(br_tap, torch.exp(1j * torch.pi/180. * br_shift))

  y_series = br_stat/(br_r + 1j*br_x)

  y_tt = y_series + 1j*br_b/2.
  y_ff = torch.divide(y_tt, torch.multiply(br_tap, torch.conj(br_tap)))
  y_ft = -1*torch.divide(y_series, torch.conj(br_tap))
  y_tf = -1*torch.divide(y_series, br_tap)
  # Build the bus components to the Ybus
  Ysh = ((bus_Gs + 1j * bus_Bs)) / mpcd.baseMVA

  # store as sparse matrices, add and send to array

  y_tt = coo_matrix((y_tt.cpu(), (br_t.cpu(), br_t.cpu())), (size, size))  
  y_tf = coo_matrix((y_tf.cpu(), (br_t.cpu(), br_f.cpu())), (size, size))  
  y_ft = coo_matrix((y_ft.cpu(), (br_f.cpu(), br_t.cpu())), (size, size))  
  y_ff = coo_matrix((y_ff.cpu(), (br_f.cpu(), br_f.cpu())), (size, size))  
  Ysh = coo_matrix((Ysh.cpu(), (bus_i.cpu(), bus_i.cpu())), (size, size))
  Ybus = torch.tensor((y_tt + y_tf + y_ft + y_ff + Ysh).toarray(), device = device)
  return Ybus

# Build the mpc's Sbus. Outputs a torch tensor
# for the specific device type.
def MakeSbus(mpcd, device = 'cpu'):
  # get dispatch power and reactive power 
  Pd = mpcd.bus.Pd
  Qd = mpcd.bus.Qd
  # get bus internal indices
  bus_idx = mpcd.bus.ibus_i
  size = bus_idx.shape[0]

  # get dispatch power and reactive power 
  Pg = torch.multiply(mpcd.gen.status, mpcd.gen.Pg)
  Qg = torch.multiply(mpcd.gen.status, mpcd.gen.Qg)
  # get generator-bus internal indices
  gen_bus_idx = mpcd.gen.ibus

  zd = torch.zeros_like(bus_idx)
  zg = torch.zeros_like(gen_bus_idx)

  # build matrices for generation and distribution as coo arrays
  Pd = coo_matrix((Pd.cpu(), (bus_idx.cpu(), zd.cpu())), shape = (size,1))
  Qd = coo_matrix((Qd.cpu(), (bus_idx.cpu(), zd.cpu())), shape = (size,1))
  Pg = coo_matrix((Pg.cpu(), (gen_bus_idx.cpu(), zg.cpu())), shape = (size,1))
  Qg = coo_matrix((Qg.cpu(), (gen_bus_idx.cpu(), zg.cpu())), shape = (size,1))

  # Determine Sbus 
  Sbus = ((Pg + 1j*Qg) - (Pd + 1j*Qd)) / mpcd.baseMVA

  # send to array
  Sbus = torch.tensor(Sbus.toarray(), device = device)
  return Sbus

def getJacobian(mpcd, V, Ybus,device = 'cpu'):
    PQtype = mpcd.bus.type
    PQidcs = mpcd.bus.ibus_i
    PQidcs = PQidcs[PQtype == 1]

    PVtype = mpcd.bus.type
    PVidcs = mpcd.bus.ibus_i
    PVidcs = PVidcs[PVtype == 2]

    PVPQidcs = torch.hstack([PVidcs, PQidcs])
    # Get Vall and convert the angle
    VA = torch.angle(V)
    VM = torch.abs(V)
    
    # make the Ybus
    # Ybus = MakeYbus(mpcd, device = device)

    # make the Ibus
    Ibus = torch.matmul(Ybus, V)

    # make diagonal buses
    diagV = torch.diag(V)
    diagIbus = torch.diag(Ibus)
    diagVnorm = torch.diag(torch.divide(V,torch.abs(V)))

    # get the derivative wrt VM and VA
    dSbus_dVM = torch.matmul(diagV, (torch.conj(torch.matmul(Ybus,diagVnorm)))) + torch.matmul(torch.conj(diagIbus),diagVnorm)
    dSbus_dVA = 1j * torch.matmul(diagV,torch.conj(diagIbus - torch.matmul(Ybus, diagV)))

    # compute sub-jacobians 
    j11 = dSbus_dVA[PVPQidcs, :].real
    j11 = j11[:,PVPQidcs]

    j12 = dSbus_dVM[PVPQidcs, :].real
    j12 = j12[:,PQidcs]

    j21 = dSbus_dVA[PQidcs, :].imag
    j21 = j21[:, PVPQidcs]

    j22 = dSbus_dVM[PQidcs, :].imag
    j22 = j22[:,PQidcs]
    
    # stack the jacobians
    j1112 = torch.hstack((j11,j12))
    j2122 = torch.hstack((j21,j22))
    J = torch.vstack((j1112,j2122))

    return J




def convert2dict(mpc, device = 'cpu'):
    busDict = DottedDict()
    for col in mpc.bus.columns:
      busDict[col] = torch.tensor(mpc.bus[col], device = device)

    genDict = DottedDict()
    for col in mpc.gen.columns:
      genDict[col] = torch.tensor(mpc.gen[col], device = device)

    branchDict = DottedDict()
    for col in mpc.branch.columns:
      branchDict[col] = torch.tensor(mpc.branch[col], device = device)

    mpcDict = DottedDict({"baseMVA": mpc.baseMVA, "bus": busDict, "gen": genDict, "branch": branchDict})
    return mpcDict

