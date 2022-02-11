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

mpc = load_case("../../data/case4gs.m")
mpc = ext2int(mpc)
d = 'cpu'
mpcd = convert2dict(mpc, device = d)
newtonPF(mpcd)

PQtype = mpcd.bus.type
PQidcs = mpcd.bus.ibus_i
PQidcs = PQidcs[PQtype == 1]
PVtype = mpcd.bus.type
PVidcs = mpcd.bus.ibus_i
PVidcs = PVidcs[PVtype == 2]
PVPQidcs = torch.hstack([PVidcs, PQidcs])
