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

  Sbus = MakeSbus(mpcd)


