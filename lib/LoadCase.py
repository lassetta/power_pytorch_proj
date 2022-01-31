import numpy as np
import pandas as pd

class MPC:
  def __init__(self, baseMVA, bus, gen, branch, gencost):
    self.baseMVA = pd.DataFrame([])
    self.bus = pd.DataFrame([])
    self.gen = pd.DataFrame([])
    self.branch = pd.DataFrame([])
    self.gencost = pd.DataFrame([])
    if "None" not in str(type(baseMVA)):
      self.baseMVA = baseMVA
    if "None" not in str(type(bus)):
      self.bus = bus
    if "None" not in str(type(gen)):
      self.gen = gen
    if "None" not in str(type(branch)):
      self.branch = branch
    if "None" not in str(type(gencost)):
      self.gencost = gencost

  def __str__(self):
    return ("__________________\nMPC OBJECT WITH: \nbaseMVA: {}\nbus: {}"
            "\ngen: {}\nbranch: {}\ngencost: {}\n__________________").format(
           self.baseMVA, self.bus.shape, self.gen.shape, self.branch.shape, 
           self.gencost.shape)


def load_case(fname):
  f = open(fname)
  line = f.readlines()
  
  # Bus labels
  bus_labels = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'Area',
                'Vm', 'Va', 'baseKV', 'zone', 'maxVm', 'minVm']

  # Generator labels
  gen_labels = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'baseMVA',
                'status', 'Pmax', 'Pmin']
  gen_labels2 = ['Pc1', 'Pc2','Qc1min', 'Qc1max', 'Qc2min', 'Qc2max',
                 'RR_MW_1','RR_MW_10', 'RR_MW_30', 'RR_MVAR_1', 'APF']

  # Branch labels
  branch_labels = ['f_bus', 't_bus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC',
                   'ratio', 'angle', 'status', 'min_ang_diff', 'max_ang_diff']

  # Gencost labels
  #### LATER
  gen_arr = None
  bus_arr = None
  gencost_arr = None
  branch_arr = None

  BaseMVA = 0


  i = 0
  while(i < len(line)):
    # extract mpc version
    if "mpc.version" in line[i]:
      i = i + 1
    # extract mpc baseMVA
    elif "mpc.baseMVA" in line[i]:
      line[i] = line[i].replace(' ', '')
      line[i] = line[i].replace(';', '')
      line[i] = line[i].replace('\n', '')
      BaseMVA = float(line[i].replace('mpc.baseMVA=', ''))
      i = i + 1
    # extract mpc data arrays
    elif "mpc." in line[i] and '%' not in line[i]:
      oldline = line[i]
      i = i + 1

      arr = []

      while('];' not in line[i]):
        line[i] = line[i].strip()
        line[i] = line[i].strip(';')
        line_data = line[i].split("\t")
        line_data = [float(i) for i in line_data]
        arr.append(line_data)
        i = i + 1

      if 'branch' in oldline:
        branch_arr = np.array(arr)

      if "gen" in oldline and "gencost" not in oldline:
        gen_arr = np.array(arr)

      if "bus" in oldline:
        bus_arr = np.array(arr)

      if "gencost" in oldline:
        gencost_arr = np.array(arr)


    else:
      i = i + 1
  int_buses = ['f_bus', 't_bus', 'status']
  float_branches = ['x', 'r', 'b', 'angle', 'ratio']
  BRANCH = pd.DataFrame(branch_arr[:,0:len(branch_labels)], columns = branch_labels)
  BRANCH[int_buses] =  BRANCH[int_buses].astype(dtype = np.int32)
  BRANCH[float_branches] =  BRANCH[float_branches].astype(dtype = np.double)
  

  int_buses = ['bus_i', 'Area', 'zone']
  float_buses = ['Gs', 'Bs']
  BUS = pd.DataFrame(bus_arr[:,0:len(bus_labels)], columns = bus_labels)
  BUS[int_buses] =  BUS[int_buses].astype(dtype = np.int32)
  BUS[float_buses] =  BUS[float_buses].astype(dtype = np.double)


  int_buses = ['bus', 'status']
  GEN = pd.DataFrame(gen_arr[:,0:len(gen_labels + gen_labels2)], columns = gen_labels + gen_labels2)
  GEN[int_buses] =  GEN[int_buses].astype(dtype = np.int32)

  mpc = dict({"BaseMVA":BaseMVA, "branch":BRANCH, "bus":BUS, "gen":GEN})
  mpc = MPC(BaseMVA, BUS, GEN, BRANCH, None)
  return mpc


def ext2int(mpc):
  # get external to internal conversion map
  ext_idcs = mpc.bus.bus_i
  int_idcs = mpc.bus.index
  ext2int = dict(zip(ext_idcs, int_idcs))

  # Make a conversion from external series to internal
  # corresponding series
  mpc.bus['ibus_i'] = mpc.bus['bus_i'].map(ext2int)
  mpc.gen['ibus'] = mpc.gen['bus'].map(ext2int)
  mpc.branch['if_bus'] = mpc.branch['f_bus'].map(ext2int)
  mpc.branch['it_bus'] = mpc.branch['t_bus'].map(ext2int)

  return mpc

    
