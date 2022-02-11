import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

class MLP(nn.Module):
  def __init__(self, insize):
    super(MLP, self).__init__()
    self.lin1 = nn.Linear(insize, 1024)
    self.lin2 = nn.Linear(1024, 1024)
    self.lin3 = nn.Linear(1024, 1024)
    self.lin4 = nn.Linear(1024, insize)

  def forward(self, x):
    x = F.elu(self.lin1(x))
    x = F.elu(self.lin2(x))
    x = F.elu(self.lin3(x))
    x = torch.sigmoid(self.lin4(x))
    return x




if __name__ == "__main__":
  torch.set_default_dtype(torch.float64)
  BS = 16

  data_in = np.loadtxt("data_input.csv", delimiter = ',')
  data_out = np.loadtxt("data_output.csv", delimiter = ',')
  
  max_din = data_in.max(axis = 0)
  max_dout = data_out.max(axis = 0)
  max_all = np.vstack([max_din, max_dout])
  max_d = max_all.max(axis = 0)

  min_din = data_in.min(axis = 0)
  min_dout = data_out.min(axis = 0)
  min_all = np.vstack([min_din, max_dout])
  min_d = min_all.min(axis = 0)

  din_norm = (data_in - min_d) / (max_d - min_d)
  dout_norm = (data_out - min_d) / (max_d - min_d)
  din_norm = np.nan_to_num(din_norm, 0.5)
  dout_norm = np.nan_to_num(dout_norm, 0.5)
  

  din_norm = torch.tensor(din_norm, dtype = torch.float64)
  dout_norm = torch.tensor(dout_norm, dtype = torch.float64)

  d_data = list(zip(din_norm, dout_norm))

  dl = DataLoader(d_data, batch_size = 16, shuffle = True)
  m = MLP(din_norm.shape[1])
  device = "cuda:0"
  m.to(device)

  opt = torch.optim.Adam(params = m.parameters(), lr = 1e-4)
  crit = torch.nn.MSELoss()

  NUM_EPOCHS = 500
  for j in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dl):
      opt.zero_grad()
      nn_input, gt = data
      nn_input = nn_input.to(device)
      gt = gt.to(device)
      out = m(nn_input)
      loss = crit(gt, out)
      running_loss += loss.item()
      loss.backward()
      opt.step()
    print(loss/i)
  #din_dset = DataLoader(din_norm, batch_size = 16, shuffle = True)
  #dout_dset = DataLoader(dout_norm, batch_size = 16, shuffle = True)



  '''
  m = MLP(din_norm.shape[1])
  out = m(torch.tensor(din_norm[0:2, :], dtype = torch.float32))
  print(out.shape)
  '''



