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
    self.lin4 = nn.Linear(1024, 1)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    x = F.relu(self.lin3(x))
    x = torch.sigmoid(self.lin4(x))
    return x




if __name__ == "__main__":
  torch.set_default_dtype(torch.float64)
  BS = 16

  data_in = np.loadtxt("input.csv", delimiter = ',')
  print(data_in.shape)
  gt = np.loadtxt("gt.csv", delimiter = ',')

  
  max_din = data_in.max(axis = 0)
  min_din = data_in.min(axis = 0)

  din_norm = (data_in - min_din) / (max_din - min_din)
  din_norm = np.nan_to_num(din_norm, 0.5)
  

  din_norm = torch.tensor(din_norm, dtype = torch.float64)
  gt = torch.tensor(gt, dtype = torch.float64)

  d_data = list(zip(din_norm, gt))

  dl = DataLoader(d_data, batch_size = 16, shuffle = True)

  m = MLP(din_norm.shape[1])
  device = "cuda:0"
  m.to(device)

  opt = torch.optim.Adam(params = m.parameters(), lr = 1e-3)
  crit = torch.nn.BCELoss()

  NUM_EPOCHS = 100
  for j in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dl):
      opt.zero_grad()
      nn_input, h = data
      nn_input = nn_input.to(device)
      h = h.to(device)
      out = m(nn_input)
      loss = crit(out.flatten(), h)
      running_loss += loss.item()
      loss.backward()
      opt.step()
    print(running_loss/i)
  #din_dset = DataLoader(din_norm, batch_size = 16, shuffle = True)
  #dout_dset = DataLoader(dout_norm, batch_size = 16, shuffle = True)
  torch.save(m, "model.pth")
  



  '''
  m = MLP(din_norm.shape[1])
  out = m(torch.tensor(din_norm[0:2, :], dtype = torch.float32))
  print(out.shape)
  '''



