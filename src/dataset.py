import torch
from torch.utils.data import Dataset

import json
import numpy as np


class fulContxtDataset(Dataset):
  def __init__(self,
    data_file,
    json_file,
    max_length=512):

    self.max_length = max_length

    with open(json_file, 'r') as jf:
      self.token = json.load(jf)

    self.data = []
    self.data_lens = []
    self.num = 0
    
    with open(data_file, 'r') as df:
      for line in df:
        data = line.split()
        if len(data) == 0:
          continue

        data_tmp = []
        for p in data:
          try:
            data_tmp.append(self.token['p2i'][p])
          except KeyError:
            data_tmp.append(self.token['p2i'][p[0]])
            
        data = data_tmp[:max_length]

        self.data.append(data)
        self.data_lens.append(len(data))
        self.num += 1

    self.max_data_len = max(self.data_lens)

  def __len__(self):
    return self.num

  def __getitem__(self, idx):
    data = self.data[idx]
    pad = [0] * (self.max_length - self.data_lens[idx] + 1)
    x = torch.LongTensor(data[:-1] + pad)
    y = torch.LongTensor(data[1:] + pad)
    return x, y

    
if __name__ == "__main__":
  
  data_file = "./data/cc100split/00001.txt"
  json_file = "token.json"

  dataset = fulContxtDataset(data_file, json_file)

  print(len(dataset))
  print(dataset.max_data_len)
  print(min(dataset.data_lens))
  print(np.mean(dataset.data_lens))
  print(dataset[1])

  """
  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.hist(dataset.data_lens, bins=100)
  fig.savefig('fig.png')
  """

  


