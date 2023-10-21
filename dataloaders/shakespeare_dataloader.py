import os
import requests

import numpy as np
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
import tiktoken

from custom_utilities import resolvePath
from tokenization import stoiEncoder

class CustomIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, np_data_array, block_size):
    self.data = np_data_array
    self.block_size = block_size

  def __iter__(self):
    while True:
      i = np.random.randint(low=0, high=len(self.data) - self.block_size)
      x = torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64))
      y = torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64))
      yield x, y

class ShakespeareDataModule(L.LightningDataModule):
  def __init__(self, batch_size=12, block_size=1024, character_encoding=True, data_dir="~/datasets", num_workers=8):
    super().__init__()
    self.batch_size = batch_size
    self.block_size = block_size
    self.character_encoding = character_encoding
    self.data_dir = resolvePath(data_dir)
    self.num_workers = num_workers

  def prepare_data(self):
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(self.data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    print(f"length of dataset in characters: {n,}")
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]


    # encode with tiktoken gpt2 bpe
    self.codex = tiktoken.get_encoding("gpt2") if not self.character_encoding else stoiEncoder(train_data)
    train_ids = self.codex.encode_ordinary(train_data)
    val_ids = self.codex.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    self.train = CustomIterableDataset(train_ids, self.block_size)
    self.valid = CustomIterableDataset(val_ids, self.block_size)

  def setup(self, stage=None):
    pass

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    raise NotImplementedError