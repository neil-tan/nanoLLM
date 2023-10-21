# %% import stuff
import lightning as L
import os
from os import path
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from zipfile import ZipFile
from typing import Union
from .dataset_utilities import JsonDataset
from tokenization import stoiEncoder
from .dataset_utilities import ConcatenatedDataset
import numpy as np

"""
repo: the owner/repo
path: the full path to the original file
func_name: the function or method name
original_string: the raw string before tokenization or parsing
language: the programming language
code: the part of the original_string that is code
code_tokens: tokenized version of code
docstring: the top-level comment or docstring, if it exists in the original string
docstring_tokens: tokenized version of docstring
sha: this field is not being used [TODO: add note on where this comes from?]
partition: a flag indicating what partition this datum belongs to of {train, valid, test, etc.} This is not used by the model. Instead we rely on directory structure to denote the partition of the data.
url: the url for the code snippet including the line numbers
"""

# %% class defs

# https://stackoverflow.com/questions/55109684/how-to-handle-large-json-file-in-pytorch
class AutoRegressiveLMDataset(IterableDataset):
  def __init__(self, src_json_dataset, bptt:int, tokenizer, num_repeats:int=None, dtype=torch.int64):
    self.src_dataset = src_json_dataset
    self.bptt = bptt
    self.tokenizer = tokenizer
    self.unk_id = self.tokenizer(self.tokenizer.unk_token).input_ids[0]
    self.eos_id = self.tokenizer(self.tokenizer.eos_token).input_ids[0]
    self.padding_tensor = torch.tensor([self.unk_id for i in range(0, self.bptt+1)])
    self.num_repeats = num_repeats if num_repeats is not None else 0
    self.dtype = dtype
  
  def __iter__(self):
    repeat_counter = 0
    while self.num_repeats == -1 or repeat_counter <= self.num_repeats:
      repeat_counter += 1

      for sample in self.src_dataset:
        sample = sample[0] # first field is code

        sample = self.tokenizer(sample).input_ids
        sample_len = len(sample)
        if sample_len > self.bptt:
          i = np.random.randint(low=0, high=sample_len - self.bptt)
          x = torch.tensor((sample[i:i+self.bptt]), dtype=self.dtype)
          y = torch.tensor((sample[i+1:i+1+self.bptt]), dtype=self.dtype)
          # FIXME: no eos token?
        else:
          sample = torch.tensor((sample), dtype=self.dtype)
          sample = torch.cat((sample, self.padding_tensor[sample_len:]), dim=0)
          x = sample[:self.bptt].clone().detach()
          y = sample[1:1+self.bptt].clone().detach()

        yield x, y


class MaskedLMDataset(IterableDataset):
  def __init__(self, src_json_dataset, bptt:int, tokenizer, mask_type:str="torch", num_repeats:int=None, dtype=torch.int64):
    self.src_dataset = src_json_dataset
    self.bptt = bptt
    self.tokenizer = tokenizer
    self.unk_id = self.tokenizer(self.tokenizer.unk_token).input_ids[0]
    self.eos_id = self.tokenizer(self.tokenizer.eos_token).input_ids[0]
    self.padding_tensor = torch.tensor([self.unk_id for i in range(0, self.bptt)])
    self.causal_mask = self._generate_square_subsequent_mask(self.bptt)
    self.null_mask = torch.ones(self.bptt, self.bptt) * float('-inf')
    self.mask_type = mask_type
    self.num_repeats = num_repeats if num_repeats is not None else 0
    self.dtype = dtype
  
  def __iter__(self):
    repeat_counter = 0
    while self.num_repeats == -1 or repeat_counter <= self.num_repeats:
      repeat_counter += 1

      for sample in self.src_dataset:
        sample = sample[0] # first field is code

        if not isinstance(sample, torch.Tensor):
          sample = torch.tensor(self.tokenizer(sample).input_ids, dtype=self.dtype)
        
        if sample.dtype != self.dtype:
          sample = sample.to(self.dtype)

        sample_len = sample.shape[0]
        if sample_len >= self.bptt:
          # print("WARNING: sample length is greater than bptt, truncating")
          sample = sample[:self.bptt]
          sample_len = self.bptt

        # padding and inserting eos token
        if sample_len < self.bptt:
          sample = torch.cat((sample, self.padding_tensor[sample_len:]), dim=0)
        sample[sample_len-1] = self.eos_id
        
        # masking
        if self.mask_type == "torch":
          if sample_len < self.bptt:
            mask = torch.cat((self.causal_mask[:sample_len,:], self.null_mask[sample_len:,:]), dim=0)
          else:
            mask = self.causal_mask
        elif self.mask_type == "huggingface":
          mask = torch.Tensor([1 if i < sample_len else 0 for i in range(self.bptt)])

        yield sample, mask

  def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
      """Generates an upper-triangular matrix of -inf, with zeros on diag."""
      return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def resolvePath(_path):
  _path = path.expandvars(_path)
  _path = path.expanduser(_path)
  _path = path.abspath(_path)
  return _path


class CodeSearchNetPythonDataModule(L.LightningDataModule):
  def __init__(self, language:str, batch_size:int, bptt:int, tokenizer_factory:callable, fields=["code"], data_dir="~/datasets", download_dir="~/tmp", mask_type:str="torch", dtype=torch.int64, auto_regressive:bool=False, ds_loop:bool=False, num_workers:int=8):
    super().__init__()
    self.language = language
    self.zip_filename = self.language + '.zip'
    self.data_path = path.join(resolvePath(data_dir), "codesearchnet")
    self.download_path = path.join(resolvePath(download_dir), self.zip_filename)
    self.url = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/" + self.zip_filename
    self.fields = fields
    self.batch_size = batch_size
    self.json_file_extension = ".jsonl.gz"
    self.bptt = bptt
    self.tokenizer_factory=tokenizer_factory
    self.ds_loop = ds_loop
    self.num_workers = num_workers
    self.dtype = dtype

    assert mask_type in [None, "torch", "huggingface"]
    if auto_regressive:
      assert mask_type is None

    self.mask_type = mask_type
    self.auto_regressive = auto_regressive

  def prepare_data(self):
    if not path.exists(self.data_path):
      os.makedirs(self.data_path)
    
    if not len(os.listdir(self.data_path)) > 0:
      if not path.exists(self.download_path):
        # urllib.request.urlretrieve(self.url, self.download_path)
        self._download_file(self.url, self.download_path)
      
      self._unzip_file(self.download_path, self.data_path,  keep_file_extension=self.json_file_extension)

    self.prepare_datasets()

  
  def prepare_datasets(self):
    json_train_ds = JsonDataset(path.join(self.data_path, self.language, "final", "jsonl", "train"), file_extension=self.json_file_extension, fields= self.fields)
    json_valid_ds = JsonDataset(path.join(self.data_path, self.language, "final", "jsonl", "valid"), file_extension=self.json_file_extension, fields= self.fields)
    json_test_ds = JsonDataset(path.join(self.data_path, self.language, "final", "jsonl", "test"), file_extension=self.json_file_extension, fields= self.fields)

    self.codex = self.tokenizer_factory((json_train_ds, json_valid_ds, json_test_ds))

    ds_num_repeat = -1 if self.ds_loop else 0
    
    if self.auto_regressive:
      self.train = AutoRegressiveLMDataset(json_train_ds, bptt=self.bptt, tokenizer=self.codex, num_repeats=ds_num_repeat, dtype=self.dtype)
      self.valid = AutoRegressiveLMDataset(json_valid_ds, bptt=self.bptt, tokenizer=self.codex, num_repeats=ds_num_repeat, dtype=self.dtype)
      self.test = AutoRegressiveLMDataset(json_test_ds, bptt=self.bptt, tokenizer=self.codex, num_repeats=ds_num_repeat, dtype=self.dtype)
    else:
      self.train = MaskedLMDataset(json_train_ds, bptt=self.bptt, tokenizer=self.codex, mask_type=self.mask_type, num_repeats=ds_num_repeat, dtype=self.dtype)
      self.valid = MaskedLMDataset(json_valid_ds, bptt=self.bptt, tokenizer=self.codex, mask_type=self.mask_type, num_repeats=ds_num_repeat, dtype=self.dtype)
      self.test = MaskedLMDataset(json_test_ds, bptt=self.bptt, tokenizer=self.codex, mask_type=self.mask_type, num_repeats=ds_num_repeat, dtype=self.dtype)


  # https://stackoverflow.com/questions/4006970/monitor-zip-file-extraction-python
  def _unzip_file(self, zip_path, destination, keep_file_extension:Union[str, tuple]=None, ignore_file_extension:Union[str, tuple]=None):
    with ZipFile(file=zip_path) as zip_file:
      file_list = zip_file.namelist()
      if keep_file_extension is not None:
        file_list = [filename for filename in file_list if filename.endswith(keep_file_extension) or filename.endswith("/")]
      if ignore_file_extension is not None:
        file_list = [filename for filename in file_list if not filename.endswith(ignore_file_extension)]

      for file in tqdm(iterable=file_list, total=len(file_list)):
          zip_file.extract(member=file, path=destination)

  # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
  def _download_file(self, url, path):
    # import urllib.request
    import requests
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 #1 MB
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


  def setup(self, stage=None):
    pass

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

  def valid_dataloader(self):
    return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


def tokenizer_factory(datasets, utf8_filter=False, unk_token="\u0000"):
  unique_chars = set()
  dataset = ConcatenatedDataset(datasets)
  for sample in dataset:
    code_string = sample[0]
    for c in code_string:
      if utf8_filter and ord(c) > 128:
        unique_chars.add(unk_token)
        continue
      unique_chars.add(c)
  print("".join(unique_chars))
  return stoiEncoder(unique_chars)

# # %% Tests

# csnPythonDataLoader = CodeSearchNetPythonDataModule("python", batch_size=4, fields=["code"], data_dir="~/dataset", download_dir="~/tmp")
# csnPythonDataLoader.prepare_data()
# csnPythonDataLoader.setup()
# train_dataloader = csnPythonDataLoader.train_dataloader()

# # %% Run the test
# sampled_batch = None
# for batch_item, i in zip(train_dataloader, range(0,3)):
#   if(i == 2):
#     sampled_batch = batch_item
#     print("{0}, len({1}): ".format(i, len(batch_item)))
#     input_ids, mask = batch_item
#     print("---------", input_ids)
#     print(mask)

# # %%
# sampled_batch
# # %%
# x, mask = sampled_batch
# x = x[0]
# mask = mask[0]
# mask.shape
# mask
# # %%
