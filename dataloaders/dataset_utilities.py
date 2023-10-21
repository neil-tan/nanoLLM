import json
from torch.utils.data import IterableDataset
import gzip
from typing import Union
import os
from os import path


class JsonDataset(IterableDataset):
  def __init__(self, _path, file_extension:Union[str, tuple]=None, fields:tuple=None):
      self.files = [path.join(_path, f) for f in os.listdir(_path) if f.endswith(file_extension) or file_extension is None]
      self.fields = fields

  def __iter__(self):
      for json_file in self.files:
          if json_file.endswith(('.jsonl.gz', '.json.gz')):
            file_reader = gzip.open(json_file, 'rt')
          elif json_file.endswith(('.json', ".jsonl")):
            file_reader = open(json_file, 'r')
          else:
            raise ValueError("File extension not supported")

          with file_reader as f:
              for sample_line in f:
                  sample = json.loads(sample_line)
                  yield tuple(value for name, value in sample.items() if name in self.fields or self.fields is None)

class MapIterDataset(IterableDataset):
  def __init__(self, src_dataset,  map_fns:Union[tuple, list]):
    self.src_dataset = src_dataset
    self.map_fns = map_fns
  
  def __iter__(self):
    for sample in self.src_dataset:
      yield tuple((map_func(data) if map_func is not None else data) for data, map_func in zip(sample, self.map_fns))

class ConcatenatedDataset(IterableDataset):
  def __init__(self, datasets:tuple):
    self.datasets = datasets
  
  def __iter__(self):
    for dataset in self.datasets:
      for sample in dataset:
        yield sample