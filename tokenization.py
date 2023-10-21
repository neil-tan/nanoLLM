from collections import namedtuple

Encoding = namedtuple('Encoding', ['input_ids', 'attention'])

class CodexBase():
  unk_token = None
  eos_token = None
  sot_token = None
  # @property
  # def unk_token(self):
  #   raise AttributeError("unk_token is not defined")

  # @property
  # def eos_token(self):
  #   raise AttributeError("eos_token is not defined")
  
  # @property
  # def sot_token(self):
  #   raise AttributeError("sot_token is not defined")

  def encode(self, text):
    raise NotImplementedError("encode method is not implemented")
  
  def decode(self, input_ids):
    raise NotImplementedError("decode method is not implemented")

  def __call__(self, text):
    ret = Encoding(input_ids=self.encode(text), attention=None)
    return ret

class stoiEncoder(CodexBase):
  def __init__(self, data, unk_token="\u0000", eos_token="\u0003", sot_token="\u0002", allow_special_tokens_in_data=False):
    # get all the unique characters that occur in this text

    self.chars = list(set(data))

    if not allow_special_tokens_in_data:
      assert not set((unk_token, eos_token, sot_token)).issubset(set(self.chars)), "special tokens must not be in the data"
    self.unk_token = unk_token
    self.eos_token = eos_token
    self.sot_token = sot_token

    self.chars.append(self.unk_token)
    self.chars.append(self.eos_token)
    self.chars.append(self.sot_token)

    self.chars = sorted(self.chars)
    self.vocab_size = len(self.chars)
    print("all the unique characters:", ''.join(self.chars))
    print(f"vocab size: {self.vocab_size:,}")
    
    # create a mapping from characters to integers
    self.stoi = { ch:i for i,ch in enumerate(self.chars) }
    self.itos = { i:ch for i,ch in enumerate(self.chars) }

  def encode(self, s):
    return self.encode_ordinary(s)

  # encoder: take a string, output a list of integers
  def encode_ordinary(self, s):
    result = []
    for c in s:
      try:
        result.append(self.stoi[c])
      except KeyError:
        result.append(self.stoi[self.unk_token])  
    return result
  
  def decode_ordinary(self, l):
    # decoder: take a list of integers, output a string
    result = []
    for i in range(len(l)):
      try:
        result.append(self.itos[l[i]])
      except KeyError:
        result.append(self.unk_token)
    return ''.join(result)

  def state_dict(self):
    states = {'chars': self.chars,
              'vocab_size': self.vocab_size,
              'stoi': self.stoi,
              'itos': self.itos,
              'unk_token': self.unk_token,
              'eos_token': self.eos_token,
              'sot_token': self.sot_token
              }
    
    return states

  def load_state_dict(self, state_dict):
    self.chars = state_dict['chars']
    self.vocab_size = state_dict['vocab_size']
    self.stoi = state_dict['stoi']
    self.itos = state_dict['itos']
    self.unk_token = state_dict['unk_token']
    self.eos_token = state_dict['eos_token']
    self.sot_token = state_dict['sot_token']


class UnicodeTokenizer:
  def __init__(self):
    self.unk_token = '<|endoftext|>'
    self.eos_token = '<|endoftext|>'
    self.sot_token = '<|endoftext|>'
    self.special_tokens = [self.unk_token, self.eos_token, self.sot_token]

  def __call__(self, text):
    ret = Encoding(input_ids=self.encode(text), attention=None)
    return ret

  def id_special_tokens(self, str):
    for special_token in self.special_tokens:
      if str.startswith(special_token):
        return special_token
  
  def encode_special_token(self, special_token):
    return (self.special_tokens.index(special_token) + 1) * -1
  
  def decode_special_token(self, special_token_id):
    return self.special_tokens[special_token_id * -1 - 1]
  
  def encode_char(self, char):
    return ord(char)

  # return ids and fake mask
  def encode(self, text):
    ids = []
    while True:
      if len(text) == 0:
        break

      special_token = self.id_special_tokens(text)
      if special_token is not None:
        ids.append(self.encode_special_token(special_token))
        text = text[len(special_token):]
      else:
        ids.append(self.encode_char(text[0]))
        text = text[1:]
    
    return ids

  def decode_char(self, char_id):
    return chr(char_id)

  def decode(self, char_ids):
    text = ''
    for char_id in char_ids:
      if char_id < 0:
        text += self.decode_special_token(char_id)
      else:
        text += self.decode_char(char_id)
    return text
