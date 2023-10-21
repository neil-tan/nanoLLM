# %%import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from dataloaders.shakespeare_dataloader import ShakespeareDataModule
from dataloaders.codesearchnet_python_dataloader import CodeSearchNetPythonDataModule, tokenizer_factory
from custom_utilities import GlobalStepProgressBar, resolvePath
from model_L import NanoGPT

from functools import partial
import torch

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200

batch_size = 64
block_size = 1024 # context of up to 1024 previous characters
precision = "bf16-true"
num_workers = 8

# datasets
dataset_name = "codesearchnet_python"
datasets = {'shakespeare_char': lambda batch_size, block_size:
              ShakespeareDataModule(
                                    batch_size=batch_size, 
                                    block_size=block_size,
                                    num_workers=num_workers, 
                                    character_encoding=True),

            'codesearchnet_python': lambda batch_size, block_size:
              CodeSearchNetPythonDataModule(
                                    language="python", 
                                    batch_size=batch_size, 
                                    bptt=block_size, 
                                    num_workers=num_workers,
                                    tokenizer_factory=partial(tokenizer_factory, utf8_filter=True, unk_token="💩"), 
                                    mask_type=None,
                                    auto_regressive=True, 
                                    ds_loop=True)
            }

checkpoint_dir = "./out-" + dataset_name + "-char-L"

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 384
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 100 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

bias = False # do we use bias inside LayerNorm and Linear layers?


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
betas = (beta1, beta2)

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
gradient_accumulation_steps = 5

checkpoint_dir = resolvePath(checkpoint_dir)
# %%


# %%

dataloader = datasets[dataset_name](batch_size, block_size)
dataloader.prepare_data()
dataloader.setup()
model_args['vocab_size'] = dataloader.codex.vocab_size

# %%
nanoGPT = NanoGPT(model_args, learning_rate, weight_decay, betas, warmup_iters, lr_decay_iters, min_lr, compile_model=True)

model_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='checkpoint', save_top_k=1)

# gradient accumulation, scaling and clipping is done automatically
# see: https://github.com/Lightning-AI/lightning/discussions/17035
trainer = L.Trainer(accelerator="gpu",
                     devices=1,
                     precision=precision,
                     max_epochs=1,
                     max_steps=max_iters, 
                     gradient_clip_val=grad_clip,
                     gradient_clip_algorithm='norm',
                     accumulate_grad_batches=gradient_accumulation_steps,
                     default_root_dir=checkpoint_dir,
                     callbacks=[GlobalStepProgressBar(), model_checkpoint_callback],
                     )

trainer.fit(nanoGPT, dataloader)

# %%
model_path = model_checkpoint_callback.best_model_path

with open("L_saved_model_path.txt", "w") as f:
    f.write(model_path)

print("model saved to: " + model_path)
