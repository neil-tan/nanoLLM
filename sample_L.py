import torch
from model_L import NanoGPT

model_path = None
start = "def sort(" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 2000 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

with open("L_saved_model_path.txt", "r") as f:
    model_path = f.read().strip()

print("loading model from: " + model_path)

device_str = 'cuda'

loaded_model = NanoGPT.load_from_checkpoint(model_path)
loaded_model.model.eval()
loaded_model.model.to(device_str)

start_ids = loaded_model.codex.encode_ordinary(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device_str)[None, ...])
for i in range(num_samples):
    y = loaded_model.model.generate(x, max_new_tokens, temperature, top_k)
    generated_text = loaded_model.codex.decode_ordinary((y[0].tolist()))
    generated_text = generated_text.rstrip('\x00') 
    print(" ========== sample {sample_num}, length: {length} ==========".format(sample_num=i, length=len(generated_text)))
    print(generated_text)