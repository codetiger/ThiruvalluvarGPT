import torch
import torch.nn as nn
from torch.nn import functional as F
import json

from bigram import BigramLanguageModel

config = {}
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)
    print("Config: ", config)
    jsonfile.close()

device = 'cpu'
if config["allow_gpu"]:
    if torch.cuda.is_available():
        device = 'cuda' 
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps' 
print("Using device: " + device)
# ------------

with open(config["data_path"], 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


model = BigramLanguageModel(device=device, config=config, vocab_size=vocab_size)
m = model.to(device)
m.load_state_dict(torch.load(config["model_path"]))

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
