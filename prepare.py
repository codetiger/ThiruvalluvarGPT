import os
import pickle
import numpy as np
import json

input_file_path = os.path.join(os.path.dirname(__file__), 'kural.json')
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    kuralset = json.load(json_file)

text = ""
for k in kuralset:
    text += k.replace("$", "\n" )
    text += "\n\n"
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(len(text), " Text Chars")
print(vocab_size, " Uinque Chars")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(text)
print(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
