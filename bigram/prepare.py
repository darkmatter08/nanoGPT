"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

## Download and save the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")


## Define vocab
vocab = sorted(set(data))
vocab_size = len(vocab)
print(f"vocab size: {len(vocab):,}")
print(f"{vocab=}")


## Tokenize
# define tokenization map and functions
ctoi_map = {vocab[i]: i for i in range(vocab_size)}
itoc_map = {i: s for s, i in ctoi_map.items()}

ctoi = lambda c: ctoi_map[c]
itoc = lambda i: itoc_map[i]

# Actually tokenize
# TODO: parallelize this operation by chunking the data using joblib or similar.
# n_threads: int = 8
# slice_starts = ...

# TODO: wrap this in a tqdm progress bar or similar
tokens = list(map(ctoi, data))
print("Tokenization complete")


## Split into train (90%) and test (10%)
cutoff_index = int(len(tokens) * 0.9)
train_tokens = tokens[:cutoff_index]
test_tokens = tokens[cutoff_index:]


## Save to train.bin and val.bin
np.asarray(train_tokens, dtype=np.uint16).tofile("train.bin")
np.asarray(test_tokens, dtype=np.uint16).tofile("test.bin")


## Save encoder and decoder to meta.pkl
meta = {
    'ctoi_map': ctoi_map,
    'itoc_map': itoc_map,
    'vocab': vocab,
}
with open("./meta.pkl", "wb") as f:
    pickle.dump(meta, f)


## Report total number of tokens in train, val, and total
print(f"# Total Tokens: {len(tokens):,}")
print(f"# Train Tokens: {len(train_tokens):,}")
print(f"# Test Tokens: {len(test_tokens):,}")
