"""
Prepare the world192 (cia) and the bible dataset for CHARACTER-LEVEL language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import zipfile
import shutil

## Download and save the dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = "http://corpus.canterbury.ac.nz/resources/large.zip"
    response = requests.get(data_url)
    zip_file_path = 'large.zip'

    with open(zip_file_path, 'wb') as file:
        file.write(response.content)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('extracted')  # Specify the directory to extract to

    # From 'extracted', combine world192.txt and bible.txt into a single file called input.txt
    file1_path = 'extracted/world192.txt'
    file2_path = 'extracted/bible.txt'
    combined_file_path = 'input.txt'

    # Open the new file in write mode and the source files in read mode
    with open(combined_file_path, 'w') as combined_file:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            combined_file.write(file1.read())
            combined_file.write(file2.read())

    os.remove(zip_file_path)
    shutil.rmtree('extracted')

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
