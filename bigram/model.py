"""
A bigram language model just predicts the next token, conditioned on the previous token only. It does not consider any other tokens.

This implementation will use a NxN matrix, where N is the vocab size.
The matrix will be trained by going over every adjacent token pair in the dataset, and iterating the count in the matrix.

Math:
matrix[r,c]
r will be the "input word".
c will be the "output word".

Probability of c following r:
p(c | r) = matrix[r, c] / sum(matrix[r, :])

The full probability distribution for sampling (generative inference):
p(c = C | r) = matrix[r, C] / sum(matrix[r, :])
"""

import numpy as np
import pickle 
import os
from tqdm import tqdm

## Config
# TODO: Allow overriding these from a config file or cmdline
# Model Config
MODEL_SAVEPATH: str = "bigram_model.npy"
RELOAD_MODEL: bool = False

# Sampling Config
STARTING_CHAR: str = "T"
TOKENS_TO_SAMPLE: int = 1_000
# TEMPERATURE: float = 0.0  # TODO: Figure out how to implment this


## Load dataset and override VOCAB_SIZE
prefix = "./"
prefix = "/Users/jains/code/nanoGPT/data/world192"
train_tokens = np.memmap(os.path.join(prefix, "train.bin"), dtype=np.int16, mode='r')
test_tokens = np.memmap(os.path.join(prefix, "test.bin"), dtype=np.int16, mode='r')
with open(os.path.join(prefix, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

VOCAB_SIZE = len(set(train_tokens))
assert VOCAB_SIZE == len(meta["vocab"])

ctoi = lambda c: meta["ctoi_map"][c]
itoc = lambda i: meta["itoc_map"][i]


print(f"{train_tokens=} {test_tokens=}")

## Bigram Matrix
if RELOAD_MODEL and os.path.exists(MODEL_SAVEPATH):
    print(f"Reloading model from {MODEL_SAVEPATH}")
    model = np.load(MODEL_SAVEPATH, allow_pickle=True)
    with open(MODEL_SAVEPATH, "rb") as f:
        model = pickle.load(f)
else:
    model = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)

    ## Training loop
    # Could parallelize this computation too.
    for i in tqdm(range(len(train_tokens) - 1)):
        input_token = train_tokens[i]
        output_token = train_tokens[i+1]
        model[input_token, output_token] += 1

TOTAL_PAIRS = np.prod(model.shape)
print(f"{TOTAL_PAIRS=}")

# Save model to file
with open(MODEL_SAVEPATH, "wb") as f:
    pickle.dump(model, f)


## Print some stats about the model.

# 1. Most frequently occuring pairs of chars.
# Note that this is different than the "most probable" output conditioned on input.
# Get indexes of the top 10 values in the array
min_indicies_r, min_indicies_c = np.unravel_index(np.argsort(model, axis=None), model.shape)
min_indicies = np.transpose((min_indicies_r, min_indicies_c))
max_indicies_r = min_indicies_r[::-1]
max_indicies_c = min_indicies_c[::-1]
max_indicies = np.transpose((max_indicies_r, max_indicies_c))
# model[max_indicies_r, max_indicies_c]  # sanity check values
print("Most frequently occuring pairs of chars")
selected: int = 0
for i, (input, output) in enumerate(max_indicies):
    if selected > 10:
        break
    selected += 1
    print(f"{i=} input:", repr(itoc(input)), "output: ", repr(itoc(output)))


# 2. Non-Occuring pairs of chars
indexes_nonoccuring = np.transpose(np.nonzero(model == 0))
n_nonoccuring = len(indexes_nonoccuring)
print(f"{n_nonoccuring:,} non-occuring pairs, {n_nonoccuring / (TOTAL_PAIRS):,.2%} of total pairs ({TOTAL_PAIRS})")

print("Non-occuring pairs of chars")
selected: int = 0
for i, (input, output) in enumerate(indexes_nonoccuring):
    if selected > 10:
        break
    if np.random.random() < 0.01:  # 1% selection
        selected += 1
        print(f"{i=} input:", repr(itoc(input)), "output:", repr(itoc(output)))


# 3. Highest probability output for each input.
min_indicies_by_row = np.argsort(model, axis=-1)
# Sanity checks:
# min_indicies_by_row[0]  # look at row 0 (which is input 0) sorted ascendingly by value
# model[0]  # look at the values in row 0
# model[0][min_indicies_by_row[0]]  # confirm that sorting by min_indicies_by_row actually shows ascendingly sorted values
for input, row in enumerate(model):
    output = min_indicies_by_row[input][-1]
    output_occurences = row[output]
    print(f"{input=} input:", repr(itoc(input)), "output:", repr(itoc(output)), f"occurences={output_occurences}, % occurences with this input: {output_occurences / np.sum(row):,.2%}")


## Sample from model (generative inference)
result = STARTING_CHAR
while len(result) - 1 < TOKENS_TO_SAMPLE:
    last_token = ctoi(result[-1])
    prob_distribution = model[last_token] / np.sum(model[last_token])  # model[last_token] is a frequency count; turn it into a prob distribution
    sampled_token = np.random.choice([i for i in range(VOCAB_SIZE)], p=prob_distribution)  # TODO: multiple sampling and beam search or nucleus sampling
    result += itoc(sampled_token)

print("Result:")
print(result)


"""
Question to think about:
How does this "frequentist" bigram model compare to using a NN for a bigram model?
I think this model is 'as good as' possible with a bigram model. It captures the whole distribution, which the NN just attempts to approximate.
"""
