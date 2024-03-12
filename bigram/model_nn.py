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
# from tqdm import tqdm

import torch

## Config
# TODO: Allow overriding these from a config file or cmdline
# Model Config
MODEL_SAVEPATH: str = "bigram_model.pt"
RELOAD_MODEL: bool = False

# Sampling Config
STARTING_CHAR: str = "T"
TOKENS_TO_SAMPLE: int = 1_000
# TEMPERATURE: float = 0.0  # TODO: Figure out how to implment this

BATCH_SIZE: int = 1024
TRAIN_ITERS: int = 10_000


## Load dataset and override VOCAB_SIZE
train_tokens = np.memmap("train.bin", dtype=np.int16, mode='r')
test_tokens = np.memmap("test.bin", dtype=np.int16, mode='r')
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

VOCAB_SIZE = len(set(train_tokens))
assert VOCAB_SIZE == len(meta["vocab"])

ctoi = lambda c: meta["ctoi_map"][c]
itoc = lambda i: meta["itoc_map"][i]


print(f"{train_tokens=} {test_tokens=}")

# Convert to a tensor
# TODO: Shuffle these batches.
current_index: int = 0
def make_batch():
    global current_index

    outputs = train_tokens[current_index+1: current_index+BATCH_SIZE+1]
    inputs = train_tokens[current_index: current_index+len(outputs)]

    output_batch = torch.tensor(outputs, dtype=torch.long)
    input_batch = torch.tensor(inputs, dtype=torch.long)

    assert input_batch.shape == output_batch.shape

    current_index += BATCH_SIZE

    if current_index >= len(train_tokens):
        current_index = 0

    return (input_batch, output_batch)

print(make_batch())
(input_batch, output_batch) = make_batch()


class Network(torch.nn.Module):
    def __init__(self, h1_size=256, h2_size=512):
        super(Network, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.layer1 = torch.nn.Linear(VOCAB_SIZE, self.h1_size)
        self.nonlinearity1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(self.h1_size, self.h2_size)
        self.nonlinearity2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(self.h2_size, VOCAB_SIZE)
        self.final = torch.nn.Softmax()

        torch.nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')

        # T x B x C ??
        # How does Linear deal with TxBxC?

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=VOCAB_SIZE)
        x = x.float()
        # print(f"{x.shape=}")  # [16, 65] # B x vocab_size  # TODO: unsqueeze to get B x T=1 x vocab_size
        x = self.layer1(x)
        # print(f"{x.shape=}")  # [16, 256]
        x = self.nonlinearity1(x)
        x = self.layer2(x)
        # print(f"{x.shape=}")  # [16, 512]
        x = self.nonlinearity2(x)
        x = self.layer3(x)
        # print(f"{x.shape=}")  # [16, 65]
        x = self.final(x)
        # print(f"{x.shape=}")  # [16, 65]  # Should be (B, C)
        return x

model = Network()
model.train(True)

if RELOAD_MODEL and os.path.exists(MODEL_SAVEPATH):
    print(f"Reloading model from {MODEL_SAVEPATH}")
    model.load_state_dict(torch.load(MODEL_SAVEPATH))

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for iter in range(TRAIN_ITERS):
    # get batch
    batch_inputs, batch_outputs = make_batch()
    
    optimizer.zero_grad()

    # forward
    pred = model(batch_inputs)
    # compute loss
    # print(pred)
    # print(pred.shape)
    # print(batch_outputs)
    # print(batch_outputs.shape)
    loss = loss_fn(pred, batch_outputs)
    # print(loss, loss.item())
    # raise ValueError
    # compute gradients
    loss.backward()
    # apply loss
    optimizer.step()

    loss_scalar = loss.item()
    if iter % 10 == 0:
        print(f"{iter=} {loss_scalar=}")


# WHY is it not training? is there something wrong in the loss function computation?


## TODO: eval on test set
        

## TODO: Generative Sampling (Inference)
