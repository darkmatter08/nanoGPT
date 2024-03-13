"""
A bigram language model just predicts the next token, conditioned on the previous token only. It does not consider any other tokens.

This implementation will use a 3 layer MLP that is trained on next token prediction.
"""

import numpy as np
import pickle 
import os

import torch

## Config
# TODO: Allow overriding these from a config file or cmdline
# Model Config
MODEL_SAVEPATH: str = "bigram_model.pt"
RELOAD_MODEL: bool = False

# Sampling Config
STARTING_CHAR: str = "?"
TOKENS_TO_SAMPLE: int = 1_000
# TEMPERATURE: float = 0.0  # TODO: Figure out how to implment this

# Training Config
BATCH_SIZE: int = 1024
TRAIN_ITERS: int = 10_000

# Optimizer Config
LR: float = 0.0001
MOMENTUM: float = 0.90

EVAL: bool = True
SAMPLE: bool = True

## Load dataset
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


def make_batch(test=False):
    tokens = test_tokens if not test else train_tokens
    # TODO: Shuffle within the batch
    current_index = np.random.randint(0, len(tokens) - BATCH_SIZE - 1)

    outputs = tokens[current_index+1: current_index+BATCH_SIZE+1]
    inputs = tokens[current_index: current_index+len(outputs)]

    output_batch = torch.tensor(outputs, dtype=torch.long)
    input_batch = torch.tensor(inputs, dtype=torch.long)

    assert input_batch.shape == output_batch.shape

    return (input_batch, output_batch)


class Network(torch.nn.Module):
    def __init__(self, h1_size=256, h2_size=2048):
        super(Network, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.layer1 = torch.nn.Linear(VOCAB_SIZE, self.h1_size)
        self.nonlinearity1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(self.h1_size, self.h2_size)
        self.nonlinearity2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(self.h2_size, VOCAB_SIZE)

        torch.nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=VOCAB_SIZE)
        x = x.float()
        x = self.layer1(x)
        x = self.nonlinearity1(x)
        x = self.layer2(x)
        x = self.nonlinearity2(x)
        x = self.layer3(x)
        return x


model = Network()
# Because we use CrossEntropyLoss, we don't apply a Softmax on the final output of the model. It should be un-normalized.
loss_fn = torch.nn.CrossEntropyLoss()


if RELOAD_MODEL and os.path.exists(MODEL_SAVEPATH):
    print(f"Reloading model from {MODEL_SAVEPATH}")
    model.load_state_dict(torch.load(MODEL_SAVEPATH))
else:
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for iter in range(TRAIN_ITERS):
        batch_inputs, batch_outputs = make_batch()

        optimizer.zero_grad()

        pred = model(batch_inputs)

        loss = loss_fn(pred, batch_outputs)

        # compute gradients
        loss.backward()

        # apply loss
        optimizer.step()

        loss_scalar = loss.item() / BATCH_SIZE
        if iter % 10 == 0:
            print(f"{iter=:05} {loss_scalar=:.6f}")

    torch.save(model.state_dict(), MODEL_SAVEPATH)


## Evals
if EVAL:
    model.train(False)
    total_loss = 0
    with torch.no_grad():
        EVAL_ITERS = 100
        for _ in range(EVAL_ITERS):  # TODO: figure out how to not shuffle
            batch_inputs, batch_outputs = make_batch(test=True)
            pred = model(batch_inputs)
            loss = loss_fn(pred, batch_outputs)

            loss_scalar = loss.numpy().item()
            total_loss += loss_scalar

        total_loss /= (EVAL_ITERS * BATCH_SIZE)
        print(f"Test Loss: {total_loss=:.5f}")


## Generative Sampling (Inference)
if SAMPLE:
    model.train(False)

    result = STARTING_CHAR
    while len(result) - 1 < TOKENS_TO_SAMPLE:
        last_token = ctoi(result[-1])
        ltt = torch.tensor(last_token)
        ltt = ltt.unsqueeze(0)
        with torch.no_grad():
            prob_distribution = model(ltt)
            prob_distribution = torch.nn.functional.softmax(prob_distribution, dim=-1)
        prob_distribution = torch.squeeze(prob_distribution) / torch.sum(prob_distribution)
        sampled_token = np.random.choice([i for i in range(VOCAB_SIZE)], p=prob_distribution.numpy())  # TODO: multiple sampling and beam search or nucleus sampling
        result += itoc(sampled_token)

    print("Result:")
    print(result)
