import math
import argparse

import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import datasets
from torchinfo import summary

import data_utils
from model import TransformerModel

# Hyperparameters
embedding_file_path = "data/glove.6B.100d.txt"
input_dimension = 28
nhead = 64
nlayers_input = None
nlayers = 12
batch_size = 256
n_epoch = 200
train_ratio = 0.5
test_ratio = 0.3
device = torch.device("cuda")
limit = -1
seed = 599

parser = argparse.ArgumentParser()
parser.add_argument("weight_path")
args = parser.parse_args()

print("Preparing weights.")
weight_path = args.weight_path
weights = np.loadtxt(weight_path)

# Visit every word regardless of frequency in wikitext.
weights = np.maximum(weights, 1 * np.ones_like(weights))[:limit]

X, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
    embedding_file_path, limit=limit
)
output_dimension = embeddings.shape[1]

generator = random.Random(seed)
weight_word_sequence = list(zip(weights, word_list))
generator.shuffle(weight_word_sequence)
weights = []
word_list = []
for weight, word in weight_word_sequence:
    weights.append(weight)
    word_list.append(word)

train_length = int(len(word_list) * train_ratio)
test_length = int(len(word_list) * test_ratio)
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights[:train_length], int(len(word_list) * train_ratio)
)
test_sampler = torch.utils.data.WeightedRandomSampler(
    weights[train_length : train_length + test_length], int(len(word_list) * test_ratio)
)
train_dataloader = torch.utils.data.DataLoader(
    list(zip(X, word_list))[:train_length],
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=lambda x: x,
)
test_dataloader = torch.utils.data.DataLoader(
    list(zip(X, word_list))[train_length : train_length + test_length],
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=lambda x: x,
)

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    nhead=nhead,
    nlayers=nlayers,
    dropout=0.5,
)
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)

print(summary(model.to(device), [(1, 2, input_dimension)]))
print(f"Embedding dimension: {output_dimension}.")

writer = SummaryWriter(filename_suffix=f"batch_{batch_size}")
step = 0
for epoch in tqdm(range(n_epoch)):
    model.eval()
    test_loss_tally = 0
    words_processed = 0
    for words in tqdm(test_dataloader):
        X, X_length_mask, Y = data_utils.prepare_batch(
            words, embeddings, word_lookup, device=device
        )

        with torch.no_grad():
            Y_predictions = model(X, length_mask=X_length_mask)
            loss = criterion(Y, Y_predictions)
            test_loss_tally += loss
            words_processed += Y.shape[0]

    test_loss_tally = test_loss_tally / words_processed
    writer.add_scalar("Loss/Test", test_loss_tally, step)

    model.train()
    train_loss_tally = 0
    words_processed = 0

    for batch in tqdm(train_dataloader):
        step += 1
        X, X_length_mask, Y = data_utils.prepare_batch(
            batch, embeddings, word_lookup, device=device
        )

        Y_predictions = model(X, length_mask=X_length_mask)
        loss = criterion(Y, Y_predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tally += loss
        words_processed += Y.shape[0]

        episode_loss = loss / Y.shape[0]

        if step % 100 == 0:
            writer.add_scalar("Loss/Train (per episode)", episode_loss, step)

    writer.add_scalar("Loss/Train", train_loss_tally / words_processed, step)
    print(
        f"{epoch+1}: Train loss {train_loss_tally/words_processed:.5f}, Test loss {test_loss_tally:.5f}"
    )

    path = f"models/{output_dimension}d_batch_{batch_size}_{nlayers_input}_{nlayers}_layers_{nhead}_heads_step_{step}.pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
