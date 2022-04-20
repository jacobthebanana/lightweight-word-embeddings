import math
import argparse
import os

import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import datasets
from torchinfo import summary

import data_utils
from model import TransformerModel

torch.manual_seed(599)

# Hyperparameters
embedding_file_path = "data/glove.6B.100d.txt"
input_dimension = 28
nhead = 24
hidden_dimension = 1024
nlayers = 6
batch_size = 256
evaluation_batch_size = 1024
train_epoch_size = batch_size * 256
test_epoch_size = evaluation_batch_size * 16
n_epoch = 200
train_ratio = 0.5
test_ratio = 0.3
embedding_update_rate = 0.1
negative_samples = 5
device = torch.device("cuda")
limit = -1
seed = 599

weights = np.loadtxt("data/Archive/weights_train.txt")

parser = argparse.ArgumentParser()
train_ngram_ds = datasets.load_from_disk("data/ngram_train")
test_ngram_ds = datasets.load_from_disk("data/ngram_test")

print(train_ngram_ds)
print(test_ngram_ds)

(_, _, word_lookup, word_list,) = data_utils.load_embedding_file(
    embedding_file_path, limit=limit, device=device, word_list_only=True
)
output_dimension = 100
assert os.path.isdir(f"data/output/")

print("Preparing dataset.")
train_sampler = torch.utils.data.RandomSampler(
    train_ngram_ds, replacement=True, num_samples=train_epoch_size
)
train_dataloader = torch.utils.data.DataLoader(
    train_ngram_ds, batch_size=batch_size, sampler=train_sampler
)

test_sampler = torch.utils.data.RandomSampler(
    test_ngram_ds, replacement=True, num_samples=test_epoch_size
)
test_dataloader = torch.utils.data.DataLoader(
    test_ngram_ds, batch_size=evaluation_batch_size, sampler=test_sampler
)

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    hidden_dimension=hidden_dimension,
    nhead=nhead,
    nlayers=nlayers,
    dropout=0.5,
)
model = model.to(device)
metric = torch.nn.BCEWithLogitsLoss()

print(summary(model.to(device), [(1, 2, input_dimension)]))
print(f"Embedding dimension: {output_dimension}.")
writer = SummaryWriter(
    filename_suffix=f"{output_dimension}d_"
    + f"batch_{batch_size}_"
    + f"{hidden_dimension}_hidden_"
    + f"{nlayers}_layers_"
    + f"{nhead}_heads"
)

step = 0
words_processed = 0

output_layer = torch.nn.Linear(output_dimension, len(word_list)).to(device)
optimizer = torch.optim.AdamW([*model.parameters(), *output_layer.parameters()])

for epoch in tqdm(range(n_epoch)):
    model.train()
    train_loss_tally = 0
    words_processed_epoch = 0

    for words in tqdm(train_dataloader):
        step += 1
        X, X_length_mask, Y = data_utils.prepare_batch(
            words, word_lookup, len(word_list), device=device
        )

        if X is None:
            continue

        word_vector_prediction = model(X, length_mask=X_length_mask)
        word_vector_prediction = word_vector_prediction.to(device)

        # Context prediction
        Y_predictions = output_layer(word_vector_prediction)
        negative_samples = []
        loss = metric(Y_predictions, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tally += loss
        words_processed += Y.shape[0]
        words_processed_epoch += Y.shape[0]

        episode_loss = loss

        if step % 10 == 0:
            writer.add_scalar("Loss/Train (per episode)", episode_loss, step)
            writer.add_scalar("Tokens/Processed (train)", words_processed, step)

    print(f"Epoch {epoch}: Testing model on step {step}.")
    model.eval()
    test_loss_tally = 0
    words_processed_test = 0
    for sentences in tqdm(test_dataloader):
        X, X_length_mask, Y = data_utils.prepare_batch(
            sentences, word_lookup, len(word_list), device=device
        )
        if X is None:
            continue

        with torch.no_grad():
            word_vector_prediction = model(X, length_mask=X_length_mask)
            word_vector_prediction = word_vector_prediction.to(device)
            Y_predictions = output_layer(word_vector_prediction)
            loss = metric(Y_predictions, Y)

            test_loss_tally += loss
            words_processed_test += X.shape[0]

    writer.add_scalar("Loss/Test", test_loss_tally, epoch)
    writer.add_scalar("Loss/Train", train_loss_tally, epoch)
    writer.add_scalar("Tokens/Processed (test)", words_processed_test, epoch)

    print(
        f"{epoch+1}: Train loss {train_loss_tally:.5f}, Test loss {test_loss_tally:.5f}"
    )

    path = (
        f"models/{output_dimension}d_"
        + f"batch_{batch_size}_"
        + f"{hidden_dimension}_hidden_"
        + f"{nlayers}_layers_"
        + f"{nhead}_heads_"
        + f"step_{step}_skipgram_a.pth"
    )
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
