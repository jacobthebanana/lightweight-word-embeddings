import math
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import datasets
from torchinfo import summary

import data_utils
from model import TransformerModel

# Hyperparameters
embedding_file_path = "data/glove.6B.300d.txt"
input_dimension = 28
nhead = 24
nlayers = 12
batch_size = 256
n_epoch = 200
train_ratio = 0.5
test_ratio = 0.3
device = torch.device("cuda")
writer = SummaryWriter(filename_suffix=f"batch_{batch_size}")

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()

dataset_path = args.dataset_path
dataset = datasets.load_from_disk(dataset_path)

_, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
    embedding_file_path, limit=-1
)
output_dimension = embeddings.shape[1]

print("Preparing dataset.")
train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size)

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    nhead=nhead,
    nlayers=nlayers,
    dropout=0.5,
)
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

print(summary(model.to(device), [(1, 2, input_dimension)]))
print(f"Embedding dimension: {output_dimension}.")

step = 0
for epoch in tqdm(range(n_epoch)):
    model.train()
    train_loss_tally = 0
    words_processed = 0

    for words in tqdm(train_dataloader):
        step += 1
        X, X_length_mask, Y = data_utils.prepare_batch(
            words, embeddings, word_lookup, device=device
        )

        Y_predictions = model(X, length_mask=X_length_mask)
        loss = criterion(Y, Y_predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tally += loss
        words_processed += Y.shape[0]

        episode_loss = loss / Y.shape[0]
        writer.add_scalar("Loss/Train (per episode)", episode_loss, step)

        if step % 1000 == 0:
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

            model.train()
            test_loss_tally = test_loss_tally / words_processed
            writer.add_scalar("Loss/Test", test_loss_tally, step)
            print(
                f"{epoch+1}: Train loss {train_loss_tally:.5f}, Test loss {test_loss_tally:.5f}"
            )

            path = (
                f"models/batch_{batch_size}_{step}_{nlayers}_layers_{nhead}_heads.pth"
            )
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")
