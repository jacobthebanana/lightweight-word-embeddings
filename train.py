import math
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import datasets

import data_utils
from model import TransformerModel

# Hyperparameters
data_file = "./data/glove.6B.100d.txt"
input_dimension = 28
nhead = 2
nhid = 200
nlayers = 2
batch_size = 8192
n_epoch = 200
train_ratio = 0.5
test_ratio = 0.3
device = torch.device("cuda")
writer = SummaryWriter(filename_suffix=f"batch_{batch_size}")

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument(
    "--train_percentage", dest="train_percentage", type=float, default=1, required=False
)
args = parser.parse_args()

train_percentage = args.train_percentage
dataset_path = args.dataset_path
dataset = datasets.load_from_disk(dataset_path)

embedding_file_path = "data/glove.6B.100d.txt"
_, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
    embedding_file_path, limit=-1
)
output_dimension = embeddings.shape[1]

truncated_training_dataset = dataset["train"].train_test_split(
    train_size=train_percentage
)["train"]
train_dataloader = torch.utils.data.DataLoader(
    truncated_training_dataset, batch_size=batch_size
)
test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size)
num_steps_per_epoch = len(truncated_training_dataset) // batch_size

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    nhead=nhead,
    nhid=nhid,
    nlayers=nlayers,
    dropout=0.5,
)
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

for epoch in tqdm(range(n_epoch)):
    model.train()
    train_loss_tally = 0
    words_processed = 0
    step = 0

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

    train_loss_tally = train_loss_tally / words_processed
    writer.add_scalar("Loss/Train", train_loss_tally, epoch)

    model.eval()
    test_loss_tally = 0
    words_processed = 0
    for words in tqdm(test_dataloader):
        X, X_length_mask, Y = data_utils.prepare_batch(
            words, embeddings, word_lookup, device=device
        )

        with torch.no_grad():
            Y_predictions = model(X, length_mask=X_length_mask)
            loss = torch.sum(criterion(Y, Y_predictions, axis=0))
            test_loss_tally += loss
            words_processed += Y.shape[0]

    test_loss_tally = test_loss_tally / words_processed
    writer.add_scalar("Loss/Test", test_loss_tally, epoch)
    print(
        f"{epoch+1}: Train loss {train_loss_tally:.5f}, Test loss {test_loss_tally:.5f}"
    )

    torch.save(model.state_dict(), f"models/batch_{batch_size}_{epoch}.pth")
