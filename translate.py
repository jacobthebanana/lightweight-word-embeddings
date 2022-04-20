from typing import Tuple, List, Dict
import argparse

import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchinfo import summary

import data_utils
from model import TransformerModel, TranslatorModel

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


parser = argparse.ArgumentParser()
parser.add_argument("weight_path")
parser.add_argument("state_dict_path")
args = parser.parse_args()

print("Preparing weights.")
weight_path = args.weight_path
state_dict_path = args.state_dict_path
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
    hidden_dimension=hidden_dimension,
    nhead=nhead,
    nlayers=nlayers,
    dropout=0.5,
)

model = model.to(device)
model.load_state_dict(torch.load(state_dict_path))

translator = TranslatorModel(output_dimension, 1024).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(translator.parameters())

print(f"Embedding dimension: {output_dimension}.")


def prepare_batch(
    batch: List[Tuple[torch.tensor, str]],
    embeddings: torch.Tensor,
    word_lookup: Dict[str, int],
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper around get_batch.
    Input:
    - list of (tensor, word) pairs.
    - embeddings, torch matrix of embeddings, one row for each word (num_words, d)
    - word_lookup, dictionary mapping word (str) to row number in the embedding matrix.
    Output:
    - X_batch: 3D Tensor. (batch_size, l, 28)
    - X_batch_padding_mask: 2D Tensor. (batch_size, l).
    - Y_batch: 2D array. (batch_size, d).
    """
    X_items = []
    Y_items = []
    for X, word in batch:
        word_embedding_key = word_lookup.get(word)

        if word_embedding_key:
            word_tokenized = X
            word_embedding = embeddings[word_embedding_key, :]

            X_items.append(word_tokenized)
            Y_items.append(word_embedding)

    num_words = len(Y_items)
    embedding_dimension = embeddings.shape[1]
    Y = torch.zeros((num_words, embedding_dimension), device=device)
    for word_index, embedding in enumerate(Y_items):
        Y[word_index, :] = embedding

    return data_utils.get_batch(
        X_items, Y, start=0, batch_size=num_words, device=device
    )


writer = SummaryWriter(filename_suffix="translator")
step = 0
for epoch in tqdm(range(n_epoch)):
    model.eval()
    test_loss_tally = 0
    words_processed = 0
    for words in tqdm(test_dataloader):
        X, X_length_mask, Y = prepare_batch(
            words, embeddings, word_lookup, device=device
        )

        with torch.no_grad():
            transformer_embeddings = model(X, length_mask=X_length_mask)
            Y_predictions = translator(transformer_embeddings)
            loss = criterion(Y, Y_predictions)
            test_loss_tally += loss
            words_processed += Y.shape[0]

    test_loss_tally = test_loss_tally
    writer.add_scalar("Loss/Test", test_loss_tally, step)

    model.train()
    train_loss_tally = 0
    words_processed = 0

    for batch in tqdm(train_dataloader):
        step += 1
        X, X_length_mask, Y = prepare_batch(
            batch, embeddings, word_lookup, device=device
        )

        transformer_embeddings = model(X, length_mask=X_length_mask)
        Y_predictions = translator(transformer_embeddings)
        loss = criterion(Y, Y_predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tally += loss
        words_processed += Y.shape[0]

        episode_loss = loss

        if step % 10 == 0:
            writer.add_scalar("Loss/Train (per episode)", episode_loss, step)

    writer.add_scalar("Loss/Train", train_loss_tally, step)
    print(
        f"{epoch+1}: Train loss {train_loss_tally:.5f}, Test loss {test_loss_tally:.5f}"
    )

    path = f"models/translator_{step}_{hidden_dimension}.pth"
    torch.save(translator.state_dict(), path)
    print(f"Model saved to {path}")
