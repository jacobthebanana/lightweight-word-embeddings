import math

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

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


X, embeddings = data_utils.load_embedding_file(data_file, device=device, limit=-1)
X, embeddings = data_utils.shuffle(X, embeddings)
X_train, Y_train, X_test, Y_test = data_utils.split(X, embeddings, train_ratio)

output_dimension = embeddings.shape[1]
num_words = X.shape[0]

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
    num_iterations = math.ceil((num_words * train_ratio) / batch_size)

    model.train()
    train_loss_tally = 0
    for iteration in range(num_iterations):
        start_index = batch_size * iteration
        X_batch, X_length_mask, Y_batch = data_utils.get_batch(
            X_train,
            Y_train,
            start=start_index,
            batch_size=batch_size,
            device=device,
        )

        Y_predictions = model(X_batch, length_mask=X_length_mask)
        loss = criterion(Y_batch, Y_predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_tally += loss

    writer.add_scalar("Loss/Train", train_loss_tally, epoch)

    num_iterations = math.ceil((num_words * test_ratio) / batch_size)

    model.eval()
    test_loss_tally = 0
    for iteration in range(num_iterations):
        start_index = batch_size * iteration
        X_batch, X_length_mask, Y_batch = data_utils.get_batch(
            X_test,
            Y_test,
            start=start_index,
            batch_size=batch_size,
            device=device,
        )

        with torch.no_grad():
            Y_predictions = model(X_batch, length_mask=X_length_mask)
            loss = criterion(Y_batch, Y_predictions)
            test_loss_tally += loss

    writer.add_scalar("Loss/Test", test_loss_tally, epoch)
    print(
        f"{epoch+1}: Train loss {train_loss_tally:.5f}, Test loss {test_loss_tally:.5f}"
    )

    if epoch == n_epoch or epoch % 9 == 0:
        torch.save(model.state_dict(), f"models/batch_{batch_size}_{epoch}.pth")
