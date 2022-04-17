import math

import torch
from tqdm.auto import tqdm

from model import TransformerModel
import data_utils

data_file = "./data/glove.6B.100d.txt"
input_dimension = 28
output_dimension = 100
nhead = 2
nhid = 200
nlayers = 2
state_dict_path = f"models/batch_8192_189.pth"

batch_size = 8192
device = torch.device("cuda")

# List of word-character matricies.
word_list, _ = data_utils.load_embedding_file(data_file, device=device, limit=-1)
num_words = len(word_list)

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    nhead=nhead,
    nhid=nhid,
    nlayers=nlayers,
    dropout=0.5,
)

model = model.to(device)
model.load_state_dict(torch.load(state_dict_path))


output = torch.empty((num_words, output_dimension), device=device)
num_iterations = math.ceil(num_words / batch_size)

model.eval()
test_loss_tally = 0
for iteration in tqdm(range(num_iterations)):
    head_index = batch_size * iteration
    tail_index = min(num_words, (head_index + batch_size))
    X_batch, X_length_mask, _ = data_utils.get_batch(
        word_list,
        None,  # No embedding is provided during inference.
        start=head_index,
        batch_size=batch_size,
        device=device,
    )

    with torch.no_grad():
        Y_predictions = model(X_batch, length_mask=X_length_mask)
        output[head_index:tail_index, :] = Y_predictions


output = output.cpu().numpy()
with open(f"data/predictions.{output_dimension}d.txt", "w") as output_file:
    # Iterate over rows (words) of the 2D numpy array "output".
    for word_index, word_embedding in enumerate(tqdm(output)):
        word: str = word_list[word_index]
        word_embedding = tuple(map(str, word_embedding))
        word_embedding = " ".join(word_embedding)

        output_line = f"word {word_embedding}\n"
        output_file.write(output_line)
