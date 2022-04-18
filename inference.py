import math
import argparse

import torch
from tqdm.auto import tqdm

from model import TransformerModel
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument("word_list_file")
parser.add_argument("model_path")
parser.add_argument("nhead", type=int)
parser.add_argument("nlayers_input", type=int)
parser.add_argument("nlayers", type=int)
parser.add_argument("output_dimension", type=int)
args = parser.parse_args()

data_file = args.word_list_file
state_dict_path = args.model_path

input_dimension = 28
output_dimension = args.output_dimension
nhead = args.nhead
nlayers_input = args.nlayers_input
nlayers = args.nlayers
batch_size = 256
device = torch.device("cuda")

# List of word-character matricies.
X, _, _, word_list = data_utils.load_embedding_file(data_file, device=device, limit=-1)
num_words = len(word_list)

model = TransformerModel(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    nhead=nhead,
    nlayers_input=nlayers_input,
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
        X,
        None,  # No embedding is provided during inference.
        start=head_index,
        batch_size=batch_size,
        device=device,
    )

    with torch.no_grad():
        Y_predictions = model(X_batch, length_mask=X_length_mask)
        output[head_index:tail_index, :] = Y_predictions


output = output.cpu().numpy()
with open(f"{args.model_path}.predictions.{output_dimension}d.txt", "w") as output_file:
    # Iterate over rows (words) of the 2D numpy array "output".
    for word_index, word_embedding in enumerate(tqdm(output)):
        word: str = word_list[word_index]
        word_embedding = tuple(map(str, word_embedding))
        word_embedding = " ".join(word_embedding)

        output_line = f"{word} {word_embedding}\n"
        output_file.write(output_line)
