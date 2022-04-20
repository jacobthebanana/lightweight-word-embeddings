import datasets
import data_utils
import numpy as np

_, _, word_lookup, word_list = data_utils.load_embedding_file(
    "data/glove.6B.100d.txt", word_list_only=True
)
dataset = datasets.load_from_disk("data/preprocessed-word-only")
weights = data_utils.get_weights(
    dataset["train"], word_list, word_lookup, num_processes=8
)

np.savetxt("data/weights_train.txt", weights, delimiter=",")
