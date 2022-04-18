import datasets
import data_utils
import numpy as np

_, _, _, word_list = data_utils.load_embedding_file(
    "data/glove.6B.100d.txt", word_list_only=True
)
dataset = datasets.load_from_disk("data/preprocessed-word-only")
weights = data_utils.get_weights(dataset["train"], word_list, num_processes=32)

np.savetxt("data/weights_train.txt", weights, delimiter=",")
