from typing import Tuple, List, Dict
import math
from collections import Counter

import torch
from tqdm.auto import tqdm
import numpy as np
from fast_transformers.masking import LengthMask


def tokenize_word(word: str):
    """
    Input: word (str, n characters).
    Output: Array with [CLS] (n + 1, 28)
    """
    word = word.lower()
    output = torch.zeros((len(word) + 1, 28))
    for index, character in enumerate(word):
        character_ascii = ord(character)
        character_codepoint = character_ascii - ord("a") + 1

        output[0][27] = 1

        if 1 <= character_codepoint <= ord("z") - ord("a") + 1:
            output[index + 1][character_codepoint] = 1
        else:
            output[index + 1][0] = 1

    return output


def load_embedding_file(
    path: str,
    device: torch.device = None,
    limit: int = -1,
    word_list_only: bool = False,  # Load only the list of input words.
    words: set = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, int], List[str]]:
    """
    Input: path to an embedding file.
    - device (optional): device for storing the embedding tensor.

    Output:
    - list of code point vectors of the input words
    (lower-cased, with "a" being 1) and all non-letter tokens
    replaced with 0. List[(*, 28)] of length n.
    - array of embedding vectors. Shape (n, d).
    - dictionary, mapping word to row number in the embedding matrix.
    - list of all words in the embedding vocabulary.
    """
    with open(path, "r") as embedding_text_file:
        word_lookup = {}
        X = []
        vocabulary = []
        embedding_lines = embedding_text_file.readlines()[:limit]
        embedding_dim = len(embedding_lines[0].split()) - 1
        embeddings = []
        line_id = 0

        for embedding_line in tqdm(embedding_lines):
            embedding_line = embedding_line.split(" ")
            word = embedding_line[0]

            if (not words) or (word in words):
                word_lookup[word] = line_id
                vocabulary.append(word)

                if not word_list_only:
                    tokenized_word = tokenize_word(word)
                    X.append(tokenized_word)

                    embedding = torch.tensor(list(map(float, embedding_line[1:])))
                    embeddings.append(embedding)

                line_id += 1

        embeddings_matrix = torch.zeros((len(embeddings), embedding_dim))
        for line_id, embedding in enumerate(tqdm(embeddings)):
            embeddings_matrix[line_id, :] = embedding

        embeddings_matrix = embeddings_matrix.to(device)
        del embedding_lines

    return X, embeddings_matrix, word_lookup, vocabulary


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

    return get_batch(X_items, Y, start=0, batch_size=num_words, device=device)


def get_batch(
    X: List[torch.Tensor],
    Y: torch.Tensor,
    start: int,
    batch_size: int,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input:
    - X: List of n 2D tensors (l_j, 28), where l_j is at most l.
    - Y: 2D Array. (n, d)
    - start: starting position. (int)
    - batch_size: number of elements to include. (int)

    Output:
    - X_batch: 3D Tensor. (batch_size, l, 28)
    - X_batch_padding_mask: 2D Tensor. (batch_size, l).
    - Y_batch: 2D array. (batch_size, d).
    """
    X_sliced = X[start : (start + batch_size)]
    actual_batch_length = len(X_sliced)
    if Y is not None:
        Y_batch = Y[start : (start + batch_size), :]
    else:  # Inference mode.
        Y_batch = None

    embedding_dimension = X_sliced[0].shape[-1]
    max_word_length = max(map(len, X_sliced))

    # Must properly initialize, despite the masking.
    X_batch = torch.zeros((actual_batch_length, max_word_length, embedding_dimension))
    X_word_lengths = torch.empty((actual_batch_length,))

    for word_index, word in enumerate(X_sliced):
        word_length = word.shape[0]
        X_word_lengths[word_index] = word_length

        for character_index, character in enumerate(word):
            X_batch[word_index, character_index, :] = character

    X_batch = X_batch.to(device)
    X_batch_mask = LengthMask(X_word_lengths, device=device)
    return X_batch, X_batch_mask, Y_batch


def shuffle(
    X: List[torch.Tensor], Y: torch.Tensor, seed: int = 599, device=None
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Input:
    - X: list of n tensors of variable lengths.
    - Y: Tensor. (n, d)

    Output:
    - X_shuffled: list of n Tensors on specified device.
    - Y_shuffled: Tensor. (n, d)
    """
    rng = np.random.default_rng(seed)
    randomized_indices = rng.permutation(len(X))

    X_shuffled = np.array(X, dtype=object)
    X_shuffled = X_shuffled[randomized_indices]
    Y_shuffled = Y[randomized_indices, :]

    print("Transferring X data to device.")
    for index in tqdm(range(len(X))):
        X_shuffled[index] = X_shuffled[index].to(device)

    return X_shuffled, Y_shuffled


def split(
    X: List[torch.Tensor], Y: torch.Tensor, ratio: float
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Input:
    - X: list of n tensors of variable lengths.
    - Y: Tensor. (n, d)
    - ratio: float between 0 and 1.

    Output:
    - X_a: list of math.ceil(n * ratio) Tensors.
    - Y_a: Tensor. (math.ceil(n * ratio), d)
    - X_b: list of math.floor(n * (1 - ratio)) Tensors.
    - Y_b: Tensor. (math.floor(n * (1 - ratio)), d)
    """
    num_a_items = math.ceil(len(X) * ratio)
    num_b_items = math.floor(len(X) * (1 - ratio))
    X_a = X[0:num_a_items]
    Y_a = Y[0:num_a_items, :]
    X_b = X[num_a_items:-1]
    Y_b = Y[num_a_items:-1, :]

    return X_a, Y_a, X_b, Y_b


def get_nearest(
    query: torch.Tensor, embeddings: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """
    Input:
    - a query vector: (n, 1)
    - a matrix of vectors arranged in rows: (m, n)

    Output:
    - row number of the vector nearest to the query vector: integer
    - vector nearest to the query vector: (n, 1).
    """
    query_norm = torch.norm(query, p=2)
    embeddings_norm = torch.norm(embeddings, p=2, dim=1)

    product = torch.matmul(embeddings, query) / (query_norm * embeddings_norm)
    nearest_index = torch.argmax(product)
    nearest_word = embeddings[nearest_index, :]

    return nearest_index, nearest_word


def process_wikitext(examples, word_lookup: Dict[str, int]):
    """
    Input:
    - examples["text"] (List[str]) of n total words.
    - word_lookup, key lookup matrix of all embedding words.

    Output:
    - output["words"], list of words that are in the word lookup dictionary.

    Tokens that are not in the word_lookup dictionary are skipped.
    """
    input_text = " ".join(examples["text"])
    input_words = input_text.split(" ")
    word_list = []

    for word in input_words:
        if word in word_lookup.keys():
            word_list.append(word)

    output = {"word": word_list}
    return output


def get_weights(
    dataset: List[Dict[str, str]], word_list: List[str], num_processes: int = 1
) -> List[int]:
    """
    Input:
    - dataset, with each entry having a "word" as feature.
    - word_list, list of words.

    Output:
    - weight list, ordered as in word_list.
    """

    def process_slice(dataset_slice):
        word_counter = Counter()
        output: List[int] = []
        for word in dataset_slice["word"]:
            word_counter[word] += 1

        for word in word_list:
            count = word_counter[word]
            output.append(count)

        return {"count": [np.array(output)]}

    weight_slices_dataset = dataset.map(
        process_slice,
        batched=True,
        remove_columns=["word"],
        num_proc=num_processes,
    )
    weight_slices = []
    for entry in tqdm(weight_slices_dataset):
        weight_slices.append(entry["count"])

    weight_slices = np.array(weight_slices)
    weights = np.sum(weight_slices, axis=0)

    return weights
