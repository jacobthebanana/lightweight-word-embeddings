from typing import Tuple, List
import math

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
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Input: path to an embedding file.

    Output:
    - list of code point vectors of the input words
    (lower-cased, with "a" being 1) and all non-letter tokens
    replaced with 0. List[(*, 28)] of length n.
    - array of embedding vectors. Shape (n, d).
    - device (optional): device for storing the embedding tensor.
    """
    with open(path, "r") as embedding_text_file:
        X = []
        embedding_lines = embedding_text_file.readlines()[:limit]
        embedding_dim = len(embedding_lines[0].split()) - 1
        embeddings = torch.empty((len(embedding_lines), embedding_dim)).to(device)

        for line_id, embedding_line in tqdm(
            enumerate(embedding_lines), total=len(embedding_lines)
        ):
            embedding_line = embedding_line.split(" ")
            word = embedding_line[0]
            word = tokenize_word(word)
            word = word
            X.append(word)

            if not word_list_only:
                embedding = torch.tensor(list(map(float, embedding_line[1:]))).to(
                    device
                )
                embeddings[line_id, :] = embedding

        del embedding_lines

    return X, embeddings


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
    start = 0

    X_sliced = X[start : (start + batch_size)]
    Y_batch = Y[start : (start + batch_size), :]
    actual_batch_length = len(X_sliced)

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
