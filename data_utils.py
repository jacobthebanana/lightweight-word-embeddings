from typing import Tuple, List
import torch
from tqdm.auto import tqdm


def tokenize_word(word: str):
    """
    Input: word (str, n characters).
    Output: Array (n, 27)
    """
    word = word.lower()
    output = torch.zeros((len(word), 27))
    for index, character in enumerate(word):
        character_ascii = ord(character)
        character_codepoint = character_ascii - ord("a") + 1

        if 1 <= character_codepoint <= ord("z") - ord("a") + 1:
            output[index][character_codepoint] = 1
        else:
            output[index][0] = 1

    return output


def load_embedding_file(path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Input: path to an embedding file.

    Output:
    - list of code point vectors of the input words
    (lower-cased, with "a" being 1) and all non-letter tokens
    replaced with 0. List[(*, 27)] of length n.
    - array of embedding vectors. Shape (n, d).
    """
    device = torch.device("cuda")

    with open(path, "r") as embedding_text_file:
        X = []
        embedding_lines = embedding_text_file.readlines()
        embedding_dim = len(embedding_lines[0].split()) - 1
        embeddings = torch.empty((len(embedding_lines), embedding_dim)).to(device)

        for line_id, embedding_line in tqdm(
            enumerate(embedding_lines), total=len(embedding_lines)
        ):
            embedding_line = embedding_line.split(" ")
            word = embedding_line[0]
            word = tokenize_word(word)
            X.append(word)
            embedding = torch.tensor(list(map(float, embedding_line[1:]))).to(device)
            embeddings[line_id, :] = embedding

        del embedding_lines

    return X, embeddings


def get_batch(
    X: List[torch.Tensor], Y: torch.Tensor, start: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input:
    - X: List of n 2D tensors (l_j, 27), where l_j is at most l.
    - Y: 2D Array. (n, d)
    - start: starting position. (int)
    - batch_size: number of elements to include. (int)

    Output:
    - X_batch: 3D Tensor. (batch_size, l, 27)
    - X_batch_mask: 3D Tensor (a number of 2D square matrices). (batch_size, l, l).
    - Y_batch: 2D array. (batch_size, d).
    """
    start = 0
    batch_size = 2

    X_sliced = X[start : (start + batch_size)]
    Y_batch = Y[start : (start + batch_size), :]

    embedding_dimension = X_sliced[0].shape[-1]
    max_word_length = max(map(len, X_sliced))
    X_batch = torch.zeros(
        (batch_size, max_word_length, embedding_dimension)
    )  # "torch.empty" would also work.
    X_batch_mask = torch.full(
        (batch_size, max_word_length, max_word_length), float("-inf")
    )

    for word_index, word in enumerate(X_sliced):
        word_length = word.shape[0]
        X_batch_mask[word_index, :word_length, :word_length] = 0

        for character_index, character in enumerate(word):
            X_batch[word_index, character_index, :] = character

    return X_batch, X_batch_mask, Y_batch
