from typing import Tuple, List, Dict
import math
from collections import Counter
import random

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


def prepare_data(items):
    output = []
    window_size = 10

    sentences = items["text"]
    for sentence in sentences:
        words = sentence.split(" ")
        for word_index, current_word in enumerate(words):
            if not current_word.isalpha():
                continue

            current_word_output = [current_word]

            min_context_index = max(0, word_index - window_size)
            max_context_index = min(len(words), word_index + window_size)

            current_word_output.extend(words[min_context_index:max_context_index])

        output.append(" ".join(current_word_output))

    return {"data": output}


def prepare_batch(
    items: Dict[str, List[str]],
    word_lookup: Dict[str, int],
    vocabulary_size: int,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate skipgram inputs and references.
    - items["data"].
    - word_lookup: dictionary mapping words to indices.

    Output:
    - X: tokenized and padded words characters. (b x l x 28)
    - X_length_mask: length mask. (b x l)
    - Y: target. (b x n)
    """
    data_entries = items["data"]
    X = []
    Y = torch.zeros(len(data_entries), vocabulary_size)

    for data_index, data in enumerate(data_entries):
        data = data.split(" ")
        current_word = data[0]
        context_words = data[1:]

        word_tokenized = tokenize_word(current_word)
        X.append(word_tokenized)

        for context_word in context_words:
            context_word_key = word_lookup.get(context_word)
            if context_word_key:
                Y[data_index, context_word_key] = 1

    Y = Y.to(device)
    return get_batch(X, Y, start=0, batch_size=len(X), device=device)


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

    if len(X_sliced) == 0:
        return None, None, None

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

    if Y_batch is not None:
        Y_batch = Y_batch.to(device)

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
    dataset: List[Dict[str, str]],
    word_list: List[str],
    word_lookup: Dict[str, int],
    num_processes: int = 1,
) -> List[int]:
    """
    Input:
    - dataset, with each entry having a "word" as feature.
    - word_list, list of words.

    Output:
    - weight list, ordered as in word_list.
    """

    def process_slice(dataset_slice):
        output = np.zeros(len(word_list), dtype=np.int32)
        for word in dataset_slice["word"]:
            word_key = word_lookup.get(word)
            if word_key:
                output[word_key] += 1

        return {"count": [output]}

    weight_slices_dataset = dataset.map(
        process_slice,
        batched=True,
        remove_columns=["word"],
        num_proc=num_processes,
    )
    aggregated_weights = np.zeros(len(word_list), dtype=np.int32)
    for entry in tqdm(weight_slices_dataset):
        weight_slice = entry["count"]
        aggregated_weights += weight_slice

    return aggregated_weights


def generate_embeddings(
    model: torch.nn.Module,
    tokenized_vocabulary: List[torch.tensor],
    output_dimension: int,
    batch_size: int = 256,
    device: torch.device = None,
) -> torch.tensor:
    """
    Evaluate model on tokenized vocabulary to obtain
    new embeddings.
    """
    num_words = len(tokenized_vocabulary)
    output = torch.empty((num_words, output_dimension), device=device)

    num_iterations = math.ceil(num_words / batch_size)

    model.eval()
    for iteration in tqdm(range(num_iterations)):
        head_index = batch_size * iteration
        tail_index = min(num_words, (head_index + batch_size))
        X_batch, X_length_mask, _ = get_batch(
            tokenized_vocabulary,
            None,  # No embedding is provided during inference.
            start=head_index,
            batch_size=batch_size,
            device=device,
        )

        with torch.no_grad():
            Y_predictions = model(X_batch, length_mask=X_length_mask)
            output[head_index:tail_index, :] = Y_predictions

    return output


def negative_sample(
    weights: List[int], positive: List[int], item_count: int, generator: random.Random
) -> List[int]:
    """
    Negative sample.
    """
    items = []
    num_choices = len(weights)
    while len(items) < item_count:
        item = generator.choices(range(num_choices), weights=weights)[0]

        if (item not in positive) and (item not in items):
            items.append(item)

    return items


def get_negative_sample_loss(
    output: torch.tensor, positive: List[int], negative: List[int]
) -> Tuple[torch.tensor, torch.tensor]:
    all_relevant_indices = positive + negative
    device = output.device
    new_output = torch.zeros(len(all_relevant_indices), requires_grad=True, device=device)
    reference = torch.zeros(len(all_relevant_indices), requires_grad=True, device=device)
    for index in positive:
        new_output[index] = output[index]
        reference[index] = 1
    for index in negative:
        new_output[index] = output[index]
        reference[index] = 0

    return new_output, reference
