"""
Given a word embedding file in the GloVe format 
and word analogy list "questions-words.txt",
return the MSE between expected output and actual output.
"""
import argparse
from typing import List, Tuple

import torch
from torch.nn import MSELoss
from tqdm.auto import tqdm

import data_utils

parser = argparse.ArgumentParser("Evaluate embedding quality.")
parser.add_argument("embedding_file")
parser.add_argument("question_words_list")
parser.add_argument("--open", dest="is_open", type=bool, default=False, required=False)
args = parser.parse_args()
embedding_file_path = args.embedding_file
question_words_path = args.question_words_list
is_open = args.is_open

device = torch.device("cuda")

if not is_open:
    question_words = set()
    with open(question_words_path, "r") as question_words_file:
        for line in question_words_file.readlines():
            for word in line.split(" "):
                question_words.add(word.lower())

    _, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
        embedding_file_path,
        device=device,
        words=question_words,
    )
else:
    _, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
        embedding_file_path,
        device=device,
    )

embeddings_norms = torch.norm(embeddings, p=2, dim=0)
embeddings_norm_mean = torch.mean(embeddings_norms)

# List of Tuples, with each tuple including the name of the task,
# followed by a list of tuples like
# (Entity_A_X, Entity_A_Y, Entity_B_X, Entity_B_Y)
analogy_tasks: List[Tuple[List[Tuple[str, str, str, str]]]] = []
tqdm_bar = tqdm()
with open(question_words_path, "r") as question_words_file:
    line = question_words_file.readline()
    task_name = line[2:-2]
    list_of_challenges: List[Tuple[str, str, str, str]] = []
    while line:
        if line[0] == ":":
            if list_of_challenges:
                task = (task_name, list_of_challenges)
                analogy_tasks.append(task)

            task_name = line[2:-1]
            list_of_challenges: List[Tuple[str, str, str, str]] = []
        else:
            challenge_pair = tuple(line[:-1].lower().split(" "))
            assert len(challenge_pair) == 2 + 2, challenge_pair
            list_of_challenges.append(challenge_pair)

        tqdm_bar.update(1)
        line = question_words_file.readline()

tqdm_bar.close()
criterion = MSELoss()
missing_words = []
incorrect_pairs = []

for task_name, task_pairs in tqdm(analogy_tasks):
    task_loss_tally = 0
    num_tasks_completed = 0
    num_tasks_correct = 0
    for words in task_pairs:
        embedding_lookup_keys = list(map(word_lookup.get, words))
        task_embeddings = []
        for word_index, embedding_lookup_key in enumerate(embedding_lookup_keys):
            if not embedding_lookup_key:
                missing_word = words[word_index]
                missing_words.append(missing_word)
            else:
                embedding = embeddings[embedding_lookup_key, :]
                task_embeddings.append(embedding)

        # No missing word in this pair.
        if len(task_embeddings) == len(words):
            A_X, A_Y, B_X, B_Y = task_embeddings
            A_difference = A_X - A_Y
            B_difference = B_X - B_Y

            task_loss_tally += (
                1e5 * criterion(A_difference, B_difference) / embeddings_norm_mean
            )
            num_tasks_completed += 1

            # A_X should be similar to A_Y + (B_X - B_Y)
            # (Athen - Greece) should be similar (Paris - France)
            A_X_prediction = A_Y + (B_X - B_Y)
            A_X_predicted_index, _ = data_utils.get_nearest(A_X_prediction, embeddings)
            A_X_real_index = embedding_lookup_keys[0]

            if A_X_real_index == A_X_predicted_index:
                num_tasks_correct += 1
            else:
                A_X_predicted_word = word_list[A_X_predicted_index]
                pair = (task_name, *words, A_X_predicted_word)
                incorrect_pairs.append(pair)

    task_loss_tally = task_loss_tally / num_tasks_completed
    print(
        f"{task_name}: {task_loss_tally:.5f} ({num_tasks_correct}/{num_tasks_completed})"
    )

for incorrect_pair in incorrect_pairs:
    print(" ".join(incorrect_pair))
