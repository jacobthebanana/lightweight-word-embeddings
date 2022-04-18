import argparse
import os

import datasets

import data_utils

parser = argparse.ArgumentParser()
parser.add_argument("embedding_file")
parser.add_argument("output_folder")
parser.add_argument("-j", dest="num_proc", type=int, default=1, required=False)
args = parser.parse_args()

embedding_file_path = args.embedding_file
output_path = args.output_folder
assert os.path.isdir(output_path)

num_proc = int(args.num_proc)

_, embeddings, word_lookup, word_list = data_utils.load_embedding_file(
    embedding_file_path
)


def prepare_wikitext(examples):
    return data_utils.process_wikitext(examples, word_lookup)


dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")
processed_dataset = dataset.map(
    prepare_wikitext,
    batched=True,
    remove_columns=["text"],
    num_proc=num_proc,
    batch_size=256,
)
processed_dataset.save_to_disk(output_path)
