# Adapted from RetroMAE
# https://github.com/staoxiao/RetroMAE/blob/master/examples/retriever/BEIR/preprocess.py
import argparse
import os
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tok_data")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--use_title", action="store_true")
    return parser.parse_args()


def to_tokens(examples):
    if "title" in examples and args.use_title:
        content = []
        for title, text in zip(examples["title"], examples["text"]):
            content.append(title + tokenizer.sep_token + text)
    else:
        content = examples["text"]
    return tokenize_fn(content)


args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
tokenize_fn = partial(
    tokenizer,
    add_special_tokens=False,
    truncation=True,
    max_length=args.max_seq_length,
    return_attention_mask=False,
    return_token_type_ids=False,
)

if __name__ == "__main__":
    corpus_path = os.path.join(args.output_dir, "corpus")
    queries_path = os.path.join(args.output_dir, "queries")
    Path(corpus_path).mkdir(parents=True, exist_ok=True)
    Path(queries_path).mkdir(parents=True, exist_ok=True)

    corpus = load_dataset("json", data_files=args.corpus, split="train")
    corpus = corpus.map(
        to_tokens,
        num_proc=args.threads,
        remove_columns=["title", "text", "metadata"],
        batched=True,
    )
    corpus.save_to_disk(corpus_path)
    print("corpus dataset:", corpus)
    with open(os.path.join(corpus_path, "mapping_id.txt"), "w") as f:
        for idx, _id in enumerate(corpus["_id"]):
            f.write(f"{_id}\t{idx}\n")

    queries = load_dataset("json", data_files=args.queries, split="train")
    queries = queries.map(
        to_tokens,
        num_proc=args.threads,
        remove_columns=["text"],
        batched=True,
    )
    queries.save_to_disk(queries_path)
    print("query dataset:", queries)
    with open(os.path.join(queries_path, "mapping_id.txt"), "w") as f:
        for idx, _id in enumerate(queries["id"]):
            f.write(f"{_id}\t{idx}\n")
