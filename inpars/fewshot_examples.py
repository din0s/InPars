import json
import os
import random

import ftfy

random.seed(1)
import argparse

import pandas as pd
from tqdm.auto import tqdm


def load_examples(dataset_dir, split="test"):
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")

    print("Loading documents...")
    doc_df = pd.read_json(corpus_path, lines=True)
    # Convert _id to str
    doc_df["_id"] = doc_df["_id"].astype(str)
    # Concatenate title and text if title is not empty
    doc_df["text"] = doc_df.apply(
        lambda row: f"{row['title']} {row['text']}" if row["title"] else row["text"],
        axis=1,
    )
    # Fix text
    doc_df["text"] = doc_df["text"].apply(ftfy.fix_text)

    print("Loading queries...")
    query_df = pd.read_json(queries_path, lines=True)
    # Convert _id to str
    query_df["_id"] = query_df["_id"].astype(str)
    # Fix text
    query_df["text"] = query_df["text"].apply(ftfy.fix_text)

    print("Loading qrels...")
    qrel_df = pd.read_csv(
        qrels_path, sep="\t", header=None, names=["q_id", "doc_id", "score"]
    )
    # Convert q_id and doc_id to str
    qrel_df["q_id"] = qrel_df["q_id"].astype(str)
    qrel_df["doc_id"] = qrel_df["doc_id"].astype(str)

    print("Merging...")
    qrel_df = qrel_df.merge(query_df, left_on="q_id", right_on="_id").merge(
        doc_df, left_on="doc_id", right_on="_id"
    )
    # Keep query_id, doc_id, query, document
    qrel_df = qrel_df[["_id_x", "_id_y", "text_x", "text_y"]]

    return qrel_df.values.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", default=500_000)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = load_examples(args.dataset_dir, args.split)
    random.shuffle(examples)

    with open(args.output, "w") as f:
        for (q_id, doc_id, query, document) in tqdm(
            examples[: args.num_examples],
            total=args.num_examples,
            desc="Writing",
        ):
            line = {
                "query_id": q_id,
                "doc_id": doc_id,
                "query": query,
                "document": document,
            }
            f.write(json.dumps(line) + "\n")
