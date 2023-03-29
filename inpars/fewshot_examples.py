import csv
import json
import os
import random

import ftfy

random.seed(1)
import argparse

from tqdm.auto import tqdm


def load_examples(dataset_dir, split="test"):
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")

    documents = {}
    with open(corpus_path, "r") as f:
        n_docs = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=n_docs, desc="Loading documents"):
            doc = json.loads(line)
            text = (
                f"{doc['title']} {doc['text']}"
                if "title" in doc and doc["title"]
                else doc["text"]
            )
            documents[doc["_id"]] = ftfy.fix_text(text)

    queries = {}
    with open(queries_path, "r") as f:
        n_queries = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=n_queries, desc="Loading queries"):
            query = json.loads(line)
            queries[query["_id"]] = ftfy.fix_text(query["text"])

    qrels = []
    with open(qrels_path, "r") as f:
        n_qrels = sum(1 for _ in f)
        f.seek(0)
        qrel_reader = csv.reader(f, delimiter="\t", lineterminator="\n")
        for qrel in tqdm(qrel_reader, total=n_qrels, desc="Loading qrels"):
            q_id, doc_id, _ = qrel
            qrels.append((queries[q_id], documents[doc_id]))

    return qrels


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
        for (query, document) in tqdm(
            examples[: args.num_examples], total=args.num_examples, desc="Writing"
        ):
            f.write(json.dumps({"query": query, "document": document}) + "\n")
