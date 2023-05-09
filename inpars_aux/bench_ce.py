"""Benchmarks a trained biencoder on the training qrel file."""
import argparse
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--corpus", type=argparse.FileType("r"), required=True)
    parser.add_argument("--queries", type=argparse.FileType("r"), required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_hits", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    q_triples = []
    queries = pd.read_json(args.queries, lines=True, dtype={"query_id": str})
    with open("./topics.tsv", "w") as f:
        for i, row in enumerate(queries.itertuples()):
            query = row.query.replace("\n", " ")
            f.write(f"{i}\t{query}\n")

    queries.set_index("query_id", inplace=True)

    # fmt: off
    subprocess.run([
        "python", "-m", "pyserini.search.lucene",
        "--threads", str(args.threads),
        "--batch-size", str(args.batch_size),
        "--hits", str(args.max_hits),
        "--index", args.index,
        "--topics", "./topics.tsv",
        "--output", "./run.trec",
        "--bm25",
    ])
    # fmt: on

    os.remove("./topics.tsv")

    results = defaultdict(list)
    with open("./run.trec", "r") as f:
        for line in f:
            q_idx, _, doc_id, _, _, _ = line.split()
            results[q_idx].append(doc_id)

    os.remove("./run.trec")

    model = CrossEncoder(args.model_path)

    corpus = pd.read_json(args.corpus, lines=True)
    corpus.set_index("_id", inplace=True)

    mrrs = []
    for q_idx, doc_ids in tqdm(results.items(), desc="Reranking"):
        mrr = 0
        q = queries.iloc[int(q_idx)]
        q_list = [q["query"]] * len(doc_ids)
        d_list = corpus.loc[doc_ids]["text"].tolist()
        scores = model.predict(list(zip(q_list, d_list)))

        indices = np.argsort(scores)[::-1][: args.top_k]
        doc_ids = [doc_ids[idx] for idx in indices]
        for k, doc_id in enumerate(doc_ids):
            if doc_id == q["doc_id"]:
                mrr = 1 / (k + 1)
                break

        mrrs.append(mrr)

    print(f"MRR@{args.top_k}: {np.mean(mrrs)}")
