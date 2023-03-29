import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict

import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from transformers import set_seed

from .dataset import load_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType("r"), required=True)
    parser.add_argument("--output_dir", type=str, default="./benc_data")
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--dataset_source",
        default="ir_datasets",
        help="The dataset source: ir_datasets or pyserini",
    )
    parser.add_argument("--index", type=str)
    parser.add_argument("--max_hits", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.threads > os.cpu_count():
        print(
            f"Warning: n_threads ({args.threads}) > n_cpus ({os.cpu_count()})."
        )

    set_seed(args.seed)

    index = args.index
    if os.path.exists(args.dataset):
        if args.dataset.endswith(".csv"):
            corpus = pd.read_csv(args.input)
        else:
            corpus = pd.read_json(args.input, lines=True)
    else:
        corpus = load_corpus(args.dataset, args.dataset_source)
        index = f"beir-v1.0.0-{args.dataset}-flat"

    # Convert to {'doc_id': 'text'} format
    corpus = dict(zip(corpus["doc_id"], corpus["text"]))

    if os.path.isdir(index):
        searcher = LuceneSearcher(index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(index)

    dir = os.path.join(args.output_dir, args.dataset)
    topics_path = os.path.join(dir, "topics.tsv")
    os.makedirs(dir, exist_ok=True)

    n_no_query = 0
    n_docs_not_found = 0
    queries = {}
    with args.input as f_in, open(topics_path, "w") as f_out:
        n_queries = sum(1 for _ in f_in)
        f_in.seek(0)
        for i, line in tqdm(enumerate(f_in), desc="Loading synthetic queries", total=n_queries):
            row = json.loads(line.strip())

            if not row["query"]:
                n_no_query += 1
                continue

            query = " ".join(row["query"].split())  # Removes line breaks and tabs.
            q_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
            q_id = f'synth-{q_hash}-{row["doc_id"]}'
            queries[q_id] = {
                "text": query,
                "doc_id": row["doc_id"],
            }

            f_out.write(f"{i}\t{query}\n")

    # Convert queries to tuples (query, q_id, doc_id)
    queries = [(q["text"], q_id, q["doc_id"]) for q_id, q in queries.items()]

    print("Retrieving candidates...")
    run_path = os.path.join(dir, "pyserini_run.txt")
    subprocess.run([
        "python", "-m", "pyserini.search.lucene",
        "--threads", str(args.threads),
        "--batch-size", str(args.batch_size),
        "--hits", str(args.max_hits + 1),
        "--index", index,
        "--topics", topics_path,
        "--output", run_path,
        "--bm25",
    ])

    results = defaultdict(list)
    with open(run_path) as f:
        for line in f:
            q_idx, _, doc_id, _, _, _ = line.split()
            results[q_idx].append(doc_id)

    csv_args = {
        "delimiter": "\t",
        "lineterminator": "\n",
        "quoting": csv.QUOTE_NONE,
        "quotechar": "",
    }

    out_negs = open(os.path.join(dir, "negs.tsv"), "w")
    out_qrels = open(os.path.join(dir, "qrels.tsv"), "w")
    out_qmap = open(os.path.join(dir, "queries.jsonl"), "w")
    with out_negs as f_negs, out_qrels as f_qrels, out_qmap as f_map:
        neg_writer = csv.writer(f_negs, **csv_args)
        qrel_writer = csv.writer(f_qrels, **csv_args)
        for q_idx in tqdm(results, desc="Sampling"):
            query, q_id, pos_doc_id = queries[int(q_idx)]
            f_map.write(json.dumps({"id": q_id, "text": query}) + "\n")
            qrel_writer.writerow([q_id, pos_doc_id])

            hits = results[q_idx]
            n_hits = len(hits)
            sampled_ranks = random.sample(
                range(n_hits),
                k=min(n_hits, args.n_samples + 1),
            )

            neg_docs = []
            for rank in sorted(sampled_ranks):
                neg_doc_id = hits[rank]

                if pos_doc_id == neg_doc_id:
                    continue

                if neg_doc_id not in corpus:
                    n_docs_not_found += 1
                    continue

                neg_docs.append(neg_doc_id)

                if len(neg_docs) > args.n_samples:
                    break

            neg_writer.writerow([q_id] + neg_docs)

    if n_no_query > 0:
        print(f"{n_no_query} lines without queries.")

    if n_docs_not_found > 0:
        print(
            f"{n_docs_not_found} docs returned by the search engine but not found in the corpus."
        )

    print("Done!")
