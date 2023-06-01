"""Benchmarks a trained biencoder on the training qrel file."""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus", type=argparse.FileType("r"), required=True)
    parser.add_argument("--queries", type=argparse.FileType("r"), required=True)
    parser.add_argument("--qrels", type=argparse.FileType("r"), required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--add_prefixes", action="store_true")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_path)
    model = torch.compile(model)
    model.eval()

    qrels = pd.read_csv(args.qrels, sep="\t", header=None, names=["query_id", "doc_id"])
    qrels = qrels.groupby("query_id")["doc_id"].apply(set).to_dict()
    qrels = defaultdict(set, qrels)

    corpus_ids = []
    corpus_embeddings = []
    corpus = pd.read_json(args.corpus, lines=True)
    for i in tqdm(range(0, len(corpus), args.batch_size), desc="Embedding corpus"):
        batch = corpus[i : i + args.batch_size]
        if args.add_prefixes:
            batch["text"] = "passage: " + batch["text"]
        batch_embeddings = model.encode(batch["text"].tolist(), convert_to_tensor=True)
        corpus_embeddings.extend(batch_embeddings)
        corpus_ids.extend(batch["_id"].tolist())

    corpus_embeddings = torch.stack(corpus_embeddings)

    query_embeddings = []
    queries = pd.read_json(args.queries, lines=True)
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Embedding queries"):
        batch = queries[i : i + args.batch_size]
        if args.add_prefixes:
            batch["query"] = "query: " + batch["query"]
        batch_embeddings = model.encode(batch["query"].tolist(), convert_to_tensor=True)
        query_embeddings.extend(batch_embeddings)

    query_embeddings = torch.stack(query_embeddings)

    mrrs = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Evaluating queries"):
        batch = queries[i : i + args.batch_size]
        batch_embeddings = query_embeddings[i : i + args.batch_size]
        scores = batch_embeddings @ corpus_embeddings.T
        _, indices = torch.topk(scores, args.top_k, dim=1)

        for j in range(len(batch)):
            mrr = 0
            query_id = batch.iloc[j]["query_id"]
            doc_ids = [corpus_ids[idx] for idx in indices[j]]
            for k, doc_id in enumerate(doc_ids):
                if doc_id in qrels[query_id]:
                    mrr = 1 / (k + 1)
                    break

            mrrs.append(mrr)

    print(f"MRR@{args.top_k}: {np.mean(mrrs)}")
