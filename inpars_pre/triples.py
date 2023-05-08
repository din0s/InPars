"""Convert hard_negatives.tsv to triples.tsv"""
import argparse
import csv

import pandas as pd
from tqdm import tqdm


def fix_text(t):
    return " ".join(t.split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType("r"), required=True)
    parser.add_argument("--output", type=argparse.FileType("w"), required=True)
    parser.add_argument("--corpus", type=argparse.FileType("r"), required=True)
    parser.add_argument("--queries", type=argparse.FileType("r"), required=True)
    args = parser.parse_args()

    queries = pd.read_json(args.queries, lines=True, dtype={"query_id": str})
    queries.set_index("query_id", inplace=True)

    corpus = pd.read_json(args.corpus, lines=True)
    corpus.set_index("_id", inplace=True)

    with args.input as f_in, args.output as f_out:
        writer = writer = csv.writer(f_out, delimiter='\t', lineterminator='\n')

        n_lines = sum(1 for _ in f_in)
        f_in.seek(0)
        for line in tqdm(f_in, total=n_lines):
            line = line.strip().split("\t")

            q_id = line[0]
            if q_id not in queries.index:
                continue

            query = queries.loc[q_id]
            q_text = fix_text(query["query"])
            pos_text = fix_text(query["doc_text"])

            neg_ids = line[1:]
            for neg_id in neg_ids:
                neg_text = fix_text(corpus.loc[neg_id]["text"])
                writer.writerow([q_text, pos_text, neg_text])
