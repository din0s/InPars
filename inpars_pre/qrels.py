"""Extract qrels.tsv from queries.jsonl"""
import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output TSV file")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    df = df[["query_id", "doc_id"]]

    df.to_csv(args.output, sep="\t", header=False, index=False)
