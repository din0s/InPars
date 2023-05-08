"""Convert queries.jsonl to tokenization format"""
import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    df["query_id"] = df["query_id"].astype(str)
    df = df.rename(columns={"query_id": "id", "query": "text"})
    df.to_json(args.output, orient="records", lines=True)
