import argparse
import os
import re
from glob import glob

import pandas as pd


def remove_weird_suffix(text):
    m = re.match(r"^(.*)(?: # [^#]+)$", text)
    if m:
        return m.group(1)
    else:
        return text


def get_clean_df(fpath):
    # Load jsonl with pandas
    df = pd.read_json(fpath, lines=True)

    # Only keep "query", "doc_id", "doc_text"
    df = df[["query", "doc_id", "doc_text"]]

    # Filter queries to remove weird hashtag suffixes
    df["query"] = df["query"].apply(remove_weird_suffix)

    # Remove empty queries
    df = df[df["query"].str.len() > 0]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    # Get all jsonl files
    for fpath in glob(os.path.join(args.in_dir, "*.jsonl")):
        # Read jsonl & clean data
        df = get_clean_df(fpath)

        # Write back as jsonl
        fname = os.path.basename(fpath)
        output = os.path.join(args.out_dir, fname)
        df.to_json(output, orient="records", lines=True)


if __name__ == "__main__":
    main()
