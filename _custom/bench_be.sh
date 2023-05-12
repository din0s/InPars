#!/bin/bash

# Change these!
MODEL_PATH=".../mpnet_sbert"
CORPUS_FILE=".../corpus.jsonl"
QUERIES_FILE=".../queries.jsonl"
QRELS_FILE=".../qrels.tsv"

python -m inpars_aux.bench_be \
    --model_path $MODEL_PATH \
    --corpus $CORPUS_FILE \
    --queries $QUERIES_FILE \
    --qrels $QRELS_FILE \
    --batch_size 128 \
    --top_k 10
