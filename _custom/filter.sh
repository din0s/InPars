#!/bin/bash

CORPUS_FILE=".jsonl"
QUERIES_FILE=".jsonl"

OUTPUT_DIR=$(dirname $QUERIES_FILE)

python -m inpars.filter \
    --input $QUERIES_FILE \
    --output $OUTPUT_DIR/queries_filtered.jsonl \
    --dataset $CORPUS_FILE \
    --dataset_source jsonl \
    --filter_strategy reranker \
    --keep_top_k 10000 \
    --batch_size 128 \
    --fp16
