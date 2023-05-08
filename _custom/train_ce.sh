#!/bin/bash

CORPUS_FILE=".jsonl"
QUERIES_FILE=".jsonl"
NEGS_FILE=".tsv"

OUTPUT_DIR=$(dirname $QUERIES_FILE)

# Convert hard_negatives.tsv to triples.tsv
python -m inpars_aux.triples \
    --input $NEGS_FILE \
    --output $OUTPUT_DIR/triples.tsv \
    --corpus $CORPUS_FILE \
    --queries $QUERIES_FILE

# Train cross-encoder
mkdir -p $OUTPUT_DIR/models
python -m inpars.train \
    --triples $OUTPUT_DIR/triples.tsv \
    --base_model castorini/monot5-3b-msmarco-10k \
    --output_dir $OUTPUT_DIR/models/monot5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --learning_rate 3e-6 \
    --weight_decay 5e-5 \
    --warmup_steps 156 \
    --max_steps 156 \
    --optim adamw_bnb_8bit \
    --bf16
