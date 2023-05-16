#!/bin/bash

# Change these!
CORPUS_FILE=".../corpus.jsonl"
QUERIES_FILE=".../queries.jsonl"
NEGS_FILE=".../negs.tsv"

OUTPUT_DIR=$(dirname $QUERIES_FILE)

# Convert hard_negatives.tsv to triples.tsv
python -m inpars_aux.triples \
    --input $NEGS_FILE \
    --output $OUTPUT_DIR/triples.tsv \
    --corpus $CORPUS_FILE \
    --queries $QUERIES_FILE

# Train cross-encoder
mkdir -p $OUTPUT_DIR/models
python -m inpars.train_v2 \
    --triples $OUTPUT_DIR/triples.tsv \
    --base_model castorini/monot5-3b-msmarco-10k \
    --model_type t5 \
    --output_dir $OUTPUT_DIR/models/monoT5 \
    --save_total_limit 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-6 \
    --weight_decay 5e-5 \
    --warmup_ratio 0.1 \
    --max_steps 1968 \
    --optim adamw_bnb_8bit \
    --bf16
