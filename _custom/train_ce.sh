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
python -m inpars.train_minilm \
    --triples $OUTPUT_DIR/triples.tsv \
    --base_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --output_dir $OUTPUT_DIR/models/miniLM \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-6 \
    --weight_decay 5e-5 \
    --warmup_ratio 0.1 \
    --max_steps 624 \
    --optim adamw_torch \
    --bf16

# Save in SBERT format & cleanup
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$OUTPUT_DIR/models/miniLM-strat').save('$OUTPUT_DIR/models/miniLM_sbert')"
rm -rf $OUTPUT_DIR/models/miniLM
