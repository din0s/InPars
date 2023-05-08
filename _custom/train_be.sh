#!/bin/bash

CORPUS_FILE=".jsonl"
QUERIES_FILE=".jsonl"
NEGS_FILE=".tsv"

OUTPUT_DIR=$(dirname $QUERIES_FILE)

# Extract qrels.tsv from queries.jsonl
python -m inpars_aux.qrels $QUERIES_FILE $OUTPUT_DIR/qrels.tsv

# Convert queries to tokenization format
python -m inpars_aux.queries $QUERIES_FILE $OUTPUT_DIR/queries_tok.jsonl

# Tokenize data for bi-encoder
mkdir -p $OUTPUT_DIR/tok_data
python -m inpars.tokenize \
    --corpus $CORPUS_FILE \
    --queries $OUTPUT_DIR/queries_tok.jsonl \
    --output_dir $OUTPUT_DIR/tok_data \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --max_seq_length 512 \
    --threads 12

# Train bi-encoder
mkdir -p $OUTPUT_DIR/models
python -m bi_encoder.run \
    --do_train \
    --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
    --output_dir $OUTPUT_DIR/models/mpnet \
    --overwrite_output_dir \
    --corpus_file $OUTPUT_DIR/tok_data/corpus \
    --train_query_file $OUTPUT_DIR/tok_data/queries \
    --train_qrels $OUTPUT_DIR/qrels.tsv \
    --neg_file $NEGS_FILE \
    --query_max_len 512 \
    --passage_max_len 512 \
    --bf16 \
    --per_device_train_batch_size 16 \
    --train_group_size 2 \
    --sample_neg_from_topk 200 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --dataloader_num_workers 12 \
    --sentence_pooling_method mean

# Save in SBERT format
python -m inpars_aux.sbert $OUTPUT_DIR/models/mpnet $OUTPUT_DIR/models/mpnet_sbert

# Cleanup
rm -rf $OUTPUT_DIR/models/mpnet
