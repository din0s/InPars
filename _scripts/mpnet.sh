#/bin/bash
set -eo pipefail

DATASET="arguana"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dinos/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dinos/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/home/dinos/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/home/dinos/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/dinos/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/home/dinos/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
mamba activate thesis

STORAGE=/mnt/disks/storage
mkdir -p $STORAGE/.cache/huggingface && export TRANSFORMERS_CACHE=$_
mkdir -p $STORAGE/.cache/datasets && export HF_DATASETS_CACHE=$_

cd $HOME/InPars

# Mine negatives
mkdir -p $STORAGE/benc_data
python -u \
    -m inpars.mine_negatives \
    --input $STORAGE/synthetic-clean/$DATASET-alpaca.jsonl \
    --output_dir $STORAGE/benc_data \
    --dataset $DATASET \
    --dataset_source ir_datasets \
    --max_hits 500 \
    --n_samples 128 \
    --threads 12

# Tokenize
mkdir -p $STORAGE/tok_data/$DATASET
python -u \
    -m inpars.tokenize \
    --corpus $STORAGE/datasets/$DATASET/corpus.jsonl \
    --queries $STORAGE/benc_data/$DATASET/queries.jsonl \
    --output_dir $STORAGE/tok_data/$DATASET \
    --tokenizer sentence-transformers/all-mpnet-base-v2 \
    --max_seq_length 512 \
    --use_title \
    --threads 12

# Train
mkdir -p $STORAGE/models
python -u \
    -m bi_encoder.run \
    --do_train \
    --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
    --output_dir $STORAGE/models/mpnet_$DATASET \
    --corpus_file $STORAGE/tok_data/$DATASET/corpus \
    --train_query_file $STORAGE/tok_data/$DATASET/queries \
    --train_qrels $STORAGE/benc_data/$DATASET/qrels.tsv \
    --neg_file $STORAGE/benc_data/$DATASET/negs.tsv \
    --query_max_len 512 \
    --passage_max_len 512 \
    --bf16 \
    --per_device_train_batch_size 16 \
    --train_group_size 2 \
    --sample_neg_from_topk 200 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --dataloader_num_workers 12

# Evaluate
python -u \
    ./RetroMAE/examples/retriever/BEIR/beir_test.py \
    --model_name_or_path $STORAGE/models/mpnet_$DATASET \
    --dataset $DATASET \
    --split test \
    --batch_size 512 \
    --pooling_strategy mean \
    --score_function dot

