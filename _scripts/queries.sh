#/bin/bash

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
mkdir -p $STORAGE/synthetic

python -u -m inpars.generate \
    --prompt inpars-gbq \
    --dataset arguana \
    --dataset_source ir_datasets \
    --base_model togethercomputer/GPT-JT-6B-v1 \
    --output $STORAGE/synthetic/arguana.jsonl \
    --max_new_tokens 256 \
    --max_query_length 512 \
    --max_doc_length 512 \
    --max_generations 10000 \
    --batch_size 8 \
    --fp16

