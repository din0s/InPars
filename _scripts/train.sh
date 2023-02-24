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

python -u -m inpars.train \
    --triples arguana-triples.tsv \
    --base_model castorini/monot5-3b-msmarco-10k \
    --output_dir ./model-arguana/ \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --learning_rate 3e-6 \
    --weight_decay 5e-5 \
    --warmup_steps 156 \
    --max_steps 156 \
    --optim adamw_bnb_8bit \
    --bf16
