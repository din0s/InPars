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

python -u -m inpars.rerank \
    --model ./model-arguana/ \
    --dataset arguana \
    --output_run arguana-run.txt \
    --batch_size 128 \
    --bf16
