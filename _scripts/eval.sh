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

python -u -m inpars.evaluate \
    --dataset arguana \
    --run $STORAGE/runs/arguana.txt

