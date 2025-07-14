#!/bin/bash
cd /data/home/brahimhh/ter_project

if [ ! -d .venv ]; then
    /data/home/brahimhh/.local/bin/uv venv
    source .venv/bin/activate
    ## /data/home/brahimhh/.local/bin/uv run pip install -r /data/home/brahimhh/ter_project/requirements.txt
fi

/data/home/brahimhh/.local/bin/uv run python3 /data/home/brahimhh/ter_project/src/train_micro_avg.py \
    --data_dir ./data/clean \
    --output_dir ./output \
    --reuse True \
    --start_n 1 \
    --end_n 11 \
    --n_splits 5 \
    --iter 100 \
    --subject $1 
