#!/bin/bash

cd /home/aorko/workplace/Transformer_Research


mkdir -p vanilla

source /home/aorko/workplace/Transformer_Research/venv/bin/activate

exec python3 -u train.py \
    --attention vanilla \
    --out_dir vanilla \
    --metrics_path vanilla/metrics.json \
    >> vanilla/train.log 2>&1

# mkdir -p custom

# source /home/aorko/workplace/Transformer_Research/venv/bin/activate

# exec python3 -u train.py \
#     --attention custom \
#     --out_dir custom \
#     --metrics_path custom/metrics.json \
#     >> custom/train.log 2>&1