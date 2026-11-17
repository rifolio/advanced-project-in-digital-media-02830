#!/bin/sh
#BSUB -q gpuv100
#BSUB -J kev_ft
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 12:00
#BSUB -o logs/kev_%J.out
#BSUB -e logs/kev_%J.err

cd /zhome/1f/2/223688/repositories/kev
export HF_HOME=/zhome/1f/2/223688/.cache/huggingface
/zhome/1f/2/223688/.local/bin/uv run --no-sync python -m kev.train --data train.jsonl --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b --weights_dtype bf16 --dtype bf16 --max_state 1024 --epochs 2 --lr 2e-5 --batch 1 --accum 8 --checkpointing 1 --device cuda --out runs/mine-v2
