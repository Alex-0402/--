#!/bin/bash
# 8卡并行训练命令
# 基于第220轮checkpoint继续训练，应用改进参数

cd /home/Oliver-0402/--/protein_mdm

torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --resume checkpoints/checkpoint_epoch_220.pt \
    --epochs 300 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 20 \
    --early_stopping_patience 30 \
    --early_stopping_min_delta 0.001 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000 \
    --masking_strategy random \
    --save_dir checkpoints \
    --visualize \
    --plot_every 5
