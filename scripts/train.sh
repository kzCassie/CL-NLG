#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DOMAIN=restaurant
MODEL_SAVE_PATH=saved_models/t5-base
EPOCH=20
LR=5e-5


output_dir=${MODEL_SAVE_PATH}/${DOMAIN}
python exp.py \
    --seed 42 \
    --mode train_with_eval \
    --model_type t5 \
    --model_name t5-small \
    --output_dir ${output_dir} \
    --train_data_file data/${DOMAIN}/train.txt \
    --eval_data_file data/${DOMAIN}/train.txt \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --use_tokenizer \
    --overwrite_output_dir \
    --overwrite_cache