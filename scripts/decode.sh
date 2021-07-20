#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DOMAIN=restaurant
MODEL_SAVE_PATH=saved_models/t5-base


output_dir=${MODEL_SAVE_PATH}/${DOMAIN}
python exp.py \
    --no_cuda \
    --mode decode \
    --model_type t5 \
    --model_path ${output_dir} \
    --decode_input_file data/${DOMAIN}/test.txt \
    --decode_output_file ${output_dir}/results.json \
    --num_samples 5 \
    --top_k 5 \
    --length 80
