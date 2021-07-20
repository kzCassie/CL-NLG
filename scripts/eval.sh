#!/bin/bash

DOMAIN=restaurant
MODEL_SAVE_PATH=saved_models/t5-base


output_dir=${MODEL_SAVE_PATH}/${DOMAIN}
python evaluator.py \
    --domain ${DOMAIN} \
    --target_file ${output_dir}/results.json | tee ${output_dir}/eval.txt