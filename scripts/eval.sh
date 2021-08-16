#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
curriculum_name=$1
dataset="sgd"
domain="naive_5_shot"

## Do NOT modify
output_dir=saved_models/${dataset}/${domain}/${curriculum_name}
eval_output_file=${output_dir}/results.txt
decode_tgt_file=data/${dataset}/${domain}/test.trg

python exp.py \
    --mode "eval" \
    --output_dir ${output_dir} \
    --eval_output_file ${eval_output_file} \
    --eval_tgt_file ${decode_tgt_file}
