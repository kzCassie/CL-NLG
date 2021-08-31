#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
curriculum_name=$1

# data
dataset="sgd"
domain="naive_10_shot"
data_file_name="test.src"
data_tgt_name="test.trg"

# model
model_type=t5
seed=42
batch_size=32

## Do NOT modify
data_folder=data/${dataset}/${domain}
decode_input_file=${data_folder}/${data_file_name}
decode_tgt_file=${data_folder}/${data_tgt_name}
data_cache_dir=data_cached/${dataset}/${domain}
output_dir=saved_models/${dataset}/${domain}/${curriculum_name}

python exp.py \
    --seed ${seed} \
    --mode "decode" \
    --model_type ${model_type} \
    --model_path ${output_dir} \
    --output_dir ${output_dir} \
    --decode_input_file ${decode_input_file} \
    --decode_tgt_file ${decode_tgt_file} \
    --data_cache_dir ${data_cache_dir} \
    --overwrite_cache \
    --decode_batch_size ${batch_size}