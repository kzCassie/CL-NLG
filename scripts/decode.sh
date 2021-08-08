#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
domain=$1
curriculum_name=$2
model_type=t5

### file path ###
# data
data_file_name="test"
data_cache_dir=data_cached
# output model - use outputted model for decoding
model_save_path=saved_models/t5-small/${curriculum_name}

### hyper param ###
seed=42
batch_size=20

## Do NOT modify
output_dir=${model_save_path}/${domain}
decode_input_file=data/${domain}/${data_file_name}.txt
data_cache_path=${data_cache_dir}/${domain}/${data_file_name}.bin


python exp.py \
    --seed ${seed} \
    --mode "decode" \
    --model_type ${model_type} \
    --model_path ${output_dir} \
    --decode_input_file ${decode_input_file} \
    --data_cache_path ${data_cache_path} \
    --overwrite_cache \
    --decode_batch_size ${batch_size}
#    --top_k 5
#    --length 80
