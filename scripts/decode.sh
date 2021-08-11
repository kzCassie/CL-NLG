#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
domain=naive_5_shot
curriculum_name=$1
model_type=t5

### file path ###
# data
data_folder="other_data/sgd"
data_file_name="test.src"
data_tgt_name="test.trg"
# output model - use outputted model for decoding
model_save_path=saved_models/sgd/${curriculum_name}

### hyper param ###
seed=42
batch_size=20

## Do NOT modify
output_dir=${model_save_path}/${domain}
decode_input_file=${data_folder}/${domain}/${data_file_name}
decode_tgt_file=${data_folder}/${domain}/${data_tgt_name}
data_cache_dir=data_cached/${data_folder}/${domain}/
eval_output_file=${data_folder}/${domain}/results


python exp.py \
    --seed ${seed} \
    --mode "decode" \
    --model_type ${model_type} \
    --model_path ${output_dir} \
    --decode_input_file ${decode_input_file} \
    --decode_tgt_file ${decode_tgt_file} \
    --data_cache_dir ${data_cache_dir} \
    --overwrite_cache \
    --decode_batch_size ${batch_size}

python exp.py \
    --mode "eval" \
    --eval_output_file ${eval_output_file} \
    --eval_tgt_file ${decode_tgt_file}
