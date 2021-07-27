#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

domain=restaurant
model_type=t5

### Curriculum ###
curriculum_name="one_pass"

### file path ###
# data
data_file_name=new
data_cache_dir=data_cached
# output model
model_save_path=saved_models/t5-small/${curriculum_name}


### hyper param ###
seed=42

## Do NOT modify
eval_data_file=data/${domain}/${data_file_name}.txt
data_cache_path=${data_cache_dir}/${domain}/${data_file_name}.bin
output_dir=${model_save_path}/${domain}


python exp.py \
    --seed ${seed} \
    --mode "eval" \
    --model_type ${model_type} \
    --model_path ${output_dir} \
    --output_dir ${output_dir} \
    --eval_data_file ${eval_data_file} \
    --data_cache_path ${data_cache_path} \
    --overwrite_cache


#python evaluator.py \
#    --domain ${DOMAIN} \
#    --target_file ${output_dir}/results.json | tee ${output_dir}/eval.txt