#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

domain=restaurant
model_type=t5

### file path ###
# data
data_file_name=new
data_cache_dir=data_cached
# input model
model_name=t5-small
# output model
model_save_path=saved_models/t5-small

### hyper param ###
seed=42
epoch=50
lr=5e-5

## Do NOT modify
train_data_file=data/${domain}/${data_file_name}.txt
data_cache_path=${data_cache_dir}/${domain}/${data_file_name}.bin
output_dir=${model_save_path}/${domain}


python exp.py \
    --seed ${seed} \
    --mode train \
    --model_type ${model_type} \
    --model_name ${model_name} \
    --output_dir ${output_dir} \
    --train_data_file ${train_data_file} \
    --data_cache_path ${data_cache_path} \
    --num_train_epochs ${epoch} \
    --learning_rate ${lr} \
    --overwrite_output_dir \
    --overwrite_cache
#--eval_data_file data/${domain}/train.txt \
