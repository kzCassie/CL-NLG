#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
domain=naive_5_shot
curriculum_name=$1  #[NC, one_pass, baby_step]
model_type=t5
curriculum_num_bucket=5

### file path ###
# data
data_folder="other_data/sgd"
data_file_name="train.src"
data_tgt_name="train.trg"
dev_file_name="dev.src"
dev_tgt_name="dev.trg"
data_cache_dir=data_cached
# input model
model_name=t5-small
# output model
model_save_path=saved_models/t5-small/${curriculum_name}

### hyper param ###
seed=42
epoch=500
train_batch_size=20
lr=5e-5

## Do NOT modify
train_data_file=${data_folder}/${domain}/${data_file_name}.txt
train_tgt_file=${data_folder}/${domain}/${data_tgt_name}.txt
dev_data_file=${data_folder}/${domain}/${dev_file_name}.txt
dev_tgt_file=${data_folder}/${domain}/${dev_tgt_name}.txt
data_cache_path=${data_cache_dir}/${data_folder}/${domain}/${data_file_name}.bin
output_dir=${model_save_path}/${domain}


python exp.py \
    --seed ${seed} \
    --mode train \
    --curriculum_name ${curriculum_name} \
    --curriculum_num_bucket ${curriculum_num_bucket} \
    --model_type ${model_type} \
    --model_name ${model_name} \
    --output_dir ${output_dir} \
    --train_data_file ${train_data_file} \
    --train_tgt_file ${train_tgt_file} \
    --dev_data_file ${dev_data_file} \
    --dev_tgt_file ${dev_tgt_file} \
    --dev_batch_size 20 \
    --data_cache_path ${data_cache_path} \
    --num_train_epochs ${epoch} \
    --train_batch_size ${train_batch_size} \
    --learning_rate ${lr} \
    --train_patience 10 \
    --valid_every_epoch 10 \
    --overwrite_output_dir \
    --overwrite_cache
#--eval_data_file data/${domain}/train.txt \
