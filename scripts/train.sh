#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### Curriculum ###
curriculum_name=$1 #[NC, one_pass, baby_step, spl.hard, spl.linear, spl.mixture]
curriculum_num_bucket=5

# data
dataset="sgd"
domain="naive_5_shot"
data_file_name="train.src"
data_tgt_name="train.trg"
dev_file_name="dev2.src"
dev_tgt_name="dev2.trg"

# model
model_type=t5
model_name="t5-small"
seed=42
epoch=500
#epoch=100
train_batch_size=128
dev_batch_size=128
lr=1e-3

## Do NOT modify
data_folder=data/${dataset}/${domain}
train_data_file=${data_folder}/${data_file_name}
train_tgt_file=${data_folder}/${data_tgt_name}
dev_data_file=${data_folder}/${dev_file_name}
dev_tgt_file=${data_folder}/${dev_tgt_name}
data_cache_dir=data_cached/${dataset}/${domain}
output_dir=saved_models/${dataset}/${domain}/${curriculum_name}

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
  --data_cache_dir ${data_cache_dir} \
  --num_train_epochs ${epoch} \
  --train_batch_size ${train_batch_size} \
  --dev_batch_size ${dev_batch_size} \
  --learning_rate ${lr} \
  --train_patience 10 \
  --overwrite_output_dir
#    --overwrite_cache
