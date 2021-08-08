#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

curriculum=$1
domains="attraction hotel laptop restaurant taxi train tv"

for domain in $domains; do
  echo "$domain"
  bash scripts/train.sh "$domain" "$curriculum"
done
