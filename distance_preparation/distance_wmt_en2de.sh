#!/usr/bin/env bash

cd .. ;

LOG_DIR=distance_prior/wmt14en2de
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/distance_prior.py \
data-bin/wmt14_en_de/ \
--seed 1 \
--source-lang en \
--target-lang de \
--arch distance_transformer_wmt_en_de \
--dataset-impl mmap \
--task-name wmt14en2de \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--generate-distance
