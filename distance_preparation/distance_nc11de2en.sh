#!/usr/bin/env bash

cd .. ;

LOG_DIR=distance_prior/nc11de2en
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/distance_prior.py \
data-bin/nc11de2en/ \
--seed 1 \
--source-lang de \
--target-lang en \
--arch transformer_nc11_de_en \
--dataset-impl mmap \
--task-name nc11de2en \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--generate-distance
