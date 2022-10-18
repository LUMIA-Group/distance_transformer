#!/usr/bin/env bash

cd ../fairseq_cli ;

CUDA_VISIBLE_DEVICES=0 python distance_prior.py \
../data-bin/aspec_ch_ja/ \
--seed 1 \
--source-lang ch \
--target-lang ja \
--arch distance_transformer_aspec_ch_ja \
--dataset-impl mmap \
--task-name aspecch2ja \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--generate-distance