#!/usr/bin/env bash

cd ../fairseq_cli ;

CUDA_VISIBLE_DEVICES=0 python distance_prior.py \
../data-bin/iwslt14.tokenized.de-en.joined/ \
--seed 1 \
--source-lang de \
--target-lang en \
--arch distance_transformer_iwslt_de_en \
--dataset-impl mmap \
--task-name iwslt14de2en \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--generate-distance
