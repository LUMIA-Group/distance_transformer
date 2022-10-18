#!/usr/bin/env bash

cd ../fairseq_cli ;

CUDA_VISIBLE_DEVICES=0 python distance_prior.py \
../data-bin/iwslt14.tokenized.de-en.joined/ \
--seed 1 \
--source-lang en \
--target-lang de \
--arch distance_transformer_iwslt_en_de \
--dataset-impl mmap \
--task-name iwslt14en2de \
--max-epoch 4 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--generate-distance
