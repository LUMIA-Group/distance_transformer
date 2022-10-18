#!/bin/bash


function train () {
cd ..

LOG_DIR=log/iwslt_de_en/transformer_seed${1}
mkdir -p ${LOG_DIR}
cp run/train_iwslt_de2en_baseline.sh ${LOG_DIR}/train_iwslt_de2en_baseline.sh

CUDA_VISIBLE_DEVICES=0,1 python train.py \
data-bin/iwslt14.tokenized.de-en \
--distance-path distance_prior \
--seed ${1} \
--source-lang de \
--target-lang en \
--arch transformer_iwslt_de_en \
--optimizer adam \
--adam-betas '(0.9,0.98)' \
--clip-norm 0.0 \
--dropout 0.3 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-7 \
--lr 7e-3 \
--task-name iwslt14de2en \
--max-epoch 100 \
--update-freq 1 \
--warmup-updates 6000 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--no-progress-bar \
--no-epoch-checkpoints \
--stop-min-lr 1e-09 \
--ddp-backend no_c10d \
--keep-interval-updates 5 \
--log-interval 50 \
--share-all-embeddings \
--keep-best-checkpoints 1 \
--trainalpha-after-epoches 60 \
--validate-after-updates 20000 \
--save-dir ${LOG_DIR} \
--tensorboard-logdir ${LOG_DIR}/tensorboard_log \
| tee -a ${LOG_DIR}/train_log.txt


BEAM_SIZE=5
LPEN=1.0

DATA_PATH=data-bin/iwslt14.tokenized.de-en/

CKPT=checkpoint_best.pt
echo ${LOG_DIR}/${CKPT}

CUDA_VISIBLE_DEVICES=0 python generate.py \
	${DATA_PATH} \
	--distance-path distance_prior \
	--batch-size 128 \
	--path ${LOG_DIR}/${CKPT} \
	--beam ${BEAM_SIZE} \
	--lenpen ${LPEN} \
	--remove-bpe \
	--arch transformer_iwslt_de_en \
	--task-name iwslt14de2en \
	--source-lang de \
	--target-lang en \
| tee -a ${LOG_DIR}/test_best_log.txt

cd run
}

train 1;
