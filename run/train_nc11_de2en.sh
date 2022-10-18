#!/usr/bin/env bash


function train () {
cd ..

LOG_DIR=log/nc11_de_en/distance_seed${1}_dropout2_layer1_4head
mkdir -p ${LOG_DIR}
cp run/train_nc11_de2en.sh ${LOG_DIR}/train_nc11_de2en.sh

CUDA_VISIBLE_DEVICES=0,1 python train.py \
distance_transformer/data-bin/nc11de2en \
--distance-path distance_prior \
--seed ${1} \
--source-lang de \
--target-lang en \
--arch distance_transformer_nc11_de_en \
--optimizer adam \
--adam-betas '(0.9,0.98)' \
--clip-norm 0.0 \
--dropout 0.3 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-7 \
--lr 1e-3 \
--task-name nc11de2en \
--update-freq 2 \
--warmup-updates 4000 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--no-progress-bar \
--stop-min-lr 1e-09 \
--keep-best-checkpoints 1 \
--ddp-backend no_c10d \
--log-interval 50 \
--no-epoch-checkpoints \
--max-epoch 80 \
--tau 10.0 \
--syntactic-head-num 4 \
--attention-dropout 0.2 \
--trainalpha-after-epoches 100 \
--validate-after-updates 15000 \
--share-all-embeddings \
--save-dir ${LOG_DIR} \
--tensorboard-logdir ${LOG_DIR}/tensorboard_log \
| tee -a ${LOG_DIR}/train_log.txt


BEAM_SIZE=4
LPEN=0.6

DATA_PATH=data-bin/nc11de2en/

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
	--arch distance_transformer_nc11_de_en \
	--task-name nc11de2en \
	--syntactic-head-num 4 \
	--source-lang de \
	--target-lang en \
| tee -a ${LOG_DIR}/test_best_log.txt

cd run
}

train 1;
