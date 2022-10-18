#!/bin/bash


function train () {
cd ..

LOG_DIR=log/iwslt_en_de/drop2_layer1_head3_seed${1}
mkdir -p ${LOG_DIR}
cp run/train_iwslt_en2de.sh ${LOG_DIR}/train_iwslt_en2de.sh

CUDA_VISIBLE_DEVICES=2,3 python train.py \
data-bin/iwslt14.tokenized.de-en \
--distance-path distance_prior \
--seed ${1} \
--source-lang en --target-lang de \
--task-name iwslt14en2de \
--share-decoder-input-output-embed \
--arch distance_transformer_iwslt_en_de \
--tau 10 \
--syntactic-head-num 3 \
--attention-dropout 0.2 \
--optimizer  adam  \
--adam-betas  '(0.9,0.98)' \
--clip-norm 0.0 \
--lr 1e-3 \
--gating-ratio 1 \
--lr-scheduler inverse_sqrt \
--warmup-updates 8000 \
--dropout 0.3 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--update-freq 1 \
--max-tokens 8192 \
--validate-after-updates 10000 \
--no-progress-bar \
--keep-best-checkpoints 1 \
--find-unused-parameters \
--no-epoch-checkpoints \
--max-epoch 100 \
--save-dir ${LOG_DIR} \
--tensorboard-logdir ${LOG_DIR}/tensorboard_log \
| tee -a ${LOG_DIR}/train_log.txt


BEAM_SIZE=5
LPEN=1.0

DATA_PATH=data-bin/iwslt14.tokenized.de-en/

CKPT=checkpoint_best.pt
echo ${LOG_DIR}/${CKPT}

CUDA_VISIBLE_DEVICES=3 python generate.py \
	${DATA_PATH} \
	--distance-path distance_prior \
	--batch-size 128 \
	--path ${LOG_DIR}/${CKPT} \
	--beam ${BEAM_SIZE} \
	--lenpen ${LPEN} \
	--remove-bpe \
	--syntactic-head-num 3 \
	--arch distance_transformer_iwslt_en_de \
	--task-name iwslt14en2de \
	--source-lang en \
	--target-lang de \
| tee -a ${LOG_DIR}/test_best_log.txt

cd run
}

train 1;
