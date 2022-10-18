

function train () {
cd ..

LOG_DIR=log/aspec_ch_ja/transformer_seed${1}
mkdir -p ${LOG_DIR}
cp run/train_aspec_ch2ja_baseline.sh ${LOG_DIR}/train_aspec_ch2ja_baseline.sh

DATA_PATH=data-bin/aspec_ch_ja/

CUDA_VISIBLE_DEVICES=0,1 python train.py \
${DATA_PATH} \
--distance-path distance_prior \
--seed ${1} \
--source-lang ch --target-lang ja \
--task-name aspecch2ja \
--share-decoder-input-output-embed \
--arch transformer_aspec_ch_ja \
--optimizer  adam  \
--adam-betas  '(0.9,0.98)' \
--clip-norm 0.0 \
--lr 3e-4 \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--attention-dropout 0 \
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
--max-epoch 120 \
--save-dir ${LOG_DIR} \
--tensorboard-logdir ${LOG_DIR}/tensorboard_log \
| tee -a ${LOG_DIR}/train_log.txt


BEAM_SIZE=5
LPEN=1.0


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
	--arch transformer_aspec_ch_ja \
	--task-name aspecch2ja \
	--source-lang ch \
	--target-lang ja \
| tee -a ${LOG_DIR}/test_best_log.txt

cd run
}

train 1;
