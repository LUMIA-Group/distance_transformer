

function train () {
cd ..

LOG_DIR=log/wmt_en_de/seed${1}
mkdir -p ${LOG_DIR}
cp run/train_wmt_en2de.sh ${LOG_DIR}/train_wmt_en2de.sh

DATA_PATH=data-bin/wmt14_en_de/
DISTANCE_PATH=distance_prior

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
${DATA_PATH} \
--distance-path ${DISTANCE_PATH} \
--seed ${1} \
--source-lang en --target-lang de \
--task-name wmt14en2de \
--share-decoder-input-output-embed \
--arch distance_transformer_wmt_en_de \
--ignore 0.1 \
--tau 10 \
--attention-dropout 0.1 \
--optimizer  adam  \
--adam-betas  '(0.9,0.98)' \
--clip-norm 0.0 \
--lr 0.01 \
--gating-ratio 1 \
--lr-scheduler inverse_sqrt \
--syntactic-head-num 2 \
--warmup-init-lr 1e-7 \
--warmup-updates 8000 \
--dropout 0.1 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--update-freq 1 \
--max-tokens 32768 \
--validate-after-updates 20000 \
--no-progress-bar \
--keep-best-checkpoints 1 \
--keep-last-epochs 10 \
--find-unused-parameters \
--max-epoch 220 \
--save-dir ${LOG_DIR} \
--tensorboard-logdir ${LOG_DIR}/tensorboard_log \
| tee -a ${LOG_DIR}/train_log.txt

BEAM_SIZE=4
LPEN=0.6

array_new=()
for i in {211,212,213,214,215,216,217,218,219,220}
do
    CKPT=checkpoint${i}.pt
    echo ${LOG_DIR}/${CKPT}
    result=$(CUDA_VISIBLE_DEVICES=0 python generate.py \
	  ${DATA_PATH} \
	  --distance-path ${DISTANCE_PATH} \
	  --batch-size 128 \
	  --path ${LOG_DIR}/${CKPT} \
	  --beam ${BEAM_SIZE} \
	  --lenpen ${LPEN} \
	  --remove-bpe \
	  --arch distancce_transformer_wmt_en_de \
	  --task-name wmt14en2de \
	  --source-lang en \
	  --target-lang de
	  )
	  re="BLEU4 = ..\...,"
    if [[ $result =~ $re ]]
    then
      array_new=(${array_new[@]} ${BASH_REMATCH: 8:5})
    fi
done

echo ${array_new[@]}

cd run
}

train 1;
