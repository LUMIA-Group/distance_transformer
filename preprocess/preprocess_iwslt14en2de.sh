# Preprocess/binarize the data
TEXT=preprocess/iwslt14.tokenized.de-en
cd ..;
python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --joined-dictionary \
    --dataset-impl mmap \
    --workers 20