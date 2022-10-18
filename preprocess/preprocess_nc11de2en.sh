# Preprocess/binarize the data
TEXT=preprocess/nc11deen
cd ..;
python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/nc11de2en \
    --joined-dictionary \
    --dataset-impl mmap \
    --workers 20
