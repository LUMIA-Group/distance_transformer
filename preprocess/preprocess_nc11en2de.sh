# Preprocess/binarize the data
TEXT=preprocess/nc11deen
cd ..;
python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/nc11en2de \
    --joined-dictionary \
    --dataset-impl mmap \
    --workers 20
