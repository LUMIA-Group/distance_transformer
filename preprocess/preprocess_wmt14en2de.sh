# Preprocess/binarize the data
TEXT=preprocess/wmt14_en_de
cd ..;
python fairseq_cli/preprocess.py --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_de \
    --joined-dictionary \
    --dataset-impl mmap \
    --workers 20