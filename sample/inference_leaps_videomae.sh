python inference.py \
    -m leaps_videomae \
    -cls_model MCG-NJU/videomae-base-finetuned-kinetics \
    -leaps_r 0.5 \
    -td data/wds/k400/k400_train_allframe \
    -vd data/wds/k400/wds_k400_val_allframe \
    --devices 1 \
    -fs 8
