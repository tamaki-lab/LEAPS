python inference.py \
    -m leaps_vivit \
    -cls_model google/vivit-b-16x2-kinetics400 \
    -leaps_r 0.5 \
    -td data/wds/k400/k400_train_allframe \
    -vd data/wds/k400/wds_k400_val_allframe \
    --devices 1 \
