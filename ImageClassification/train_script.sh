torchrun --nproc_per_node=2 train.py --model efficientnet_v2_s --batch-size 32 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 2 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.00002 --norm-weight-decay 0.0 --train-crop-size 300 --model-ema --val-crop-size 384 --val-resize-size 384 --ra-sampler --ra-reps 4
