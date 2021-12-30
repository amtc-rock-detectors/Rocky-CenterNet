#!/usr/bin/env bash

cd src
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/lib

python main.py rockdet --exp_id rockdet_rock_coco_dla_2x --batch_size 8 --lr 1e-4 --gpus 3 --num_epochs 500 --lr_step 20,50,80 --dataset rock_coco --folder rock_front_hammer_v1_v2_dataset --val_intervals 1 --ang_weight 0.2 --ang_anchors 5

