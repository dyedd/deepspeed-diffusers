#!/bin/bash
export WANDB_ERROR_REPORTING=False
nohup deepspeed --num_gpus=2 ./train_text_to_image.py --cfg=./cfg/default.json > test.log 2>&1 &