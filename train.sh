#!/bin/bash

deepspeed --num_gpus=2 ds_sd_train.py --cfg=cfg.json