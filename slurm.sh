#!/bin/bash
#SBATCH -J diffusers
#SBATCH -p ty_normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:2
#SBATCH -o %j.out
#SBATCH -e %j.err

module load apps/anaconda3/5.2.0
source activate diffusers

deepspeed ds_sd_train.py --cfg=cfg.json