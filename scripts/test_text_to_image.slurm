#!/bin/bash
#SBATCH -J diffusers-inference
#SBATCH -p ty_normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -o ./log/%j.output
#SBATCH -e ./log/%j.err

module load apps/anaconda3/5.2.0
source activate diffusers
module switch compiler/dtk/23.10

python ../test_text_to_image.py --cfg=../cfg/default.json