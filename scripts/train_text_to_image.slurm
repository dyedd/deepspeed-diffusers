#!/bin/bash
#SBATCH -J diffusers
#SBATCH -p ty_normal
#SBATCH -N 2
#在这里比较特殊，--ntasks-per-node=--gres
#SBATCH --ntasks-per-node=3
#SBATCH -c 4
#SBATCH --gres=dcu:3
#SBATCH -o ./log/%j.output
#SBATCH -e ./log/%j.err

module load apps/anaconda3/5.2.0
source activate diffusers
module switch compiler/dtk/23.10

export WANDB_ERROR_REPORTING=False

if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
#  这里控制deepspeed开启的-n数量，$SLURM_NTASKS_PER_NODE*节点总数
#deepspeed通过srun -n 启动多卡，ds的-n与sbatch不同，是根据slots文件里数量,这里的-n就是卡的数量，ds不会提前-gres内容
    scontrol show hostnames $SLURM_JOB_NODELIST | while read -r line; do
      echo "${line} slots=$SLURM_NTASKS_PER_NODE"
    done > "hostfile"
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=12345
    deepspeed --launcher slurm --hostfile hostfile --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../train_text_to_image.py --cfg=../cfg/default.json
else
    rm -f hostfile
    deepspeed ../train_text_to_image.py  --cfg=../cfg/default.json
fi