#!/bin/bash

#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks-per-node=32              # 64 tasks per node
#SBATCH --time=3:00:00                    # time limits: 8 hours
#SBATCH --error=myJob.err                 # standard error file
#SBATCH --output=myJob.out                # standard output file
#SBATCH --partition=amdgpu                # partition name
#SBATCH --gres=gpu:2                      # number of GPUs per node
#SBATCH --mail-user=kuceral4@fel.cvut.cz  # where send info about job
#SBATCH --mail-type=ALL                   # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

#singularity run --nv singularity/deep_learning_image.sif python main.py --batch_size=32
singularity run --nv -B /mnt/personal/kuceral4 singularity/deep_learning_image.sif python main.py --batch_size=8
