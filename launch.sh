#!/bin/bash
#SBATCH --partition         gpu_requeue
#SBATCH --array             0-7
#SBATCH --gpus              1
#SBATCH --time              0-00:30
#SBATCH --mem               8GB
#SBATCH --output            out/%x.%j.%2a.out
#SBATCH --error             out/%x.%j.%2a.err

nvidia-smi

wandb agent $SWEEP_ID --count 1
