#!/bin/bash
#SBATCH --job-name          muzero
#SBATCH --partition         gpu_requeue
#SBATCH --array             1-4
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   1
#SBATCH --gpus              1
#SBATCH --cpus-per-task     1
#SBATCH --time              0-00:40
#SBATCH --mem               2GB
#SBATCH --output            out/%j.%a.out
#SBATCH --error             out/%j.%a.err

nvidia-smi

wandb agent $SWEEP_ID --count 1
