#!/bin/bash
#SBATCH --partition         gpu_requeue
#SBATCH --array             0-3
#SBATCH --output            out/%x.%j.%2a.out
#SBATCH --error             out/%x.%j.%2a.err

nvidia-smi

# calls `main` function on the script the sweep was launched from (e.g. muzero.py)
wandb agent $SWEEP_ID --count 4
