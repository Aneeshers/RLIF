#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=sc-freegpu
#SBATCH --exclude=macondo1,macondo3,orion-dgx
#SBATCH --job-name=rlif_cosw
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=/iris/u/aneeshm/slurm-logs/%x-%j.out
#SBATCH --error=/iris/u/aneeshm/slurm-logs/%x-%j.err

set -e
cd /iris/u/aneeshm/corrections-project
source RLIF/rlif_env.sh
# add system driver dir for libcuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/lib/x86_64-linux-gnu
export D4RL_DATASET_DIR=/iris/u/aneeshm/d4rl_datasets
export WANDB_DIR=/iris/u/aneeshm/wandb
mkdir -p $WANDB_DIR

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import jax; print('jax devices:', jax.devices())"

echo "=== launching: $RUN_CMD ==="
eval "python -m RLIF.examples.train_rlif_fix $ARGS"
