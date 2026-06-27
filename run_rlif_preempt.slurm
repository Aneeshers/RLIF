#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris            # PREEMPTIBLE (PreemptMode=REQUEUE); QOS normal
#SBATCH --qos=normal
#SBATCH --requeue                   # on preemption, slurm REQUEUES this job (same JOBID, same NAME)
#SBATCH --exclude=iris-hgx-1,iris-hgx-2,macondo1,macondo3,orion-dgx
#SBATCH --job-name=rlif_pe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=21-00:00:00
#SBATCH --output=/iris/u/aneeshm/slurm-logs/%x-%j.out
#SBATCH --error=/iris/u/aneeshm/slurm-logs/%x-%j.err
# PREEMPTIBLE / AUTO-RESUMING RLIF launcher (iris partition). Resume is handled IN-PYTHON: with
# --checkpoint_model=True and --resume=True the script detects the newest checkpoint in this run's
# DETERMINISTIC chkpt_dir (log_dir/<exp_prefix>/{online_ckpt,checkpoints}) and continues from the
# recovered env-step. To isolate per-arm checkpoints (exp_prefix only encodes seed/utd/offline, NOT
# intervention/cosw), the wrapper passes a UNIQUE --log_dir per arm. Required env: ARGS (full flag
# string, must include --checkpoint_model=True and a unique --log_dir=...).
set -e
cd /iris/u/aneeshm/corrections-project
mkdir -p /iris/u/aneeshm/slurm-logs
source RLIF/rlif_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/lib/x86_64-linux-gnu
export D4RL_DATASET_DIR=/iris/u/aneeshm/d4rl_datasets
export WANDB_DIR=/iris/u/aneeshm/wandb
mkdir -p $WANDB_DIR

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import jax; print('jax devices:', jax.devices())"

echo "=== launching (preempt/auto-resume): $ARGS ==="
eval "python -m RLIF.examples.train_rlif_main $ARGS"
