#!/bin/bash
# Launch the 2x2x2 (demos x intervention x cosw) factorial for given seeds.
# Usage: bash launch_factorial.sh <seed> [seed2 ...]
cd /iris/u/aneeshm/corrections-project
EXP=./RLIF/experts/rlpd_experts/s24_hopper-expert-v2env/model.pkl
COMMON="--env_name=hopper-expert-v2 --sparse_env=Hopper-v2 --utd_ratio=15 --rlif_mode=False \
--max_traj_length=1000 --max_steps=300000 --start_training=5000 --eval_interval=20000 \
--eval_episodes=10 --log_interval=2000 --seed_replay_with_demos=False \
--intervene_threshold=0.975 --intervention_strategy=unif \
--expert_dir=$EXP --ground_truth_agent_dir=$EXP --project_name=rlif-cosw-d4rl"

# arm table: name demos itv cosw
ARMS=(
  "d0_i0_c0 0.0 False False"
  "d1_i0_c0 0.5 False False"
  "d0_i1_c0 0.0 True  False"
  "d1_i1_c0 0.5 True  False"
  "d1_i1_c1 0.5 True  True"
  "d0_i0_c1 0.0 False True"
  "d1_i0_c1 0.5 False True"
  "d0_i1_c1 0.0 True  True"
)

# Optional: DEP env var = comma-separated job ids; this wave waits (afterany) for them.
DEPFLAG=""
if [ -n "$DEP" ]; then DEPFLAG="--dependency=afterany:${DEP//,/:}"; fi

IDS=""
for SEED in "$@"; do
  for A in "${ARMS[@]}"; do
    read NAME OFR ITV COSW <<< "$A"
    export WANDB_NAME="${NAME}_s${SEED}"
    export ARGS="$COMMON --seed=$SEED --offline_ratio=$OFR --use_intervention=$ITV --cosw=$COSW"
    JID=$(sbatch --parsable $DEPFLAG --job-name="rc_${NAME}_s${SEED}" RLIF/run_rlif.slurm)
    echo "launched $WANDB_NAME -> job $JID  (ofr=$OFR itv=$ITV cosw=$COSW) dep=${DEP:-none}"
    IDS="$IDS${IDS:+,}$JID"
  done
done
echo "WAVE_IDS=$IDS"
