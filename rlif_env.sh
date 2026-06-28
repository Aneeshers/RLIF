#!/bin/bash
# Source this to set up the RLIF runtime env (mujoco210 + cuda jax + mujoco_py).
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
source /iris/u/aneeshm/envs/rlif/bin/activate
