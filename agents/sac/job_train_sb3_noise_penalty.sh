#!/bin/bash
#SBATCH --job-name=train_sb3_noise_penalty
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --account=rrg-xilinliu
#SBATCH --mail-user=yaqian.xu@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/yqxu/links/scratch/RL-AD/%x-%j.out
module load python
module load scipy-stack
export PYTHONUNBUFFERED=1
export JOB_BASEDIR="${SCRATCH:-/tmp}/RL-AD/${SLURM_JOB_ID}"
mkdir -p "$JOB_BASEDIR"

export LOGDIR="$JOB_BASEDIR/logs"
mkdir -p "$LOGDIR"

export CKPTDIR="$JOB_BASEDIR/checkpoints"
mkdir -p "$CKPTDIR"

export PLOTDIR="$JOB_BASEDIR/plots"
mkdir -p "$PLOTDIR"

export MPLCONFIGDIR="$JOB_BASEDIR/matplotlib"
export XDG_CACHE_HOME="$JOB_BASEDIR/xdg-cache"
export PYTHONPYCACHEPREFIX="$JOB_BASEDIR/pycache"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig" "$PYTHONPYCACHEPREFIX"

export MPLBACKEND=Agg
echo "[$(date)] starting job on $HOSTNAME"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "JOB_BASEDIR=$JOB_BASEDIR"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
uv run train_sb3_noise_penalty.py
echo "[$(date)] Job finished"
