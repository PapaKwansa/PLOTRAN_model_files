#!/bin/bash
#SBATCH --job-name=surrogate_ds
#SBATCH --partition=work1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=surrogate_ds_%j.out
#SBATCH --error=surrogate_ds_%j.err
#SBATCH --mail-user=harhin@clemson.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load gcc
# load the Python module you normally use on Palmetto
# module load anaconda3
# or whatever module provides python, numpy, h5py

export PETSC_DIR=/home/harhin/PFLOTRAN/petsc
export PETSC_ARCH=arch-linux-c-opt
export PFLOTRAN_BIN=/home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran

WORKDIR=/home/harhin/PLOTRAN_model_files
OUTDIR="$HOME/pflotran_surrogate_results/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
cd "$WORKDIR"

echo "=== Starting surrogate dataset generation ==="

python surrogate_dataset.py \
  --model-dir "$WORKDIR" \
  --out-dir "$OUTDIR" \
  --deck-template geomech_inj_rec.in \
  --pflotran-bin "$PFLOTRAN_BIN" \
  --nprocs "$SLURM_NTASKS" \
  --n-samples 20

echo "Done."
echo "Results saved to:"
echo "  $OUTDIR"