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
module load anaconda3        # or your normal Python module

export PETSC_DIR=/home/harhin/PFLOTRAN/petsc
export PETSC_ARCH=arch-linux-c-opt
export PFLOTRAN_BIN=/home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran

WORKDIR=/home/harhin/PLOTRAN_model_files
OUTDIR=/scratch/$USER/pflotran_surrogate_results/${SLURM_JOB_ID}

mkdir -p "$OUTDIR"
cd "$WORKDIR"

echo "======================================"
echo "Starting surrogate dataset generation"
echo "======================================"
echo "Working directory : $WORKDIR"
echo "Output directory  : $OUTDIR"
echo "PFLOTRAN          : $PFLOTRAN_BIN"
echo "MPI tasks         : $SLURM_NTASKS"
echo "Samples           : 1"
echo ""

python surrogate_dataset.py \
    --model-dir "$WORKDIR" \
    --out-dir "$OUTDIR" \
    --deck-template geomech_inj_rec.in \
    --pflotran-bin "$PFLOTRAN_BIN" \
    --nprocs "$SLURM_NTASKS" \
    --n-samples 1

echo ""
echo "======================================"
echo "Finished"
echo "Dataset saved to:"
echo "  $OUTDIR"
echo "======================================"