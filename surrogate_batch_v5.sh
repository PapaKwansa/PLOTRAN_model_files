#!/bin/bash
#SBATCH --job-name=surrogate_ds_v5
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=surrogate_ds_v5_%j.out
#SBATCH --error=surrogate_ds_v5_%j.err
#SBATCH --mail-user=harhin@clemson.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load gcc
module load anaconda3

export PETSC_DIR=/home/harhin/PFLOTRAN/petsc
export PETSC_ARCH=arch-linux-c-opt
export PFLOTRAN_BIN=/home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran
export MPIEXEC=/home/harhin/PFLOTRAN/petsc/arch-linux-c-opt/bin/mpiexec.hydra

WORKDIR=/home/harhin/PLOTRAN_model_files
OUTDIR=/home/harhin/pflotran_surrogate_results/${SLURM_JOB_ID}

mkdir -p "$OUTDIR"
cd "$WORKDIR"

echo "=============================================="
echo "North Avant V5 surrogate dataset generation"
echo "=============================================="
echo "Working directory : $WORKDIR"
echo "Output directory  : $OUTDIR"
echo "PFLOTRAN          : $PFLOTRAN_BIN"
echo "MPI launcher      : $MPIEXEC"
echo "MPI tasks         : $SLURM_NTASKS"
echo "Nodes             : $SLURM_JOB_NUM_NODES"
echo "Samples           : 32"
echo "Deck              : north_avant_v5_twoway_production_96h_final.in"
echo "=============================================="

python surrogate_dataset_v5.py \
    --model-dir "$WORKDIR" \
    --out-dir "$OUTDIR" \
    --deck-template north_avant_v5_twoway_production_96h_final.in \
    --pflotran-bin "$PFLOTRAN_BIN" \
    --mpiexec "$MPIEXEC" \
    --nprocs 64 \
    --n-samples 32 \
    --seed 1234

echo "=============================================="
echo "Finished"
echo "Dataset saved to:"
echo "  $OUTDIR"
echo "=============================================="
