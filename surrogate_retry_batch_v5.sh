#!/bin/bash
#SBATCH --job-name=surrogate_retry_v5
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=surrogate_retry_v5_%j.out
#SBATCH --error=surrogate_retry_v5_%j.err
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
DATASET_DIR=/home/harhin/pflotran_surrogate_results/15555789

mkdir -p "$DATASET_DIR"
cd "$WORKDIR"

# IMPORTANT: reuse the original dataset directory so the existing 24 successful
# samples are retained and only failed samples are retried.
python surrogate_dataset_v5.py \
    --model-dir "$WORKDIR" \
    --out-dir "$DATASET_DIR" \
    --deck-template north_avant_v5_twoway_production_96h_final.in \
    --pflotran-bin "$PFLOTRAN_BIN" \
    --mpiexec "$MPIEXEC" \
    --nprocs 64 \
    --n-samples 32 \
    --seed 1234 \
    --resume \
    --max-retries 2
