#!/bin/bash
#SBATCH --job-name=surrogate_expand_v5
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=surrogate_expand_v5_%j.out
#SBATCH --error=surrogate_expand_v5_%j.err
#SBATCH --mail-user=harhin@clemson.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ============================================================
# NORTH AVANT V5 SURROGATE DATASET EXPANSION
#
# PURPOSE
# -------
# Existing dataset:
#     30 successful PFLOTRAN realizations
#
# Expansion:
#     38 NEW permeability realizations
#
# Combined target:
#     30 + 38 = 68 potential successful realizations
#
# The Python generator reads the existing dataset only as a
# parameter-space reference. The new simulations are written
# to a completely separate SLURM-job output directory.
#
# IMPORTANT:
# This is NOT a retry of the original LHS experiment.
# ============================================================


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

module purge
module load gcc
module load anaconda3

export PETSC_DIR=/home/harhin/PFLOTRAN/petsc
export PETSC_ARCH=arch-linux-c-opt

export PFLOTRAN_BIN=/home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran

export MPIEXEC=/home/harhin/PFLOTRAN/petsc/arch-linux-c-opt/bin/mpiexec.hydra


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

WORKDIR=/home/harhin/PLOTRAN_model_files

# Immutable existing dataset:
# 32 originally requested, 30 successful.
OLD_DATASET_DIR=/home/harhin/pflotran_surrogate_results/15555789

# NEW expansion batch.
# Never write expansion samples into OLD_DATASET_DIR.
OUTDIR=/home/harhin/pflotran_surrogate_results/expansion_${SLURM_JOB_ID}

mkdir -p "$OUTDIR"

cd "$WORKDIR"


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

N_NEW_SAMPLES=38
CANDIDATE_POOL_SIZE=8192
EXPANSION_SEED=20260907
MAX_RETRIES=3

DECK=north_avant_v5_twoway_production_96h_final.in


# ------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------

if [ ! -d "$OLD_DATASET_DIR" ]; then
    echo "ERROR: existing dataset directory does not exist:"
    echo "  $OLD_DATASET_DIR"
    exit 1
fi

if [ ! -f "$OLD_DATASET_DIR/dataset_master.npz" ]; then
    echo "ERROR: existing master dataset does not exist:"
    echo "  $OLD_DATASET_DIR/dataset_master.npz"
    exit 1
fi

if [ ! -f "$WORKDIR/surrogate_dataset_v5.py" ]; then
    echo "ERROR: surrogate_dataset_v5.py not found in:"
    echo "  $WORKDIR"
    exit 1
fi

if [ ! -f "$WORKDIR/$DECK" ]; then
    echo "ERROR: PFLOTRAN deck not found:"
    echo "  $WORKDIR/$DECK"
    exit 1
fi

if [ ! -x "$PFLOTRAN_BIN" ]; then
    echo "ERROR: PFLOTRAN executable not found/executable:"
    echo "  $PFLOTRAN_BIN"
    exit 1
fi

if [ ! -x "$MPIEXEC" ]; then
    echo "ERROR: MPI launcher not found/executable:"
    echo "  $MPIEXEC"
    exit 1
fi


# ------------------------------------------------------------
# Run information
# ------------------------------------------------------------

echo "============================================================"
echo "North Avant V5 surrogate dataset EXPANSION"
echo "============================================================"
echo "Working directory       : $WORKDIR"
echo "Existing dataset        : $OLD_DATASET_DIR"
echo "Expansion output        : $OUTDIR"
echo "PFLOTRAN                : $PFLOTRAN_BIN"
echo "MPI launcher            : $MPIEXEC"
echo "MPI tasks               : $SLURM_NTASKS"
echo "Nodes                   : $SLURM_JOB_NUM_NODES"
echo "New samples requested   : $N_NEW_SAMPLES"
echo "Expected old successes  : 30"
echo "Combined target         : 68 realizations"
echo "Candidate pool          : $CANDIDATE_POOL_SIZE"
echo "Expansion seed          : $EXPANSION_SEED"
echo "Maximum retries/sample  : $MAX_RETRIES"
echo "Deck                     : $DECK"
echo "============================================================"


# ------------------------------------------------------------
# Expansion run
#
# The Python script will:
#
# 1. read the existing 30 successful k_log10 vectors;
# 2. normalize the five permeability dimensions;
# 3. generate 8192 candidate vectors;
# 4. select N_NEW_SAMPLES complementary points using greedy maximin;
# 5. verify no old/new or new/new duplicates;
# 6. save expansion_design.csv;
# 7. run the selected new PFLOTRAN realizations;
# 8. retry failed realizations using the SAME assigned vector;
# 9. save the successful expansion dataset separately.
# ------------------------------------------------------------

python surrogate_dataset_v5.py \
    --model-dir "$WORKDIR" \
    --out-dir "$OUTDIR" \
    --existing-dataset "$OLD_DATASET_DIR" \
    --n-new-samples "$N_NEW_SAMPLES" \
    --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
    --seed "$EXPANSION_SEED" \
    --deck-template "$DECK" \
    --pflotran-bin "$PFLOTRAN_BIN" \
    --mpiexec "$MPIEXEC" \
    --nprocs "$SLURM_NTASKS" \
    --max-retries "$MAX_RETRIES"


echo "============================================================"
echo "Expansion job finished"
echo "============================================================"
echo "Existing dataset remains:"
echo "  $OLD_DATASET_DIR"
echo
echo "New expansion dataset:"
echo "  $OUTDIR"
echo
echo "Expected files include:"
echo "  expansion_design.csv"
echo "  dataset_master.npz"
echo "  sample_manifest.csv"
echo "  dataset_metadata.json"
echo "  retry_history.json"
echo "============================================================"