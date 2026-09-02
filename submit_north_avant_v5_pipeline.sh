#!/usr/bin/env bash
# Submit the North Avant V5 simulation and an automatic dependent postprocess job.
#
# Usage:
#   ./submit_north_avant_v5_pipeline.sh preproduction
#   ./submit_north_avant_v5_pipeline.sh production
#
# Optional environment overrides:
#   SOURCE_DIR, PFLOTRAN_BIN, RESULT_BASE, POSTPROCESS_PYTHON,
#   SIM_NODES, SIM_TASKS_PER_NODE, SIM_WALLTIME, SIM_MEMORY,
#   POSTPROCESS_MODE, PLOT_STRAIN_UNIT, PLOT_SPREAD.

set -Eeuo pipefail

MODE="${1:-preproduction}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SIM_SCRIPT:-}" ]]; then
  if [[ -s "$SCRIPT_DIR/run_north_avant_v5_simulation.slurm" ]]; then
    SIM_SCRIPT="$SCRIPT_DIR/run_north_avant_v5_simulation.slurm"
  else
    SIM_SCRIPT="$SCRIPT_DIR/slurm/run_north_avant_v5_simulation.slurm"
  fi
fi

if [[ -z "${POST_SCRIPT:-}" ]]; then
  if [[ -s "$SCRIPT_DIR/postprocess_north_avant_v5_results.slurm" ]]; then
    POST_SCRIPT="$SCRIPT_DIR/postprocess_north_avant_v5_results.slurm"
  else
    POST_SCRIPT="$SCRIPT_DIR/slurm/postprocess_north_avant_v5_results.slurm"
  fi
fi
SOURCE_DIR="${SOURCE_DIR:-/home/harhin/PLOTRAN_model_files}"
RESULT_BASE="${RESULT_BASE:-/home/harhin/pflotran_results/north_avant_v5}"
PFLOTRAN_BIN="${PFLOTRAN_BIN:-}"
POSTPROCESS_PYTHON="${POSTPROCESS_PYTHON:-$HOME/.venvs/north_avant_postprocess/bin/python}"
POSTPROCESS_MODE="${POSTPROCESS_MODE:-full}"
PLOT_STRAIN_UNIT="${PLOT_STRAIN_UNIT:-dimensionless}"
PLOT_SPREAD="${PLOT_SPREAD:-none}"
SIM_NODES="${SIM_NODES:-1}"
SIM_TASKS_PER_NODE="${SIM_TASKS_PER_NODE:-64}"
SIM_WALLTIME="${SIM_WALLTIME:-72:00:00}"
SIM_MEMORY="${SIM_MEMORY:-250G}"

if [[ -z "$PFLOTRAN_BIN" ]]; then
  for candidate in \
    /home/harhin/PFLOTRAN/pflotran/src/pflotran/pflotran \
    /home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran
  do
    if [[ -x "$candidate" ]]; then
      PFLOTRAN_BIN="$candidate"
      break
    fi
  done
fi

[[ -x "$PFLOTRAN_BIN" ]] || {
  echo "PFLOTRAN executable not found. Set PFLOTRAN_BIN=/absolute/path/to/pflotran." >&2
  exit 2
}

case "$MODE" in
  preproduction|pre|4h)
    INPUT_PREFIX="north_avant_v5_twoway_preproduction_4h"
    SIM_JOB_NAME="NAV5-PRE4H"
    POST_JOB_NAME="NAV5-POST-PRE"
    ;;
  production|prod|96h)
    INPUT_PREFIX="north_avant_v5_twoway_production_96h_final"
    SIM_JOB_NAME="NAV5-PROD96"
    POST_JOB_NAME="NAV5-POST-PROD"
    ;;
  *)
    echo "Usage: $0 {preproduction|production}" >&2
    exit 2
    ;;
esac

for path in "$SIM_SCRIPT" "$POST_SCRIPT"; do
  [[ -s "$path" ]] || {
    echo "Missing submission script: $path" >&2
    exit 2
  }
done

[[ -s "$SOURCE_DIR/${INPUT_PREFIX}.in" ]] || {
  echo "Missing selected deck: $SOURCE_DIR/${INPUT_PREFIX}.in" >&2
  exit 2
}

simulation_job_id="$(
  sbatch --parsable \
    --job-name="$SIM_JOB_NAME" \
    --nodes="$SIM_NODES" \
    --ntasks-per-node="$SIM_TASKS_PER_NODE" \
    --mem="$SIM_MEMORY" \
    --time="$SIM_WALLTIME" \
    --export=ALL,\
INPUT_PREFIX="$INPUT_PREFIX",\
SOURCE_DIR="$SOURCE_DIR",\
RESULT_BASE="$RESULT_BASE",\
PFLOTRAN_BIN="$PFLOTRAN_BIN" \
    "$SIM_SCRIPT"
)"

# --parsable can include a federation suffix after a semicolon.
simulation_job_id="${simulation_job_id%%;*}"

postprocess_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${simulation_job_id}" \
    --job-name="$POST_JOB_NAME" \
    --export=ALL,\
SIM_JOB_ID="$simulation_job_id",\
INPUT_PREFIX="$INPUT_PREFIX",\
SOURCE_DIR="$SOURCE_DIR",\
RESULT_BASE="$RESULT_BASE",\
POSTPROCESS_PYTHON="$POSTPROCESS_PYTHON",\
POSTPROCESS_MODE="$POSTPROCESS_MODE",\
PLOT_STRAIN_UNIT="$PLOT_STRAIN_UNIT",\
PLOT_SPREAD="$PLOT_SPREAD" \
    "$POST_SCRIPT"
)"
postprocess_job_id="${postprocess_job_id%%;*}"

cat <<EOF
North Avant V5 pipeline submitted
---------------------------------
Mode                  : $MODE
Input prefix          : $INPUT_PREFIX
Simulation job        : $simulation_job_id ($SIM_JOB_NAME)
Postprocess job       : $postprocess_job_id ($POST_JOB_NAME)
Dependency            : afterok:$simulation_job_id
Simulation result dir : $RESULT_BASE/${simulation_job_id}_${INPUT_PREFIX}

Monitor:
  squeue -j $simulation_job_id,$postprocess_job_id
  tail -f ${SIM_JOB_NAME}_${simulation_job_id}.out

Postprocessing starts automatically only after the simulation exits successfully.
EOF
