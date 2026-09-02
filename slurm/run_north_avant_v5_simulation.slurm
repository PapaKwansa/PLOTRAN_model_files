#!/usr/bin/env bash
#
# North Avant V5 PFLOTRAN production runner for Clemson Palmetto 2.
#
# Safe default:
#   north_avant_v5_twoway_preproduction_4h
#
# Full production:
#   sbatch \
#     --export=ALL,INPUT_PREFIX=north_avant_v5_twoway_production_96h_final \
#     run_north_avant_v5_simulation.slurm
#
# Resource directives below are deliberately a one-node starting point.
# Override --nodes/--ntasks-per-node only after the 4-hour scaling test.
#
#SBATCH --job-name=NAV5-SIM
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=harhin@clemson.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=B:USR1@600

set -Eeuo pipefail
shopt -s nullglob
umask 027
ulimit -s unlimited
ulimit -c 0

# -----------------------------------------------------------------------------
# User-configurable paths and run mode
# -----------------------------------------------------------------------------

INPUT_PREFIX="${INPUT_PREFIX:-north_avant_v5_twoway_preproduction_4h}"

# Point SOURCE_DIR to the clean, frozen runtime bundle—not to a directory that
# also contains old meshes and previous PFLOTRAN outputs.
SOURCE_DIR="${SOURCE_DIR:-/home/harhin/PLOTRAN_model_files}"

MANIFEST="${MANIFEST:-north_avant_v5_runtime_manifest.txt}"
RESULT_BASE="${RESULT_BASE:-/home/harhin/pflotran_results/north_avant_v5}"

# Production-tested PETSc stack.
export PETSC_DIR="${PETSC_DIR:-/home/harhin/PFLOTRAN/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt}"
PFLOTRAN_BIN="${PFLOTRAN_BIN:-/home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran}"
export PFLOTRAN_BIN

# NAV5 MPICH-RUNTIME FIX
#
# arch-linux-c-opt contains its own MPICH/Hydra runtime. Never launch this
# PFLOTRAN executable with a module-provided OpenMPI mpiexec or prterun.
GCC_MODULE="${GCC_MODULE:-gcc}"

# This module is used only by h5ls_safe() in an isolated subprocess. It is
# never loaded into the PFLOTRAN process environment.
HDF5_MODULE="${HDF5_MODULE:-hdf5/1.14.3}"

PETSC_BIN="${PETSC_BIN:-$PETSC_DIR/$PETSC_ARCH/bin}"
PETSC_LIB="${PETSC_LIB:-$PETSC_DIR/$PETSC_ARCH/lib}"
PETSC_MPIEXEC="${PETSC_MPIEXEC:-$PETSC_BIN/mpiexec}"
PETSC_H5LS="${PETSC_H5LS:-$PETSC_BIN/h5ls}"

# Intentionally non-configurable for this validated PFLOTRAN executable.
MPI_LAUNCHER="petsc-mpiexec"

MPIEXEC_VERSION_TEXT=""
H5LS_MODE=""

# Copy newly completed checkpoint files to persistent storage during long runs.
CHECKPOINT_SYNC_SECONDS="${CHECKPOINT_SYNC_SECONDS:-600}"

# Set to 1 for a release run after the source tree has been cleaned and committed.
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-1}"

# Initialized early so error/signal traps are safe before scratch staging.
RESULT_DIR=""
WORK_DIR=""
ARCHIVE_DONE=0

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}


resolve_petsc_runtime() {
  [[ -d "$PETSC_BIN" ]]     || die "PETSc bin directory not found: $PETSC_BIN"

  [[ -d "$PETSC_LIB" ]]     || die "PETSc lib directory not found: $PETSC_LIB"

  if [[ ! -x "$PETSC_MPIEXEC" ]]; then
    if [[ -x "$PETSC_BIN/mpiexec.hydra" ]]; then
      PETSC_MPIEXEC="$PETSC_BIN/mpiexec.hydra"
    else
      die "PETSc MPICH/Hydra launcher not found under $PETSC_BIN."
    fi
  fi

  PETSC_MPIEXEC="$(readlink -f "$PETSC_MPIEXEC")"
  export PETSC_MPIEXEC

  MPIEXEC_VERSION_TEXT="$(
    "$PETSC_MPIEXEC" -version 2>&1 || true
  )"

  if grep -Eqi     'Open MPI|OpenRTE|PRRTE|prterun'     <<< "$MPIEXEC_VERSION_TEXT"
  then
    die "Resolved launcher is OpenMPI/PRRTE; expected PETSc MPICH/Hydra."
  fi

  if ! grep -Eqi     'HYDRA|MPICH'     <<< "$MPIEXEC_VERSION_TEXT"
  then
    die "Resolved launcher did not identify itself as MPICH/Hydra."
  fi

  if [[ -x "$PETSC_H5LS" ]]; then
    H5LS_MODE="petsc-bundled"
  else
    H5LS_MODE="isolated-hdf5-module"
  fi
}


h5ls_safe() {
  if [[ -x "$PETSC_H5LS" ]]; then
    "$PETSC_H5LS" "$@"
    return
  fi

  (
    module purge
    module load "$GCC_MODULE"
    module load "$HDF5_MODULE"

    command h5ls "$@"
  )
}

remove_conda_paths() {
  local input="${1:-}"
  local output=""
  local entry
  local lower

  IFS=':' read -r -a entries <<< "$input"

  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    lower="${entry,,}"

    case "$lower" in
      *anaconda*|*miniconda*|*conda*|*openmpi*|*prrte*|*pmix*)
        ;;
      *)
        if [[ -z "$output" ]]; then
          output="$entry"
        else
          output="${output}:$entry"
        fi
        ;;
    esac
  done

  printf '%s' "$output"
}

find_pflotran() {
  local candidate

  if [[ -n "${PFLOTRAN_BIN:-}" ]]; then
    [[ -x "$PFLOTRAN_BIN" ]] || die "PFLOTRAN_BIN is not executable: $PFLOTRAN_BIN"
    return
  fi

  for candidate in \
    /home/harhin/PFLOTRAN/pflotran/src/pflotran/pflotran \
    /home/harhin/PFLOTRAN/petsc/pflotran/src/pflotran/pflotran
  do
    if [[ -x "$candidate" ]]; then
      PFLOTRAN_BIN="$candidate"
      export PFLOTRAN_BIN
      return
    fi
  done

  die "PFLOTRAN executable not found. Submit with PFLOTRAN_BIN=/absolute/path/to/pflotran."
}

read_runtime_manifest() {
  local manifest_path="$1"

  mapfile -t RUNTIME_FILES < <(
    awk '
      {
        sub(/#.*/, "")
        gsub(/^[[:space:]]+|[[:space:]]+$/, "")
        if (length($0) > 0) print
      }
    ' "$manifest_path"
  )

  (( ${#RUNTIME_FILES[@]} > 0 )) || die "Runtime manifest is empty: $manifest_path"
}

verify_source_bundle() {
  local missing=0
  local relative_path

  [[ -d "$SOURCE_DIR" ]] || die "SOURCE_DIR does not exist: $SOURCE_DIR"
  [[ -s "$SOURCE_DIR/${INPUT_PREFIX}.in" ]] \
    || die "Missing input deck: $SOURCE_DIR/${INPUT_PREFIX}.in"
  [[ -s "$SOURCE_DIR/$MANIFEST" ]] \
    || die "Missing runtime manifest: $SOURCE_DIR/$MANIFEST"

  read_runtime_manifest "$SOURCE_DIR/$MANIFEST"

  for relative_path in "${RUNTIME_FILES[@]}"; do
    if [[ ! -s "$SOURCE_DIR/$relative_path" ]]; then
      printf 'MISSING OR EMPTY: %s\n' "$SOURCE_DIR/$relative_path" >&2
      missing=1
    fi
  done

  (( missing == 0 )) || die "Runtime-bundle verification failed."
}

record_git_state() {
  local git_root=""

  if git -C "$SOURCE_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    git_root="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel)"
  elif git -C "$(dirname "$SOURCE_DIR")" rev-parse --show-toplevel >/dev/null 2>&1; then
    git_root="$(git -C "$(dirname "$SOURCE_DIR")" rev-parse --show-toplevel)"
  fi

  if [[ -z "$git_root" ]]; then
    printf 'not_a_git_checkout\n' > "$RESULT_DIR/git_commit.txt"
    return
  fi

  git -C "$git_root" rev-parse HEAD \
    > "$RESULT_DIR/git_commit.txt"

  # Record everything for provenance.
  git -C "$git_root" status \
    --short \
    --untracked-files=all \
    > "$RESULT_DIR/git_status_at_submission.txt"

  # Only tracked or staged source changes invalidate the production run.
  # Untracked Slurm logs and preflight products are recorded above but do
  # not alter the committed deck, mesh, mapping, or postprocessing code.
  git -C "$git_root" status \
    --short \
    --untracked-files=no \
    > "$RESULT_DIR/git_tracked_status_at_submission.txt"

  if [[ -s "$RESULT_DIR/git_tracked_status_at_submission.txt" ]]; then
    log "WARNING: tracked source files differ from the committed release."

    if [[ "$REQUIRE_CLEAN_GIT" == "1" ]]; then
      die "REQUIRE_CLEAN_GIT=1 and tracked source files have changes."
    fi
  elif [[ -s "$RESULT_DIR/git_status_at_submission.txt" ]]; then
    log "NOTE: untracked files exist; recorded for provenance but not used."
  fi
}

sync_checkpoints_once() {
  [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR:-}" ]] || return 0
  [[ -n "${RESULT_DIR:-}" ]] || return 0

  local source_file
  local base
  local temporary
  local destination

  mkdir -p "$RESULT_DIR/checkpoints"

  for source_file in \
    "$WORK_DIR"/"${INPUT_PREFIX}"*restart*.h5 \
    "$WORK_DIR"/"${INPUT_PREFIX}"*checkpoint*.h5 \
    "$WORK_DIR"/"${INPUT_PREFIX}"*.chk
  do
    [[ -f "$source_file" ]] || continue

    base="$(basename "$source_file")"
    destination="$RESULT_DIR/checkpoints/$base"

    # Skip files already copied with the same size and a newer/equal mtime.
    if [[ -f "$destination" ]] \
      && [[ "$(stat -c %s "$source_file")" == "$(stat -c %s "$destination")" ]] \
      && [[ ! "$source_file" -nt "$destination" ]]
    then
      continue
    fi

    temporary="$RESULT_DIR/checkpoints/.${base}.tmp.${SLURM_JOB_ID}"

    rm -f "$temporary"
    rsync -a "$source_file" "$temporary"

    # Do not promote a partially written HDF5 checkpoint.
    if [[ "$source_file" == *.h5 ]]; then
      if ! h5ls_safe -r "$temporary" >/dev/null 2>&1; then
        log "Checkpoint copy is not yet a complete HDF5 file; will retry: $base"
        rm -f "$temporary"
        continue
      fi
    fi

    mv -f "$temporary" "$destination"
    log "Persisted checkpoint: $destination"
  done
}

checkpoint_sync_loop() {
  while [[ ! -f "$RUN_STATUS_FILE" ]]; do
    sleep "$CHECKPOINT_SYNC_SECONDS"
    sync_checkpoints_once || true
  done

  sync_checkpoints_once || true
}

archive_results() {
  local requested_status="${1:-0}"

  if [[ "${ARCHIVE_DONE:-0}" == "1" ]]; then
    return
  fi

  # There is nowhere to archive if failure occurred before RESULT_DIR setup.
  if [[ -z "${RESULT_DIR:-}" ]]; then
    return
  fi

  ARCHIVE_DONE=1
  mkdir -p "$RESULT_DIR"

  if [[ -d "${WORK_DIR:-}" ]]; then
    log "Archiving run directory to $RESULT_DIR"
    rsync -a --partial "$WORK_DIR/" "$RESULT_DIR/" || true
  fi

  {
    echo "archive_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "input_prefix=${INPUT_PREFIX:-unknown}"
    echo "requested_exit_status=$requested_status"
    echo "work_dir=${WORK_DIR:-unknown}"
    echo "source_dir=${SOURCE_DIR:-unknown}"
    echo "pflotran_bin=${PFLOTRAN_BIN:-unknown}"
  } > "$RESULT_DIR/archive_status.txt"
}

on_exit() {
  local status=$?
  trap - EXIT
  archive_results "$status"
  exit "$status"
}

on_signal() {
  local signal_name="$1"

  log "Received $signal_name. Persisting completed checkpoints and current outputs."
  sync_checkpoints_once || true
  archive_results 99

  # Slurm will terminate the active job step after the batch shell exits.
  exit 99
}

trap on_exit EXIT
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

[[ -n "${SLURM_JOB_ID:-}" ]] || die "This script must be run through sbatch."

module purge
module load "$GCC_MODULE"

# Remove inherited Conda/OpenMPI paths and place the PETSc architecture first.
CLEAN_PATH="$(remove_conda_paths "$PATH")"
CLEAN_LD_LIBRARY_PATH="$(
  remove_conda_paths "${LD_LIBRARY_PATH:-}"
)"

export PATH="$PETSC_BIN:$CLEAN_PATH"
export LD_LIBRARY_PATH="$PETSC_LIB${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

unset   LD_PRELOAD   CONDA_PREFIX   CONDA_DEFAULT_ENV   PYTHONPATH   PYTHONHOME   OPAL_PREFIX

hash -r

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

find_pflotran
resolve_petsc_runtime

for command_name in rsync sha256sum stat awk ldd readlink grep; do
  command -v "$command_name" >/dev/null 2>&1 \
    || die "Required command is unavailable after module loading: $command_name"
done

[[ "$MPI_LAUNCHER" == "petsc-mpiexec" ]]   || die "Unsafe MPI launcher selection: $MPI_LAUNCHER"

verify_source_bundle

# -----------------------------------------------------------------------------
# Persistent result directory and scratch staging
# -----------------------------------------------------------------------------

RESULT_DIR="$RESULT_BASE/${SLURM_JOB_ID}_${INPUT_PREFIX}"
mkdir -p "$RESULT_DIR"

record_git_state

# Hydra creates Slurm job steps whose working directory must be visible to
# every launched process. Use shared scratch even for a one-node run rather
# than a generic /tmp directory that a new Slurm step may not see.
WORK_DIR="/scratch/$USER/north_avant_v5/${SLURM_JOB_ID}_${INPUT_PREFIX}"
WORK_KIND="shared_scratch"

mkdir -p "$WORK_DIR"

log "Staging validated V5 runtime bundle."

rsync -a \
  --files-from="$SOURCE_DIR/$MANIFEST" \
  "$SOURCE_DIR/" \
  "$WORK_DIR/"

cp -a "$SOURCE_DIR/${INPUT_PREFIX}.in" "$WORK_DIR/"
cp -a "$SOURCE_DIR/$MANIFEST" "$WORK_DIR/"

# Generate checksums from the immutable source and verify the staged copy.
(
  cd "$SOURCE_DIR"
  sha256sum \
    "${INPUT_PREFIX}.in" \
    "$MANIFEST" \
    "${RUNTIME_FILES[@]}"
) > "$RESULT_DIR/INPUT_SHA256SUMS"

cp -a "$RESULT_DIR/INPUT_SHA256SUMS" "$WORK_DIR/"

(
  cd "$WORK_DIR"
  sha256sum -c INPUT_SHA256SUMS
)

# -----------------------------------------------------------------------------
# Reproducibility metadata
# -----------------------------------------------------------------------------

{
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-unknown}"
  echo "nodes=${SLURM_NNODES:-unknown}"
  echo "tasks=${SLURM_NTASKS:-unknown}"
  echo "tasks_per_node=${SLURM_NTASKS_PER_NODE:-unknown}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
  echo "input_prefix=$INPUT_PREFIX"
  echo "source_dir=$SOURCE_DIR"
  echo "work_dir=$WORK_DIR"
  echo "work_kind=$WORK_KIND"
  echo "result_dir=$RESULT_DIR"
  echo "petsc_dir=$PETSC_DIR"
  echo "petsc_arch=$PETSC_ARCH"
  echo "pflotran_bin=$PFLOTRAN_BIN"
  echo "mpi_launcher=$MPI_LAUNCHER"
  echo "mpi_exec=$PETSC_MPIEXEC"
  echo "mpi_exec_realpath=$(readlink -f "$PETSC_MPIEXEC")"
  echo "h5ls_mode=$H5LS_MODE"
  echo "start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RESULT_DIR/run_metadata.txt"

module list 2> "$RESULT_DIR/module_list.txt" || true
env | sort > "$RESULT_DIR/environment.txt"
sha256sum "$PFLOTRAN_BIN" > "$RESULT_DIR/PFLOTRAN_SHA256SUM"
sha256sum "$PETSC_MPIEXEC" > "$RESULT_DIR/MPIEXEC_SHA256SUM"
printf '%s\n' "$MPIEXEC_VERSION_TEXT" > "$RESULT_DIR/MPIEXEC_VERSION.txt"
ldd "$PFLOTRAN_BIN" > "$RESULT_DIR/PFLOTRAN_LDD.txt" 2>&1 || true

if grep -q   'not found'   "$RESULT_DIR/PFLOTRAN_LDD.txt"
then
  grep     'not found'     "$RESULT_DIR/PFLOTRAN_LDD.txt"     >&2

  die "PFLOTRAN has unresolved shared-library dependencies."
fi

if grep -Eqi   '/openmpi-|libmpi\.so\.40|libmpi_usempif08\.so\.40|libmpi_usempi_ignore_tkr\.so\.40|libmpi_mpifh\.so\.40|libopen-pal\.so|libopen-rte\.so|libprrte'   "$RESULT_DIR/PFLOTRAN_LDD.txt"
then
  grep -Ei     '/openmpi-|libmpi\.so\.40|libmpi_usempif08\.so\.40|libmpi_usempi_ignore_tkr\.so\.40|libmpi_mpifh\.so\.40|libopen-pal\.so|libopen-rte\.so|libprrte'     "$RESULT_DIR/PFLOTRAN_LDD.txt"     >&2

  die "OpenMPI contamination detected in the PFLOTRAN process."
fi

if ! grep -Fq   "libmpi.so.12 => $PETSC_LIB/"   "$RESULT_DIR/PFLOTRAN_LDD.txt"
then
  die "Expected PETSc MPICH libmpi.so.12 was not resolved."
fi
scontrol show job -dd "$SLURM_JOB_ID" > "$RESULT_DIR/slurm_job_definition.txt" 2>&1 || true
df -h "$WORK_DIR" > "$RESULT_DIR/work_filesystem.txt" 2>&1 || true

cp -a "$0" "$RESULT_DIR/$(basename "$0")" 2>/dev/null || true
cp -a "$SOURCE_DIR/${INPUT_PREFIX}.in" "$RESULT_DIR/"

log "============================================================"
log "North Avant V5 PFLOTRAN run"
log "Job ID       : $SLURM_JOB_ID"
log "Input prefix : $INPUT_PREFIX"
log "MPI tasks    : ${SLURM_NTASKS:-unknown}"
log "Nodes        : ${SLURM_NNODES:-unknown}"
log "Source       : $SOURCE_DIR"
log "Work         : $WORK_DIR ($WORK_KIND)"
log "Results      : $RESULT_DIR"
log "PFLOTRAN     : $PFLOTRAN_BIN"
log "PETSc arch   : $PETSC_ARCH"
log "Launcher     : $MPI_LAUNCHER"
log "MPI exec     : $PETSC_MPIEXEC"
log "============================================================"

# -----------------------------------------------------------------------------
# Launch PFLOTRAN
# -----------------------------------------------------------------------------

cd "$WORK_DIR"

LAUNCH_COMMAND=(
  "$PETSC_MPIEXEC"
  -n "$SLURM_NTASKS"
)

LOG_FILE="$WORK_DIR/${INPUT_PREFIX}.log"
RUN_STATUS_FILE="$WORK_DIR/.pflotran_exit_status"
rm -f "$RUN_STATUS_FILE"

log "Launching PFLOTRAN."

(
  set +e
  "${LAUNCH_COMMAND[@]}" \
    "$PFLOTRAN_BIN" \
    -input_prefix "$INPUT_PREFIX"
  run_status=$?
  printf '%s\n' "$run_status" > "$RUN_STATUS_FILE"
  exit "$run_status"
) > >(tee "$LOG_FILE") 2>&1 &

RUN_WRAPPER_PID=$!

checkpoint_sync_loop &
CHECKPOINT_SYNC_PID=$!

set +e
wait "$RUN_WRAPPER_PID"
wrapper_status=$?
set -e

# Give the tee process a moment to flush the final lines.
sleep 1

if [[ -f "$RUN_STATUS_FILE" ]]; then
  PFLOTRAN_STATUS="$(cat "$RUN_STATUS_FILE")"
else
  PFLOTRAN_STATUS=99
  log "WARNING: no PFLOTRAN status file was written; wrapper status=$wrapper_status"
fi

kill "$CHECKPOINT_SYNC_PID" >/dev/null 2>&1 || true
wait "$CHECKPOINT_SYNC_PID" >/dev/null 2>&1 || true
sync_checkpoints_once || true

printf 'PFLOTRAN exit status: %s\n' "$PFLOTRAN_STATUS" | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# Post-run validation
# -----------------------------------------------------------------------------

VALIDATION_STATUS=0

if [[ "$PFLOTRAN_STATUS" != "0" ]]; then
  log "PFLOTRAN returned nonzero status $PFLOTRAN_STATUS."
  VALIDATION_STATUS=1
fi

HDF5_COUNT=0
for h5_file in "$WORK_DIR"/"${INPUT_PREFIX}"*.h5; do
  [[ -f "$h5_file" ]] || continue
  HDF5_COUNT=$((HDF5_COUNT + 1))

  if ! h5ls_safe -r "$h5_file" \
    > "$WORK_DIR/$(basename "${h5_file%.h5}")_h5_tree.txt"
  then
    log "Unreadable HDF5 output: $h5_file"
    VALIDATION_STATUS=1
  fi
done

if (( HDF5_COUNT == 0 )); then
  log "No HDF5 result files were produced."
  VALIDATION_STATUS=1
fi

CRITICAL_PATTERN='fatal|diverg|nan|infinite|zero pivot|singular|maximum consecutive|caught signal|segmentation'
if grep -aEin "$CRITICAL_PATTERN" "$LOG_FILE" \
  > "$WORK_DIR/${INPUT_PREFIX}_critical_failure_scan.txt"
then
  log "Critical failure terms were found in the PFLOTRAN log."
  VALIDATION_STATUS=1
else
  : > "$WORK_DIR/${INPUT_PREFIX}_critical_failure_scan.txt"
fi

grep -aE \
  'Step|FLOW TS|GEOMECHANICS TS|Wall Clock|PFLOTRAN exit status|Dump checkpoint|Restarting with checkpoint' \
  "$LOG_FILE" \
  | tail -n 250 \
  > "$WORK_DIR/${INPUT_PREFIX}_final_log_summary.txt" || true

EXPECTED_FINAL_TOKEN="$(
  awk '$1 == "FINAL_TIME" {print $2; exit}' "${INPUT_PREFIX}.in"
)"

LAST_TIME_TOKEN="$(
  {
    grep -E '^[[:space:]]*Step[[:space:]]+[0-9]+[[:space:]]+Time=' "$LOG_FILE" \
      || true
  } \
    | tail -n 1 \
    | sed -E 's/.*Time=[[:space:]]*([^[:space:]]+).*/\1/'
)"

{
  echo "pflotran_exit_status=$PFLOTRAN_STATUS"
  echo "validation_status=$VALIDATION_STATUS"
  echo "expected_final_time_token=$EXPECTED_FINAL_TOKEN"
  echo "last_reported_time_token=$LAST_TIME_TOKEN"
  echo "hdf5_file_count=$HDF5_COUNT"
  echo "finish_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$WORK_DIR/RUN_STATUS.txt"

log "PFLOTRAN exit status : $PFLOTRAN_STATUS"
log "Validation status     : $VALIDATION_STATUS"
log "Expected final time  : $EXPECTED_FINAL_TOKEN"
log "Last reported time   : $LAST_TIME_TOKEN"

# Archive before returning the final status.
archive_results "$VALIDATION_STATUS"

if (( VALIDATION_STATUS != 0 )); then
  exit 1
fi

log "North Avant V5 run and post-run validation completed successfully."
exit 0
