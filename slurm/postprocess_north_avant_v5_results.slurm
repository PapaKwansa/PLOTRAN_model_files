#!/usr/bin/env bash
# Automatic dependent postprocessing for a completed North Avant V5 PFLOTRAN run.
#SBATCH --job-name=NAV5-POST
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=harhin@clemson.edu
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail
shopt -s nullglob
umask 027
ulimit -s unlimited
ulimit -c 0

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

SIM_JOB_ID="${SIM_JOB_ID:?SIM_JOB_ID must identify the successful simulation job}"
INPUT_PREFIX="${INPUT_PREFIX:?INPUT_PREFIX must identify the completed PFLOTRAN case}"
RESULT_BASE="${RESULT_BASE:-/home/harhin/pflotran_results/north_avant_v5}"
RESULT_DIR="${RESULT_BASE}/${SIM_JOB_ID}_${INPUT_PREFIX}"
SOURCE_DIR="${SOURCE_DIR:-/home/harhin/PLOTRAN_model_files}"
SCRIPT_DIR="${POSTPROCESS_SCRIPT_DIR:-$SOURCE_DIR/scripts/postprocess}"
POSTPROCESS_PYTHON="${POSTPROCESS_PYTHON:-$HOME/.venvs/north_avant_postprocess/bin/python}"
POSTPROCESS_MODE="${POSTPROCESS_MODE:-full}"  # full or latest
PLOT_STRAIN_UNIT="${PLOT_STRAIN_UNIT:-dimensionless}"
PLOT_SPREAD="${PLOT_SPREAD:-none}"

FLOW_H5="$RESULT_DIR/${INPUT_PREFIX}.h5"
GEOMECH_H5="$RESULT_DIR/${INPUT_PREFIX}-geomech.h5"
UGI="$SOURCE_DIR/bartlesville_hec_lime_v5_interfaces.ugi"
MAPPING="$SOURCE_DIR/bartlesville_hec_lime_v5_interfaces_median.mapping"
COUPLED_CONVERTER="$SCRIPT_DIR/pflotran_coupled_to_vtu.py"
REGION_PLOTTER="$SCRIPT_DIR/pflotran_region_timeseries_plots.py"
FINAL_POST_DIR="$RESULT_DIR/postprocess"

[[ -d "$RESULT_DIR" ]] || die "Simulation result directory is missing: $RESULT_DIR"
[[ -x "$POSTPROCESS_PYTHON" ]] || die "Postprocessing Python is not executable: $POSTPROCESS_PYTHON"

for required in \
  "$FLOW_H5" \
  "$GEOMECH_H5" \
  "$UGI" \
  "$MAPPING" \
  "$COUPLED_CONVERTER" \
  "$REGION_PLOTTER" \
  "$SOURCE_DIR/hec.vset" \
  "$SOURCE_DIR/injection_borehole.vset" \
  "$SOURCE_DIR/AVN2.vset" \
  "$SOURCE_DIR/AVN87.vset" \
  "$SOURCE_DIR/AVN31.vset"
do
  [[ -s "$required" ]] || die "Required postprocessing input is missing or empty: $required"
done

module purge

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset PYTHONPATH PYTHONHOME

if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
  WORK_DIR="$TMPDIR/north_avant_v5_post_${SLURM_JOB_ID}"
else
  WORK_DIR="/scratch/$USER/north_avant_v5/post_${SLURM_JOB_ID}"
fi

mkdir -p \
  "$WORK_DIR/paraview_coupled" \
  "$WORK_DIR/region_timeseries" \
  "$WORK_DIR/logs" \
  "$FINAL_POST_DIR"

ARCHIVE_DONE=0
archive_postprocess() {
  local status="${1:-0}"
  if [[ "$ARCHIVE_DONE" == "1" ]]; then
    return
  fi
  ARCHIVE_DONE=1

  if [[ -d "$WORK_DIR" ]]; then
    rsync -a --delete "$WORK_DIR/" "$FINAL_POST_DIR/" || true
  fi

  {
    echo "simulation_job_id=$SIM_JOB_ID"
    echo "postprocess_job_id=${SLURM_JOB_ID:-unknown}"
    echo "input_prefix=$INPUT_PREFIX"
    echo "exit_status=$status"
    echo "postprocess_mode=$POSTPROCESS_MODE"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$FINAL_POST_DIR/POSTPROCESS_STATUS.txt"
}

on_exit() {
  local status=$?
  trap - EXIT
  archive_postprocess "$status"
  exit "$status"
}
trap on_exit EXIT

exec > >(tee "$WORK_DIR/logs/postprocess_${SLURM_JOB_ID}.log") 2>&1

log "============================================================"
log "North Avant V5 automatic postprocessing"
log "Simulation job : $SIM_JOB_ID"
log "Postprocess job: ${SLURM_JOB_ID:-unknown}"
log "Input prefix   : $INPUT_PREFIX"
log "Raw results    : $RESULT_DIR"
log "Work directory : $WORK_DIR"
log "Final products : $FINAL_POST_DIR"
log "Python         : $POSTPROCESS_PYTHON"
log "Mode           : $POSTPROCESS_MODE"
log "============================================================"

"$POSTPROCESS_PYTHON" - <<'PY'
import sys
import h5py
import matplotlib
import meshio
import numpy
print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("h5py:", h5py.__version__)
print("meshio:", meshio.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

# Validate and inventory both raw HDF5 files using the same Python environment
# that will read them during conversion.
"$POSTPROCESS_PYTHON" - "$FLOW_H5" "$GEOMECH_H5" "$WORK_DIR/logs" <<'PY'
from pathlib import Path
import sys
import h5py

outdir = Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)

for filename, label in ((sys.argv[1], "flow"), (sys.argv[2], "geomech")):
    path = Path(filename)
    lines = []
    with h5py.File(path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(f"/{name} Dataset shape={obj.shape} dtype={obj.dtype}")
            else:
                lines.append(f"/{name} Group")
        h5.visititems(visitor)
    (outdir / f"{label}_h5_tree.txt").write_text("\n".join(lines) + "\n")
    print(f"HDF5 validation passed: {path}")
PY

LATEST_ARGUMENT=()
case "$POSTPROCESS_MODE" in
  full) ;;
  latest) LATEST_ARGUMENT=(--latest-only) ;;
  *) die "POSTPROCESS_MODE must be 'full' or 'latest', not '$POSTPROCESS_MODE'" ;;
esac

log "Creating coupled VTU/PVD output with mapped flow and mechanics fields."
# The converter positional order is: geomechanics HDF5, flow HDF5, UGI, mapping.
"$POSTPROCESS_PYTHON" "$COUPLED_CONVERTER" \
  "$GEOMECH_H5" \
  "$FLOW_H5" \
  "$UGI" \
  "$MAPPING" \
  --output-dir "$WORK_DIR/paraview_coupled" \
  --prefix "${INPUT_PREFIX}_coupled" \
  --vset "HEC=$SOURCE_DIR/hec.vset" \
  --vset "Injection=$SOURCE_DIR/injection_borehole.vset" \
  --vset "AVN2=$SOURCE_DIR/AVN2.vset" \
  --vset "AVN87=$SOURCE_DIR/AVN87.vset" \
  --vset "AVN31=$SOURCE_DIR/AVN31.vset" \
  "${LATEST_ARGUMENT[@]}" \
  2>&1 | tee "$WORK_DIR/logs/coupled_to_vtu.log"

AXIS_ARGUMENTS=()
[[ -n "${AVN2_AXIS:-}" ]] && AXIS_ARGUMENTS+=(--axis "AVN2=$AVN2_AXIS")
[[ -n "${AVN87_AXIS:-}" ]] && AXIS_ARGUMENTS+=(--axis "AVN87=$AVN87_AXIS")
[[ -n "${AVN31_AXIS:-}" ]] && AXIS_ARGUMENTS+=(--axis "AVN31=$AVN31_AXIS")

log "Creating regional strain/displacement CSV, PVD, PNG, and PDF products."
"$POSTPROCESS_PYTHON" "$REGION_PLOTTER" \
  "$GEOMECH_H5" \
  --output-dir "$WORK_DIR/region_timeseries" \
  --region "AVN2=$SOURCE_DIR/AVN2.vset" \
  --region "AVN87=$SOURCE_DIR/AVN87.vset" \
  --region "AVN31=$SOURCE_DIR/AVN31.vset" \
  --region "Injection=$SOURCE_DIR/injection_borehole.vset" \
  --region "HEC=$SOURCE_DIR/hec.vset" \
  "${AXIS_ARGUMENTS[@]}" \
  --plot-spread "$PLOT_SPREAD" \
  --strain-unit "$PLOT_STRAIN_UNIT" \
  --include-spread \
  2>&1 | tee "$WORK_DIR/logs/region_timeseries.log"

PVD_COUNT="$(find "$WORK_DIR" -type f -name '*.pvd' | wc -l)"
VTU_COUNT="$(find "$WORK_DIR" -type f -name '*.vtu' | wc -l)"
PNG_COUNT="$(find "$WORK_DIR" -type f -name '*.png' | wc -l)"
PDF_COUNT="$(find "$WORK_DIR" -type f -name '*.pdf' | wc -l)"
CSV_COUNT="$(find "$WORK_DIR" -type f -name '*.csv' | wc -l)"

(( PVD_COUNT >= 2 )) || die "Expected at least two PVD products, found $PVD_COUNT"
(( VTU_COUNT > 0 )) || die "No VTU files were generated"
(( PNG_COUNT > 0 )) || die "No PNG figures were generated"
(( PDF_COUNT > 0 )) || die "No PDF figures were generated"
(( CSV_COUNT > 0 )) || die "No CSV time-series files were generated"

"$POSTPROCESS_PYTHON" - "$WORK_DIR" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root = Path(sys.argv[1])
files = sorted(root.rglob("*.pvd"))
if not files:
    raise SystemExit("No PVD files found")
for path in files:
    ET.parse(path)
    print("PVD XML passed:", path)
PY

{
  echo "simulation_job_id=$SIM_JOB_ID"
  echo "postprocess_job_id=${SLURM_JOB_ID:-unknown}"
  echo "input_prefix=$INPUT_PREFIX"
  echo "postprocess_mode=$POSTPROCESS_MODE"
  echo "pvd_count=$PVD_COUNT"
  echo "vtu_count=$VTU_COUNT"
  echo "png_count=$PNG_COUNT"
  echo "pdf_count=$PDF_COUNT"
  echo "csv_count=$CSV_COUNT"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$WORK_DIR/POSTPROCESS_PRODUCT_COUNTS.txt"

cp -a "$COUPLED_CONVERTER" "$REGION_PLOTTER" "$WORK_DIR/"
cp -a "$0" "$WORK_DIR/$(basename "$0")" 2>/dev/null || true

(
  cd "$WORK_DIR"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
  sha256sum -c SHA256SUMS
)

archive_postprocess 0

log "Postprocessing completed successfully."
log "Coupled ParaView series : $FINAL_POST_DIR/paraview_coupled/${INPUT_PREFIX}_coupled.pvd"
log "Region plots            : $FINAL_POST_DIR/region_timeseries/plots"
log "Region CSV/PVD           : $FINAL_POST_DIR/region_timeseries"
exit 0
