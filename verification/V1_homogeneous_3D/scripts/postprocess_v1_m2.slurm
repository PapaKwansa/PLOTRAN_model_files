#!/usr/bin/env bash
#SBATCH --job-name=V1-POST
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
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

SIM_JOB_ID="${SIM_JOB_ID:?SIM_JOB_ID must be supplied}"
INPUT_PREFIX="${INPUT_PREFIX:-v1_m2_twoway_96h}"

RESULT_BASE="${RESULT_BASE:-$HOME/pflotran_results/v1_homogeneous_3D}"
RESULT_DIR="${RESULT_BASE}/${SIM_JOB_ID}_${INPUT_PREFIX}"

REPO_DIR="${REPO_DIR:-$HOME/Pflotran_files_codes_nav5_release}"

POSTPROCESS_PYTHON="${POSTPROCESS_PYTHON:-$HOME/.venvs/north_avant_postprocess/bin/python}"

SOURCE_DIR="$REPO_DIR"
SCRIPT_DIR="$REPO_DIR/scripts/postprocess"

FLOW_H5="$RESULT_DIR/${INPUT_PREFIX}.h5"
GEOMECH_H5="$RESULT_DIR/${INPUT_PREFIX}-geomech.h5"

UGI="$RESULT_DIR/v1_homogeneous_3d_m2.ugi"
MAPPING="$RESULT_DIR/v1_homogeneous_3d_m2_median.mapping"

COUPLED_CONVERTER="$SCRIPT_DIR/pflotran_coupled_to_vtu.py"
REGION_PLOTTER="$SCRIPT_DIR/pflotran_region_timeseries_plots.py"

FINAL_POST_DIR="$RESULT_DIR/postprocess"

[[ -d "$RESULT_DIR" ]] || die "Missing result directory: $RESULT_DIR"
[[ -x "$POSTPROCESS_PYTHON" ]] || die "Postprocessing Python is not executable: $POSTPROCESS_PYTHON"

for required in \
  "$FLOW_H5" \
  "$GEOMECH_H5" \
  "$UGI" \
  "$MAPPING" \
  "$COUPLED_CONVERTER" \
  "$REGION_PLOTTER" \
  "$RESULT_DIR/vsets/injector.vset"
do
  [[ -s "$required" ]] || die "Missing or empty required input: $required"
done

module purge
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0}"
module load "$ANACONDA_MODULE"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset PYTHONPATH PYTHONHOME

if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
  WORK_DIR="$TMPDIR/v1_m2_post_${SLURM_JOB_ID}"
else
  WORK_DIR="/scratch/$USER/v1_homogeneous_3D/post_${SLURM_JOB_ID}"
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
log "V1 M2 automatic postprocessing"
log "Simulation job : $SIM_JOB_ID"
log "Postprocess job: ${SLURM_JOB_ID:-unknown}"
log "Input prefix   : $INPUT_PREFIX"
log "Result dir     : $RESULT_DIR"
log "Work directory : $WORK_DIR"
log "Final products : $FINAL_POST_DIR"
log "============================================================"

"$POSTPROCESS_PYTHON" - <<'PY'
import h5py
import matplotlib
import meshio
import numpy
import sys

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("h5py:", h5py.__version__)
print("meshio:", meshio.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

"$POSTPROCESS_PYTHON" - "$FLOW_H5" "$GEOMECH_H5" "$WORK_DIR/logs" <<'PY'
from pathlib import Path
import h5py
import sys

flow_path = Path(sys.argv[1])
geomech_path = Path(sys.argv[2])
outdir = Path(sys.argv[3])

outdir.mkdir(parents=True, exist_ok=True)

for path, label in (
    (flow_path, "flow"),
    (geomech_path, "geomech"),
):
    lines = []

    with h5py.File(path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(
                    f"/{name} Dataset shape={obj.shape} dtype={obj.dtype}"
                )
            else:
                lines.append(f"/{name} Group")

        h5.visititems(visitor)

    (outdir / f"{label}_h5_tree.txt").write_text(
        "\n".join(lines) + "\n"
    )

    print(f"HDF5 validation passed: {path}")
PY

log "Creating coupled VTU/PVD products."

"$POSTPROCESS_PYTHON" "$COUPLED_CONVERTER" \
  "$GEOMECH_H5" \
  "$FLOW_H5" \
  "$UGI" \
  "$MAPPING" \
  --output-dir "$WORK_DIR/paraview_coupled" \
  --prefix "${INPUT_PREFIX}_coupled" \
  --vset "Injection=$RESULT_DIR/vsets/injector.vset" \
  2>&1 | tee "$WORK_DIR/logs/coupled_to_vtu.log"

log "Creating injector regional time-series products."

"$POSTPROCESS_PYTHON" "$REGION_PLOTTER" \
  "$GEOMECH_H5" \
  --flow-h5 "$FLOW_H5" \
  --mapping "$MAPPING" \
  --output-dir "$WORK_DIR/region_timeseries" \
  --region "Injection=$RESULT_DIR/vsets/injector.vset" \
  --pressure-region Injection \
  --plot-spread none \
  --strain-unit dimensionless \
  --include-spread \
  2>&1 | tee "$WORK_DIR/logs/region_timeseries.log"

PVD_COUNT="$(find "$WORK_DIR" -type f -name '*.pvd' | wc -l)"
VTU_COUNT="$(find "$WORK_DIR" -type f -name '*.vtu' | wc -l)"
PNG_COUNT="$(find "$WORK_DIR" -type f -name '*.png' | wc -l)"
PDF_COUNT="$(find "$WORK_DIR" -type f -name '*.pdf' | wc -l)"
CSV_COUNT="$(find "$WORK_DIR" -type f -name '*.csv' | wc -l)"

(( PVD_COUNT >= 2 )) || die "Expected at least two PVD files; found $PVD_COUNT"
(( VTU_COUNT > 0 )) || die "No VTU files generated"
(( PNG_COUNT > 0 )) || die "No PNG files generated"
(( PDF_COUNT > 0 )) || die "No PDF files generated"
(( CSV_COUNT > 0 )) || die "No CSV files generated"

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
  echo "pvd_count=$PVD_COUNT"
  echo "vtu_count=$VTU_COUNT"
  echo "png_count=$PNG_COUNT"
  echo "pdf_count=$PDF_COUNT"
  echo "csv_count=$CSV_COUNT"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$WORK_DIR/POSTPROCESS_PRODUCT_COUNTS.txt"

cp -a \
  "$COUPLED_CONVERTER" \
  "$REGION_PLOTTER" \
  "$WORK_DIR/"

cp -a "$0" \
  "$WORK_DIR/$(basename "$0")" \
  2>/dev/null || true

(
  cd "$WORK_DIR"

  find . \
    -type f \
    ! -name SHA256SUMS \
    ! -path './logs/*' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS

  sha256sum -c SHA256SUMS
)

archive_postprocess 0

log "V1 postprocessing completed successfully."
log "Coupled ParaView PVD: $FINAL_POST_DIR/paraview_coupled/${INPUT_PREFIX}_coupled.pvd"
log "Region PVD: $FINAL_POST_DIR/region_timeseries/region_timeseries.pvd"
