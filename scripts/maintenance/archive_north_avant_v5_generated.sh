#!/usr/bin/env bash
#
# Safely archive generated and obsolete North Avant files outside the Git tree.
#
# Dry-run is the default:
#   bash archive_north_avant_v5_generated.sh
#
# Apply the exact displayed moves:
#   bash archive_north_avant_v5_generated.sh --apply
#
# Include V5 build intermediates only after the frozen runtime bundle and its
# SHA256SUMS have passed:
#   bash archive_north_avant_v5_generated.sh --apply --include-v5-build
#
# This script NEVER removes the authoritative final decks, production scripts,
# validated median UGE, canonical UGI, validated mapping, material HDF5,
# boundary_ex_v5 directory, or required vsets.

set -Eeuo pipefail
shopt -s nullglob dotglob

APPLY=0
INCLUDE_V5_BUILD=0
INCLUDE_LEGACY_CODE=0

for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --include-v5-build) INCLUDE_V5_BUILD=1 ;;
    --include-legacy-code) INCLUDE_LEGACY_CODE=1 ;;
    *)
      echo "Usage: $0 [--apply] [--include-v5-build] [--include-legacy-code]" >&2
      exit 2
      ;;
  esac
done

ROOT="$(pwd -P)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$(dirname "$ROOT")/Pflotran_files_codes_archive/${STAMP}}"

PROTECTED=(
  north_avant_v5_twoway_preproduction_4h.in
  north_avant_v5_twoway_production_96h_final.in
  run_north_avant_v5_simulation.slurm
  postprocess_north_avant_v5_results.slurm
  submit_north_avant_v5_pipeline.sh
  pflotran_coupled_to_vtu.py
  pflotran_region_timeseries_plots.py
  preflight_north_avant_v5_bundle.py
  north_avant_v5_runtime_manifest.txt
  requirements-postprocess.txt
  NORTH_AVANT_V5_PRODUCTION_PHYSICS.md
  README_PIPELINE.md
  build_poly_layers4.py
  layers4_get_material_boundary_tags.py
  tetgen_quality_report_localized.py
  tetgen_to_avs_ugi_canonical.py
  material_h5_from_txt.py
  validate_uge_and_write_mapping.py
  build_boundary_ex.py
  compare_oneway_twoway.py
  compare_continuous_restart.py
  restart_acceptance_gate.py
  audit_xmf_h5_v2.py
  bartlesville_hec_lime_v5_interfaces_median.uge
  bartlesville_hec_lime_v5_interfaces.ugi
  bartlesville_hec_lime_v5_interfaces_median.mapping
  bartlesville_hec_lime_v5_interfaces_material_ids.h5
  boundary_ex_v5
  overburden.vset
  shallow_limestone.vset
  bartlesville_sand.vset
  basal_layer.vset
  underburden.vset
  hec.vset
  injection_borehole.vset
  strainmeter_sensors.vset
  AVN2.vset
  AVN87.vset
  AVN31.vset
  top.vset
  bottom.vset
  north.vset
  south.vset
  east.vset
  west.vset
  validation
  north_avant_v5_palmetto_bundle
)

is_protected() {
  local candidate="$1"
  local item
  for item in "${PROTECTED[@]}"; do
    if [[ "$candidate" == "$item" || "$candidate" == "$item/"* ]]; then
      return 0
    fi
  done
  return 1
}

CANDIDATES=()

add_matches() {
  local path
  for path in "$@"; do
    [[ -e "$path" || -L "$path" ]] || continue
    is_protected "$path" && continue
    CANDIDATES+=("$path")
  done
}

# Old base mesh and failed/intermediate mesh families.
add_matches \
  bartlesville_hec.1.* \
  bartlesville_hec.inp \
  bartlesville_hec.mapping \
  bartlesville_hec.poly \
  bartlesville_hec.trn \
  bartlesville_hec.uge \
  bartlesville_hec.ugi \
  bartlesville_hec_all.vset \
  bartlesville_hec_avs_validation.txt \
  bartlesville_hec_borehole_tag_report.csv \
  bartlesville_hec_boundaries.txt \
  bartlesville_hec_geometry.json \
  bartlesville_hec_hec_* \
  bartlesville_hec_material_* \
  bartlesville_hec_materials.txt \
  bartlesville_hec_mesh_counts.txt \
  bartlesville_hec_quality* \
  bartlesville_hec_refinement_* \
  bartlesville_hec_refinement_targets.csv \
  bartlesville_hec_skipped.* \
  bartlesville_hec_strainmeters.csv \
  bartlesville_hec_tube_refinement_profile.csv \
  bartlesville_hec_vertical_grading.csv \
  bartlesville_hec_voronoi.log \
  bartlesville_hec_lime_v2* \
  bartlesville_hec_lime_v3* \
  bartlesville_hec_lime_v4* \
  bartlesville_hec_lime_v5_interfaces_voronoi_clean_UNVALIDATED.uge

# Old root-level EX files. Active flow boundary files are protected under
# boundary_ex_v5/.
add_matches \
  AVN2.ex AVN87.ex AVN31.ex \
  bartlesville_sand.ex basal_layer.ex boreholes.ex bottom.ex east.ex \
  hec.ex injection_borehole.ex north.ex overburden.ex refined_targets.ex \
  sensor_pods.ex south.ex strainmeter_boreholes.ex strainmeter_sensors.ex \
  top.ex underburden.ex west.ex \
  boreholes.vset refined_targets.vset sensor_pods.vset strainmeter_boreholes.vset

# Raw validation/smoke outputs and derived visualization products.
add_matches \
  north_avant_v5_flow_smoke.h5 \
  north_avant_v5_flow_smoke.log \
  north_avant_v5_flow_smoke.out \
  north_avant_v5_flow_smoke_v2.log \
  north_avant_v5_oneway_injection_smoke-* \
  north_avant_v5_oneway_injection_smoke.h5 \
  north_avant_v5_oneway_injection_smoke.log \
  north_avant_v5_oneway_injection_smoke.out \
  north_avant_v5_oneway_zero_injection_smoke-* \
  north_avant_v5_oneway_zero_injection_smoke.h5 \
  north_avant_v5_oneway_zero_injection_smoke.log \
  north_avant_v5_oneway_zero_injection_smoke.out \
  north_avant_v5_twoway_injection_smoke-* \
  north_avant_v5_twoway_injection_smoke.h5 \
  north_avant_v5_twoway_injection_smoke.log \
  north_avant_v5_twoway_injection_smoke.out \
  north_avant_v5_twoway_restart_stage1-* \
  north_avant_v5_twoway_restart_stage1.h5 \
  north_avant_v5_twoway_restart_stage1.log \
  north_avant_v5_twoway_restart_stage1.out \
  north_avant_v5_twoway_restart_stage2-* \
  north_avant_v5_twoway_restart_stage2.h5 \
  north_avant_v5_twoway_restart_stage2.log \
  north_avant_v5_twoway_restart_stage2.out \
  paraview_oneway_injection \
  paraview_oneway_injection_latest \
  paraview_twoway_injection \
  paraview_v5_mesh \
  region_timeseries_v5 \
  region_timeseries_v5_curves_only \
  region_timeseries_v5_twoway \
  compare_oneway_twoway_v5 \
  compare_oneway_twoway_v5.log \
  quality_limestone_only \
  quality_limestone_v2 \
  quality_limestone_v3_localized \
  quality_limestone_v3_noq \
  quality_limestone_v4_halos \
  quality_v3_localized_v2 \
  quality_v4_localized_v2 \
  quality_v5_interfaces \
  xmf_h5_audit.log \
  xmf_h5_audit_v2.log \
  voronoi_mesh.pvtp \
  voronoi_mesh_proc*.vtp \
  __pycache__

# Legacy examples and old simulation artifacts.
add_matches \
  geomech_inj_rec.h5 geomech_inj_rec.out \
  meshtags.h5 meshtags.xmf \
  test.h5 test.out \
  preload_state preload_state.in \
  two_way two_way.in

# Optional: archive clearly superseded code only after the canonical files have
# been copied into decks/, slurm/, scripts/, and docs/ and the release audit
# passes. Git history still preserves tracked versions.
if (( INCLUDE_LEGACY_CODE == 1 )); then
  add_matches \
    poroelastic.sh \
    run_north_avant_v5_simulations.slurm \
    postprocess_north_avant_v5_results.sh \
    audit_xmf_h5.py \
    pflotran_geomech_to_vtu.py \
    pflotran_strainmeter_timeseries.py \
    workflow.py \
    workflow_mine.py \
    build_poly_layers4_mine.py \
    layers4_get_material_boundary_tags_mine.py \
    convert_vset_to_ex.py \
    generate_ugi.py \
    mapping.py \
    px.py \
    h5_outputs.py \
    input_validation.py \
    xdmf_outputs.py \
    delete_node_row.py \
    delete_ugi_row.py \
    swept_mesh.py
fi

# Optional: V5 build intermediates are valuable until the automated mesh
# workflow has been finalized. Archive them only on explicit request.
if (( INCLUDE_V5_BUILD == 1 )); then
  add_matches \
    bartlesville_hec_lime_v5_interfaces.1.edge \
    bartlesville_hec_lime_v5_interfaces.1.ele \
    bartlesville_hec_lime_v5_interfaces.1.face \
    bartlesville_hec_lime_v5_interfaces.1.neigh \
    bartlesville_hec_lime_v5_interfaces.1.node \
    bartlesville_hec_lime_v5_interfaces.inp \
    bartlesville_hec_lime_v5_interfaces.poly \
    bartlesville_hec_lime_v5_interfaces.trn \
    bartlesville_hec_lime_v5_interfaces.uge \
    bartlesville_hec_lime_v5_interfaces_borehole_tag_report.csv \
    bartlesville_hec_lime_v5_interfaces_boundaries.txt \
    bartlesville_hec_lime_v5_interfaces_canonical_export.txt \
    bartlesville_hec_lime_v5_interfaces_geometry.json \
    bartlesville_hec_lime_v5_interfaces_hec_interface_refinement.csv \
    bartlesville_hec_lime_v5_interfaces_hec_local_vertical_refinement.csv \
    bartlesville_hec_lime_v5_interfaces_hec_tag_geometry.xyz \
    bartlesville_hec_lime_v5_interfaces_hec_tag_report.csv \
    bartlesville_hec_lime_v5_interfaces_hec_tagged_nodes.xyz \
    bartlesville_hec_lime_v5_interfaces_hec_topview.csv \
    bartlesville_hec_lime_v5_interfaces_material_assignment_summary.csv \
    bartlesville_hec_lime_v5_interfaces_material_flags_summary.txt \
    bartlesville_hec_lime_v5_interfaces_material_ids_validation.txt \
    bartlesville_hec_lime_v5_interfaces_materials.txt \
    bartlesville_hec_lime_v5_interfaces_median_all.vset \
    bartlesville_hec_lime_v5_interfaces_median_uge_validation.json \
    bartlesville_hec_lime_v5_interfaces_median_uge_validation.txt \
    bartlesville_hec_lime_v5_interfaces_median_voronoi.log \
    bartlesville_hec_lime_v5_interfaces_refinement_target_tag_report.csv \
    bartlesville_hec_lime_v5_interfaces_refinement_targets.csv \
    bartlesville_hec_lime_v5_interfaces_strainmeters.csv \
    bartlesville_hec_lime_v5_interfaces_tube_refinement_profile.csv \
    bartlesville_hec_lime_v5_interfaces_vertical_grading.csv \
    bartlesville_hec_lime_v5_interfaces_voronoi.log
fi

# Deduplicate while preserving order.
UNIQUE=()
declare -A SEEN=()
for path in "${CANDIDATES[@]}"; do
  [[ -n "${SEEN[$path]:-}" ]] && continue
  SEEN[$path]=1
  UNIQUE+=("$path")
done

echo "Repository: $ROOT"
echo "Archive:    $ARCHIVE_ROOT"
echo "Mode:       $([[ $APPLY == 1 ]] && echo APPLY || echo DRY-RUN)"
echo
echo "Candidate paths: ${#UNIQUE[@]}"

if (( ${#UNIQUE[@]} == 0 )); then
  echo "Nothing to archive."
  exit 0
fi

printf '%s\n' "${UNIQUE[@]}" | sort

echo
du -sch "${UNIQUE[@]}" 2>/dev/null | tail -n 1 || true

if (( APPLY == 0 )); then
  echo
  echo "Dry-run only. Review the list, then rerun with --apply."
  exit 0
fi

mkdir -p "$ARCHIVE_ROOT"

for path in "${UNIQUE[@]}"; do
  destination="$ARCHIVE_ROOT/$path"
  mkdir -p "$(dirname "$destination")"
  mv -- "$path" "$destination"
done

(
  cd "$ARCHIVE_ROOT"
  find . -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)

printf '%s\n' "${UNIQUE[@]}" > "$ARCHIVE_ROOT/ARCHIVED_PATHS.txt"

echo
echo "Archived ${#UNIQUE[@]} paths to:"
echo "  $ARCHIVE_ROOT"
echo "Checksums:"
echo "  $ARCHIVE_ROOT/SHA256SUMS"
