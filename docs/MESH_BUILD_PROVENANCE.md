# North Avant V5 mesh-build provenance

The frozen production runtime files were generated and validated in this order:

1. `build_poly_layers4.py`
2. TetGen PLC tetrahedralization
3. `tetgen_quality_report_localized.py`
4. `layers4_get_material_boundary_tags.py --skip-px`
5. `tetgen_to_avs_ugi_canonical.py`
6. `material_h5_from_txt.py`
7. LANL VORONOI using the median-dual option
8. `validate_uge_and_write_mapping.py`
9. `build_boundary_ex.py`
10. PFLOTRAN flow-only, one-way, two-way, and checkpoint/restart tests

Authoritative runtime products:

- `bartlesville_hec_lime_v5_interfaces_median.uge`
- `bartlesville_hec_lime_v5_interfaces.ugi`
- `bartlesville_hec_lime_v5_interfaces_median.mapping`
- `bartlesville_hec_lime_v5_interfaces_material_ids.h5`
- `boundary_ex_v5/*.ex`
- the vsets listed in `north_avant_v5_runtime_manifest.txt`

The old top-level `workflow.py` was not used for the final V5 runtime package.
A final one-command mesh-regeneration workflow will be completed before the
surrogate-model stage. The present Palmetto production run uses the frozen,
validated files listed above.
