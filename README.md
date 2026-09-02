# North Avant V5 PFLOTRAN production workflow

This branch contains the validated North Avant V5 two-way-coupled PFLOTRAN
forward-model pipeline.

## Production entry points

Preproduction and scaling gate:

    ./submit_north_avant_v5_pipeline.sh preproduction

Full 96-hour simulation:

    ./submit_north_avant_v5_pipeline.sh production

The production run must not be submitted until the four-hour Palmetto
preproduction test, HDF5 validation, automatic VTU/PVD conversion, regional
time-series plotting, and resource-scaling checks pass.

## Authoritative runtime mesh

- Flow: `bartlesville_hec_lime_v5_interfaces_median.uge`
- Mechanics: `bartlesville_hec_lime_v5_interfaces.ugi`
- Mapping: `bartlesville_hec_lime_v5_interfaces_median.mapping`
- Materials: `bartlesville_hec_lime_v5_interfaces_material_ids.h5`

See:

- `docs/README_PIPELINE.md`
- `docs/NORTH_AVANT_V5_PRODUCTION_PHYSICS.md`
- `docs/MESH_BUILD_PROVENANCE.md`

The large authoritative runtime files are tracked with Git LFS. Raw PFLOTRAN
results, checkpoints, VTU/PVD files, and local Python environments are not
committed.
