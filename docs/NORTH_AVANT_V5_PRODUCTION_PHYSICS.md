# North Avant V5 two-way-coupled production model

## Status

This is the production candidate built from the V5 mesh and numerical settings
that passed the local flow-only, one-way-coupled, two-way-coupled, and
checkpoint/restart-equivalence tests. The 96-hour deck must still pass the
4-hour Palmetto preproduction/scaling run before it is treated as the production
release.

## Numerical grids and coupling

- **Flow grid:** `bartlesville_hec_lime_v5_interfaces_median.uge`
- **Geomechanics grid:** `bartlesville_hec_lime_v5_interfaces.ugi`
- **Flow-to-mechanics mapping:**
  `bartlesville_hec_lime_v5_interfaces_median.mapping`
- **Flow material IDs:**
  `bartlesville_hec_lime_v5_interfaces_material_ids.h5`
- **External flow boundaries:** `boundary_ex_v5/*.ex`

The geomechanics grid contains 140,456 vertices and 802,245 tetrahedra. The
median-dual flow grid contains one control volume per mechanics vertex, so the
validated mapping is one-to-one and identity ordered.

## Flow physics

The flow process uses PFLOTRAN `RICHARDS` mode: single-water-phase, isothermal,
variably saturated flow. The initial and lateral-boundary pressures are
hydrostatic and positive throughout this deep model, so the modeled state is
expected to remain fully saturated. Richards mode is retained because it is the
mode used in all successful V5 validation runs; changing to another saturated
flow formulation would be a separate model change.

### Initial pressure and far-field pressure

The model top is at `z = 0 m`; depth corresponds to negative elevation. Both the
initial condition and the four lateral boundaries use

- pressure at the top datum: 101,325 Pa;
- vertical pressure gradient: -9,810 Pa/m.

Because `z` is negative below the surface, this produces increasing pressure
with depth. The top and bottom flow boundaries are no-flow. The lateral
boundaries maintain the regional hydrostatic pressure field.

### Injection history

The production deck uses explicit `INTERPOLATION STEP`, preserving the behavior
of the old 96-hour deck, whose omitted interpolation defaulted to STEP:

| Time interval | Total source rate |
|---|---:|
| 0.0–0.5 h | 0.0 m3/s |
| 0.5–3.0 h | 1.0e-3 m3/s |
| 3.0–19.0 h | 1.9e-3 m3/s |
| 19.0–96.0 h | 0.0 m3/s |

`SYNC_TIMESTEP_WITH_UPDATE` forces a time-step boundary at each schedule change.
If the four tabulated values are instead continuous measured waypoints, the
correct interpretation would be `INTERPOLATION LINEAR`; that is a physical
choice that must be confirmed from the field injection record.

The source is applied through `injection_borehole.vset` using
`SCALED_VOLUMETRIC_RATE VOLUME`, so the listed value is distributed over the
selected source cells in proportion to their control volumes.

## Hydraulic materials

The initial porosity is 0.1 for all materials. The principal permeability values
are:

| ID | Region | kx = ky (m2) | kz (m2) |
|---:|---|---:|---:|
| 1 | Overburden | 9.869233e-18 | 9.869233e-19 |
| 2 | Bartlesville sandstone | 4.9346165e-15 | 4.9346165e-17 |
| 3 | Basal layer | 9.869233e-18 | 9.869233e-19 |
| 4 | Underburden | 9.869233e-18 | 9.869233e-19 |
| 5 | HEC | 4.9346165e-13 | 9.869233e-17 |
| 6 | Injection region | 4.9346165e-13 | 9.869233e-17 |
| 7 | AVN2 pod | 9.869233e-18 | 9.869233e-19 |
| 8 | AVN87 pod | 9.869233e-18 | 9.869233e-19 |
| 9 | AVN31 pod | 4.9346165e-15 | 4.9346165e-17 |
| 10 | Shallow limestone | 9.869233e-18 | 9.869233e-19 |

The sensor pod properties match their host lithology, so the pods act as refined
observation regions rather than artificial hydraulic contrasts.

## Geomechanics physics

The solid is modeled as isotropic, linear elastic poroelastic material. Pressure
from flow changes effective stress through the Biot coefficient. The primary
mechanics unknowns are the three displacement components at each mechanics
vertex; strain and stress are derived fields.

### Mechanical properties

| ID | Region | Density (kg/m3) | Young's modulus (Pa) | Poisson ratio | Biot coefficient |
|---:|---|---:|---:|---:|---:|
| 1 | Overburden | 2700 | 2.0e10 | 0.28 | 0.7 |
| 2 | Bartlesville sandstone | 2500 | 9.0e9 | 0.25 | 0.9 |
| 3 | Basal layer | 2500 | 2.5e10 | 0.28 | 0.7 |
| 4 | Underburden | 2700 | 2.0e10 | 0.28 | 0.7 |
| 5 | HEC | 2300 | 9.0e9 | 0.25 | 0.9 |
| 6 | Injection region | 2300 | 9.0e9 | 0.25 | 0.9 |
| 7 | AVN2 pod | 2700 | 4.2e10 | 0.28 | 0.7 |
| 8 | AVN87 pod | 2700 | 4.2e10 | 0.28 | 0.7 |
| 9 | AVN31 pod | 2500 | 9.0e9 | 0.25 | 0.9 |
| 10 | Shallow limestone | 2700 | 4.2e10 | 0.28 | 0.7 |

AVN2 and AVN87 are embedded in the explicit shallow-limestone layer and use the
same limestone properties. AVN31 uses the Bartlesville sandstone properties.

### Gravity interpretation

The geomechanics grid uses `GRAVITY 0 0 0`. This is deliberate: the run computes
pressure-induced incremental displacement, strain, and stress relative to the
initial hydrostatic state. It does not calculate lithostatic self-weight or
initial geostatic deformation. Adding gravity or an initial-stress preload is a
separate physical model and must be validated independently before replacing
this production candidate.

### Mechanical boundary conditions

- west and east: `ux = 0` roller condition;
- north and south: `uy = 0` roller condition;
- bottom: `uz = 0`;
- top: traction-free.

These constraints prevent rigid-body motion while allowing vertical surface
response and tangential motion along the lateral boundaries.

## Two-way coupling

`FLOW_COUPLING TWO_WAY_COUPLED` transfers pressure from flow to geomechanics and
returns deformation-dependent porosity and permeability to the flow model. The
short validation run showed that this feedback was active, stable, and small at
early time. It can grow during the full 19-hour injection and recovery history,
which is why the production output includes pressure, porosity, and all three
principal permeability components.

## Time stepping

The production deck retains the pressure-change governor that passed the V5
smoke tests and uses a scheduled maximum time step:

- fine early-time resolution;
- gradual growth during injection;
- renewed refinement before and immediately after the 19-hour shut-in;
- larger steps during late recovery.

The deck synchronizes with source changes, requested outputs, and checkpoint
times. Any persistent timestep collapse or repeated cuts in the 4-hour Palmetto
gate must be resolved before the 96-hour run.

## Checkpoint/restart

The production deck writes HDF5 checkpoints periodically and at selected times,
including 18.95 h immediately before shut-in. The restart workflow has already
reproduced the continuous short-run solution within field-specific numerical
tolerances. Checkpoints are numerical safety states; they do not break or reset
the physical timeline.

## Output and visualization

Flow HDF5 output includes:

- liquid pressure;
- material ID;
- porosity;
- permeability X, Y, and Z.

Geomechanics HDF5 output includes displacement, relative displacement, strain,
stress, total stress, and material ID at the same requested times.

The dependent postprocessing job creates:

- coupled VTU snapshots containing both mapped flow fields and mechanics fields;
- a PVD time-series collection for ParaView;
- a true three-component `Displacement` vector for `Warp By Vector`;
- pressure change relative to the first flow output;
- HEC, injection, AVN2, AVN87, and AVN31 flags;
- region-averaged CSV tables;
- PNG and PDF strain/displacement time-series plots;
- compact region PVD/VTU files;
- manifests and SHA-256 checksums.

## Remaining release gate

The model is numerically validated locally, but the complete production deck is
not yet released until the 4-hour Palmetto preproduction run verifies:

1. deck parsing and input record;
2. both step changes at 0.5 h and 3 h;
3. two-way solver stability on Palmetto;
4. HDF5 checkpoints and result files;
5. porosity/permeability output variable support in the installed development
   build;
6. automatic VTU/PVD and time-series postprocessing;
7. memory use and MPI scaling.
