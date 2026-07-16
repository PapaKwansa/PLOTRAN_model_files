# PFLOTRAN Hydro-Mechanical Bartlesville / North Avant Test Case

## Overview

This repository contains a PFLOTRAN hydro-mechanical model developed for permeability inversion using strainmeter observations from the Bartlesville Formation.

The model represents a four-layer geological system with an embedded high-permeability HEC (Hydraulically Enhanced Connectivity) lens. Injection occurs within the HEC while three strainmeters record the resulting deformation.

Current objectives include:

- Hydro-mechanical forward modeling
- Permeability inversion
- Strain prediction
- Validation of the PFLOTRAN hydro-mechanical workflow

---

# Model Geometry

Domain

- X: 0–10,000 m
- Y: 0–10,000 m
- Z: 0 to -750 m (elevation convention)

The mesh uses the following convention:

- Top surface: z = 0 m
- Bottom boundary: z = -750 m
- Gravity:

```
GRAVITY
0.d0 0.d0 -9.81d0
```

---

# Geological Layers

| Layer | Depth (m) |
|--------|-----------|
| Overburden | 0 to -500 |
| Basal Layer | -500 to -530 |
| Bartlesville Sand | -530 to -550 |
| Underburden | -550 to -750 |

---

# HEC

The HEC is represented as a porous high-permeability lens.

Properties

- Length: 580 m
- Width: 300 m
- Thickness: 5 m
- Orientation: 5° east of north
- Center:

```
(5000, 5000, -220)
```

---

# Injection

Injection occurs through the region

```
injection_borehole.vset
```

located inside the HEC.

The borehole material uses the same hydraulic properties as the HEC to avoid introducing an artificial permeability contrast.

---

# Strainmeters

| Sensor | Coordinates (m) |
|---------|-----------------|
| AVN2 | (5160, 5185, -720) |
| AVN87 | (5460, 5185, -720) |
| AVN31 | (5350, 4720, -230) |

---

# Mesh

The mesh was generated using TetGen and converted into PFLOTRAN Voronoi format.

The current version includes:

- graded refinement
- refined HEC
- refined injection region
- refined strainmeter regions
- smooth transition zones
- improved minimum cell volume

---

# Simulation

Current PFLOTRAN version

```
Development Version
```

Simulation type

```
SUBSURFACE
```

Flow

```
Richards
```

Geomechanics

```
Linear poroelasticity
```

---

# Current Status

## One-way coupling

✓ Stable

## Two-way coupling

Currently unstable.

The coupled simulation diverges during the second coupled timestep even with zero injection.

This suggests the instability originates from the hydro-mechanical feedback rather than the injection source term.

---

# Repository Contents

Main files

```
geomech_inj_rec.in
bartlesville_hec.uge
bartlesville_hec.ugi
bartlesville_hec.mapping
bartlesville_hec_material_ids.h5
```

Supporting files

```
*.vset
*.ex
```

Mesh metadata

```
bartlesville_hec_geometry.json
bartlesville_hec_vertical_grading.csv
bartlesville_hec_refinement_targets.csv
```

---

# Running

Example

```bash
mpiexec -n 5 \
/path/to/pflotran \
-input_prefix geomech_inj_rec
```

---

# Current Research Goal

The primary objective of this repository is the development of a hydro-mechanical permeability inversion workflow using pressure-induced strain observations.

At the current stage, the validated workflow is based on one-way hydro-mechanical coupling while the two-way implementation is being investigated.

---

# Contact

Henry Arhin

Clemson University
