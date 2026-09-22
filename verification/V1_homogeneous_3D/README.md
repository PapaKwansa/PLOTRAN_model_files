# V1 Homogeneous 3-D COMSOL-PFLOTRAN Verification Benchmark

## Purpose

V1 is the first controlled cross-code verification benchmark for the
North Avant coupled flow-geomechanics model.

The purpose is to compare an independently constructed COMSOL model
against an independently constructed PFLOTRAN model for the same
simplified physical problem before introducing geological layering,
material heterogeneity, the HEC, and sensor-specific regions.

V1 is a verification benchmark, not a calibration exercise.

PFLOTRAN parameters must be implemented from the benchmark specification.
They must not be adjusted solely to force agreement with COMSOL.

## V1 Physical Model

The benchmark consists of a homogeneous three-dimensional porous medium
with a finite cylindrical injection region.

### Domain

X: 0 to 10,000 m
Y: 0 to 10,000 m
Z: -1,535 to 0 m

### Hydraulic properties

Porosity:

    phi = 0.10

Anisotropic permeability:

    Kx = 4.9346165e-15 m^2
    Ky = 4.9346165e-15 m^2
    Kz = 4.9346165e-17 m^2

Horizontal-to-vertical permeability ratio:

    Kx/Kz = Ky/Kz = 100

### Mechanical properties

Young's modulus:

    E = 8.0e9 Pa

Poisson's ratio:

    nu = 0.25

Biot coefficient:

    alpha_B = 0.9

Rock density:

    rho_r = 2500 kg/m^3

Mechanical gravity:

    zero

### Fluid properties

Reference density:

    rho_ref = 1000 kg/m^3

Reference pressure:

    p_atm = 101325 Pa

Dynamic viscosity:

    mu_f = 1.0e-3 Pa*s

Fluid compressibility:

    cf = 4.4e-10 1/Pa

COMSOL V1 density law:

    rho(p) = rho_ref * exp(cf * (p - p_atm))

Hydraulic gravity:

    g = 9.81 m/s^2 in the negative Z direction

## Initial and Boundary Conditions

Initial hydrostatic pressure:

    p0(z) = p_atm - rho_ref * g * z

The four vertical outer boundaries use the hydrostatic pressure
condition.

Top and bottom hydraulic boundaries are no-flow.

Mechanical boundary conditions:

    West:   ux = 0
    East:   ux = 0
    North:  uy = 0
    South:  uy = 0
    Bottom: uz = 0
    Top:    traction free

No mechanical gravity is applied.

## Injection Geometry

The injection region is a finite vertical cylinder:

    center x = 5000 m
    center y = 5000 m
    center z = -529.75 m

    radius = 0.75 m
    height = 4.5 m

Cylinder volume:

    Vinj = pi * rw^2 * hinj

The source is distributed over the finite injection volume.

## Injection Schedule

The total volumetric injection rate is defined by a linear
piecewise schedule.

    0.000 h     0
    0.450 h     0
    0.475 h     1.464466094e-4 m^3/s
    0.500 h     5.000000000e-4 m^3/s
    0.525 h     8.535533906e-4 m^3/s
    0.550 h     1.000000000e-3 m^3/s

    2.900 h     1.000000000e-3 m^3/s
    2.950 h     1.131801948e-3 m^3/s
    3.000 h     1.450000000e-3 m^3/s
    3.050 h     1.768198052e-3 m^3/s
    3.100 h     1.900000000e-3 m^3/s

    18.950 h    1.900000000e-3 m^3/s
    18.975 h    1.621751442e-3 m^3/s
    19.000 h    9.500000000e-4 m^3/s
    19.025 h    2.782485579e-4 m^3/s
    19.050 h    0

    96.000 h    0

The maximum rate is approximately 30.12 gpm.

## Study Sequence

### V1-A: Hydrostatic Equilibrium

A stationary solution is first computed with injection disabled.

The stationary solution defines the pre-injection equilibrium state.

A zero-source transient test is used to verify that pressure, strain,
and Darcy velocity remain effectively unchanged after initialization.

### V1-B: Injection and Recovery

A 96-hour transient solution is computed using the stationary solution
as the initial state.

The injection source is enabled only during the transient step.

The physical sequence is:

    initial equilibrium
        ->
    injection
        ->
    shut-in
        ->
    pressure and deformation recovery

## Incremental Quantities

Injection-induced pressure change is defined as:

    Delta p = p_transient - p_equilibrium

Injection-induced strain change is defined as:

    Delta epsilon_ij =
        epsilon_ij,transient - epsilon_ij,equilibrium

These incremental quantities are the primary cross-code comparison
variables.

## Primary Verification Quantities

The COMSOL and PFLOTRAN implementations will be compared using:

1. Injector pressure perturbation

       Delta p_inj(t)

2. Injector strain tensor perturbation

       Delta epsilon_ij,injection(t)

3. Radial pressure perturbation

       Delta p(r,t)

## COMSOL V1 Preliminary Reference Results

The current COMSOL V1 model provides preliminary reference results.

At t = 18.94 h along the horizontal radial line through the injector:

    Peak Delta p = approximately 31.8745 MPa

Preliminary pressure attenuation distances:

    r50 = 4.82 m
    r10 = 32.60 m
    r1  = 94.64 m

These values are reference results from the current COMSOL mesh and
have not yet been demonstrated to be mesh-converged continuum values.

They are comparison benchmarks, not PFLOTRAN calibration targets.

## Verification Sequence

The planned verification hierarchy is:

    V1  homogeneous 3-D benchmark
    V2  geological layering
    V3  material heterogeneity
    V4  HEC
    V5  sensor-specific regions
    V6  full North Avant configuration

A major physical complexity is introduced only after the preceding
stage has been compared between COMSOL and PFLOTRAN.

## Reproducibility Requirements

Each V1 PFLOTRAN run should record:

- Git commit SHA
- input deck
- mesh-generation configuration
- generated mesh identifiers
- PFLOTRAN/PETSc environment
- runtime information
- postprocessed comparison tables

Large generated meshes and runtime files are not normally stored in Git.
Mesh-generation source and benchmark configuration are the authoritative
reproducibility artifacts.

## Model Separation

The production North Avant model is maintained separately on the
north-avant-v5-production branch.

This verification directory must not modify the production model.

The verification branch is:

    verification/comsol-pflotran

The current V1 work is contained in:

    verification/V1_homogeneous_3D/
