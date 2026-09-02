# North Avant V5 production and automatic postprocessing pipeline

This package contains the production-candidate deck, a 4-hour Palmetto gate,
a robust PFLOTRAN simulation job, a dependent postprocessing job, and a
one-command submission wrapper.

## Required runtime-bundle layout

Place the validated static model inputs under:

```text
/home/harhin/PLOTRAN_model_files/north_avant_v5_palmetto_bundle/
```

The bundle root must contain both decks, the runtime manifest, the canonical
UGI, median UGE, validated mapping, material HDF5, boundary EX directory, and
all manifest-listed vsets. Put the two Python postprocessors under:

```text
north_avant_v5_palmetto_bundle/scripts/postprocess/
```

The Slurm scripts and submit wrapper may remain in the repository root.

## One-time postprocessing environment

```bash
python3 -m venv "$HOME/.venvs/north_avant_postprocess"
"$HOME/.venvs/north_avant_postprocess/bin/python" -m pip install --upgrade pip
"$HOME/.venvs/north_avant_postprocess/bin/python" -m pip install \
  -r requirements-postprocess.txt
```

## Submit the mandatory 4-hour gate

```bash
chmod +x submit_north_avant_v5_pipeline.sh \
  slurm/run_north_avant_v5_simulation.slurm \
  slurm/postprocess_north_avant_v5_results.slurm

SIM_SCRIPT="$PWD/slurm/run_north_avant_v5_simulation.slurm" \
POST_SCRIPT="$PWD/slurm/postprocess_north_avant_v5_results.slurm" \
./submit_north_avant_v5_pipeline.sh preproduction
```

The wrapper submits the simulation and a dependent `afterok` postprocessing
job. The postprocessing job runs only when the simulation exits successfully.

## Submit the 96-hour production run

After the 4-hour gate and resource review pass:

```bash
SIM_SCRIPT="$PWD/slurm/run_north_avant_v5_simulation.slurm" \
POST_SCRIPT="$PWD/slurm/postprocess_north_avant_v5_results.slurm" \
./submit_north_avant_v5_pipeline.sh production
```

## Products

Raw PFLOTRAN outputs, checkpoints, validation metadata, and postprocessed
products are archived under:

```text
/home/harhin/pflotran_results/north_avant_v5/
  <SIM_JOB_ID>_<INPUT_PREFIX>/
```

Automatic products include:

```text
postprocess/paraview_coupled/*.vtu
postprocess/paraview_coupled/*.pvd
postprocess/region_timeseries/*.csv
postprocess/region_timeseries/*.vtu
postprocess/region_timeseries/*.pvd
postprocess/region_timeseries/plots/**/*.png
postprocess/region_timeseries/plots/**/*.pdf
```

The coupled VTU series contains mechanics fields, mapped flow fields, a true
`Displacement` vector, `Displacement_Magnitude`, and
`Flow_Liquid_Pressure_Change_Pa` when liquid pressure exists.
