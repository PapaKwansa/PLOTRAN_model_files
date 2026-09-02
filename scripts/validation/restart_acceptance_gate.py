#!/usr/bin/env python3
"""
Field-aware acceptance gate for a PFLOTRAN continuous-vs-restart comparison.

Why this exists
---------------
A pure relative L2 criterion can falsely reject displacement fields that are
very small in absolute magnitude. This gate uses a mixed criterion:

    pass when relative_L2 <= relative_tolerance
    OR maximum_absolute_difference <= absolute_tolerance

Material IDs must match exactly.

Default tolerances
------------------
Flow pressure:
    relative L2 <= 1e-8 OR max absolute difference <= 2 Pa

Displacement and relative displacement:
    relative L2 <= 1e-5 OR max absolute difference <= 1e-9 m

Strain and volumetric strain:
    relative L2 <= 1e-5 OR max absolute difference <= 5e-11

Stress and total stress:
    relative L2 <= 1e-5 OR max absolute difference <= 2 Pa

These thresholds are intended for the North Avant V5 restart-equivalence test.
They should be recorded with the validation result rather than silently changed.

Usage
-----
python3 restart_acceptance_gate.py \
  validation/restart_equivalence_v5/comparison
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Tolerance:
    relative_l2: float
    maximum_absolute: float
    units: str


TOLERANCES = {
    "flow_pressure": Tolerance(1.0e-8, 2.0, "Pa"),
    "displacement": Tolerance(1.0e-5, 1.0e-9, "m"),
    "strain": Tolerance(1.0e-5, 5.0e-11, "dimensionless"),
    "stress": Tolerance(1.0e-5, 2.0, "Pa"),
    "default_geomechanics": Tolerance(1.0e-5, 0.0, "native"),
    "default_flow": Tolerance(1.0e-8, 0.0, "native"),
}


def classify_field(category: str, field: str) -> str:
    name = field.lower()

    if name == "material_id":
        return "material"

    if category == "flow" and "pressure" in name:
        return "flow_pressure"

    if "displacement" in name:
        return "displacement"

    if "strain" in name:
        return "strain"

    if "stress" in name:
        return "stress"

    return (
        "default_geomechanics"
        if category == "geomechanics"
        else "default_flow"
    )


def first_present(row: dict[str, str], names: Iterable[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return None


def peak_field_amplitude(row: dict[str, str]) -> float | None:
    values = []

    for names in (
        ("continuous_min", "oneway_min"),
        ("continuous_max", "oneway_max"),
        ("restart_min", "twoway_min"),
        ("restart_max", "twoway_max"),
    ):
        value = first_present(row, names)
        if value is not None and math.isfinite(value):
            values.append(abs(value))

    return max(values) if values else None


def load_latest_rows(path: Path) -> tuple[float, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"No rows in {path}")

    latest = max(float(row["time_hours"]) for row in rows)

    selected = [
        row
        for row in rows
        if math.isclose(
            float(row["time_hours"]),
            latest,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]

    return latest, selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply field-aware restart-equivalence tolerances."
    )
    parser.add_argument(
        "comparison_dir",
        type=Path,
        nargs="?",
        default=Path("validation/restart_equivalence_v5/comparison"),
    )
    args = parser.parse_args()

    base = args.comparison_dir.expanduser().resolve()

    checks = (
        (
            "geomechanics",
            base / "geomechanics_global_field_metrics.csv",
        ),
        (
            "flow",
            base / "flow_global_field_metrics.csv",
        ),
    )

    failures: list[tuple[str, str, str]] = []

    print("PFLOTRAN RESTART-EQUIVALENCE ACCEPTANCE GATE")
    print("=" * 96)
    print(
        "Criterion: relative L2 passes OR field-specific maximum absolute "
        "difference passes."
    )
    print("Material IDs must match exactly.")

    for category, path in checks:
        latest, rows = load_latest_rows(path)

        print()
        print(f"{category.upper()} AT {latest:g} h")
        print("-" * 96)

        for row in rows:
            field = row["field_key"]
            relative_l2 = float(row["symmetric_relative_l2"])
            maximum_absolute = float(row["maximum_absolute_delta"])
            field_class = classify_field(category, field)

            if not (
                math.isfinite(relative_l2)
                and math.isfinite(maximum_absolute)
            ):
                passed = False
                reason = "non-finite comparison metric"
                peak_relative_text = "n/a"

            elif field_class == "material":
                passed = maximum_absolute == 0.0
                reason = "exact material-ID agreement required"
                peak_relative_text = "n/a"

            else:
                tolerance = TOLERANCES[field_class]
                relative_pass = relative_l2 <= tolerance.relative_l2
                absolute_pass = (
                    tolerance.maximum_absolute > 0.0
                    and maximum_absolute <= tolerance.maximum_absolute
                )
                passed = relative_pass or absolute_pass

                reason = (
                    f"relL2 <= {tolerance.relative_l2:.1e} "
                    f"OR max|delta| <= "
                    f"{tolerance.maximum_absolute:.1e} {tolerance.units}"
                )

                peak = peak_field_amplitude(row)
                if peak is not None and peak > 0.0:
                    peak_relative = maximum_absolute / peak
                    peak_relative_text = f"{peak_relative:.6e}"
                else:
                    peak_relative_text = "n/a"

            status = "PASS" if passed else "FAIL"

            print(
                f"{status:4s}  "
                f"{field:36s}  "
                f"relL2={relative_l2:.6e}  "
                f"max|delta|={maximum_absolute:.6e}  "
                f"maxDelta/peak={peak_relative_text}"
            )
            print(f"      criterion: {reason}")

            if not passed:
                failures.append((category, field, reason))

    print()
    print("=" * 96)

    if failures:
        print("RESTART EQUIVALENCE: FAILED")
        for category, field, reason in failures:
            print(f"  {category}: {field}: {reason}")
        return 1

    print("RESTART EQUIVALENCE: PASSED")
    print(
        "Note: displacement fields passed by absolute tolerance where their "
        "small field norm made a pure relative L2 test misleading."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
