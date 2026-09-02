#!/usr/bin/env python3
"""
Audit PFLOTRAN XMF/XDMF references without altering whitespace inside HDF5 paths.

PFLOTRAN geomechanics time-group names may contain leading/padded spaces.
HDF5 object names are whitespace-sensitive, so collapsing whitespace can create
false "dataset missing" reports.

The script:
  * checks XML well-formedness;
  * preserves the exact HDF5 object path written in each DataItem;
  * checks referenced files, datasets, and dimensions;
  * reports a unique whitespace-normalized candidate when an exact path fails;
  * can write a repaired XMF that uses the exact matching HDF5 path.

Usage:
  python3 audit_xmf_h5_v2.py result-geomech-002.xmf
  python3 audit_xmf_h5_v2.py result-geomech-002.xmf \
      --write-repaired result-geomech-002-repaired.xmf
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass
class Ref:
    element: ET.Element
    file_text: str
    h5_file: Path
    dataset_path: str
    dimensions_text: str | None


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_dimensions(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    values = re.findall(r"\d+", text)
    return tuple(int(value) for value in values) if values else None


def exact_dataitem_text(element: ET.Element) -> str:
    """
    Remove XML indentation at the outside only.

    Do not use split()/join(): spaces inside HDF5 group names are significant.
    """
    text = "".join(element.itertext())
    return text.strip("\r\n\t ")


def normalize_component(component: str) -> str:
    return " ".join(component.split())


def normalize_h5_path(path: str) -> str:
    leading = "/" if path.startswith("/") else ""
    parts = [normalize_component(part) for part in path.split("/") if part != ""]
    return leading + "/".join(parts)


def all_dataset_paths(h5: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append("/" + name)

    h5.visititems(visitor)
    return paths


def parse_refs(root: ET.Element, xmf_path: Path) -> list[Ref]:
    refs: list[Ref] = []

    for element in root.iter():
        if local_name(element.tag) != "DataItem":
            continue

        fmt = (
            element.attrib.get("Format")
            or element.attrib.get("format")
            or ""
        ).strip().upper()

        if fmt != "HDF":
            continue

        raw = exact_dataitem_text(element)

        if ":" not in raw:
            raise ValueError(
                f"HDF DataItem lacks 'file:/dataset': {raw!r}"
            )

        file_text, dataset_path = raw.split(":", 1)
        file_text = file_text.strip()

        # Strip only whitespace outside the complete path. Whitespace after the
        # leading slash and inside group/dataset names is retained.
        dataset_path = dataset_path.strip("\r\n\t ")

        if not dataset_path.startswith("/"):
            dataset_path = "/" + dataset_path

        h5_file = Path(file_text)
        if not h5_file.is_absolute():
            h5_file = (xmf_path.parent / h5_file).resolve()

        refs.append(
            Ref(
                element=element,
                file_text=file_text,
                h5_file=h5_file,
                dataset_path=dataset_path,
                dimensions_text=(
                    element.attrib.get("Dimensions")
                    or element.attrib.get("dimensions")
                ),
            )
        )

    return refs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit PFLOTRAN XMF/HDF5 references exactly."
    )
    parser.add_argument("xmf", type=Path)
    parser.add_argument(
        "--write-repaired",
        type=Path,
        help=(
            "Write a new XMF when each missing exact path has one unique "
            "whitespace-normalized HDF5 match."
        ),
    )
    args = parser.parse_args()

    xmf_path = args.xmf.expanduser().resolve()

    if not xmf_path.is_file():
        print(f"ERROR: missing XMF: {xmf_path}", file=sys.stderr)
        return 2

    print(f"Reading XMF: {xmf_path}")

    try:
        tree = ET.parse(xmf_path)
    except ET.ParseError as exc:
        print(f"ERROR: invalid XML: {exc}", file=sys.stderr)
        return 3

    root = tree.getroot()
    print("XML structure: PASSED")

    try:
        refs = parse_refs(root, xmf_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    if not refs:
        print("ERROR: no HDF DataItems found.", file=sys.stderr)
        return 5

    handles: dict[Path, h5py.File] = {}
    paths_by_file: dict[Path, list[str]] = {}
    exact_failures = 0
    hard_failures = 0
    repaired_count = 0

    try:
        for index, ref in enumerate(refs, start=1):
            print(
                f"\n[{index:02d}] "
                f"{ref.h5_file.name}:{ref.dataset_path!r}"
            )

            if not ref.h5_file.is_file():
                print(f"  ERROR: missing HDF5 file: {ref.h5_file}")
                hard_failures += 1
                continue

            if ref.h5_file not in handles:
                try:
                    handles[ref.h5_file] = h5py.File(ref.h5_file, "r")
                except OSError as exc:
                    print(f"  ERROR: cannot open HDF5: {exc}")
                    hard_failures += 1
                    continue

                paths_by_file[ref.h5_file] = all_dataset_paths(
                    handles[ref.h5_file]
                )

            h5 = handles[ref.h5_file]
            chosen_path = ref.dataset_path

            if chosen_path not in h5:
                exact_failures += 1
                target = normalize_h5_path(chosen_path)

                candidates = [
                    path
                    for path in paths_by_file[ref.h5_file]
                    if normalize_h5_path(path) == target
                ]

                if len(candidates) == 1:
                    candidate = candidates[0]
                    print("  exact path: MISSING")
                    print(
                        "  unique whitespace-normalized match:",
                        repr(candidate),
                    )
                    chosen_path = candidate

                    if args.write_repaired is not None:
                        ref.element.text = (
                            f"{ref.file_text}:{candidate}"
                        )
                        repaired_count += 1
                elif len(candidates) == 0:
                    print("  ERROR: dataset missing; no normalized match")
                    hard_failures += 1
                    continue
                else:
                    print(
                        "  ERROR: ambiguous normalized matches:",
                        [repr(value) for value in candidates],
                    )
                    hard_failures += 1
                    continue
            else:
                print("  exact path: PASSED")

            obj = h5[chosen_path]

            if not isinstance(obj, h5py.Dataset):
                print("  ERROR: referenced object is not a dataset")
                hard_failures += 1
                continue

            expected = parse_dimensions(ref.dimensions_text)
            actual = tuple(int(value) for value in obj.shape)

            print(f"  HDF5 shape={actual}, dtype={obj.dtype}")
            print(f"  XMF dimensions={expected}")

            if expected is not None and expected != actual:
                expected_ns = tuple(value for value in expected if value != 1)
                actual_ns = tuple(value for value in actual if value != 1)

                if expected_ns != actual_ns:
                    print("  ERROR: dimension mismatch")
                    hard_failures += 1
                else:
                    print(
                        "  WARNING: dimensions differ only by singleton axes"
                    )
            else:
                print("  dimensions: PASSED")
    finally:
        for handle in handles.values():
            handle.close()

    if args.write_repaired is not None and hard_failures == 0:
        output = args.write_repaired.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        tree.write(
            output,
            encoding="utf-8",
            xml_declaration=True,
        )
        print(
            f"\nWrote repaired XMF: {output} "
            f"({repaired_count} path replacements)"
        )

    print("\nSummary")
    print("-------")
    print(f"HDF references:       {len(refs)}")
    print(f"Exact-path failures:  {exact_failures}")
    print(f"Hard failures:        {hard_failures}")

    if hard_failures:
        print("XMF/HDF5 audit: FAILED")
        return 1

    if exact_failures:
        print(
            "XMF/HDF5 audit: MATCHED AFTER WHITESPACE NORMALIZATION"
        )
        if args.write_repaired is None:
            print(
                "Rerun with --write-repaired to create a corrected XMF."
            )
    else:
        print("XMF/HDF5 audit: PASSED EXACTLY")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
