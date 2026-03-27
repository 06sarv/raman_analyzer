#!/usr/bin/env python3
"""
tools/validate_database.py
--------------------------
Validate a Raman compound database JSON file against the expected schema.

Usage:
    python tools/validate_database.py data/rruff_database.json
    python tools/validate_database.py data/sample_database.json data/rruff_database.json
"""

import json
import sys
import os


def validate_database(path: str) -> tuple[list[str], list[str]]:
    """Validate a single database JSON file. Returns (errors, warnings)."""
    errors = []
    warnings = []

    if not os.path.exists(path):
        return [f"File not found: {path}"], []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {path}: {e}"], []

    if not isinstance(data, dict):
        return [f"Top-level must be a dict (category -> compounds), got {type(data).__name__}"], []

    total_compounds = 0
    total_peaks = 0

    for category, compounds in data.items():
        if not isinstance(category, str):
            errors.append(f"Category key must be string, got {type(category).__name__}")
            continue

        if not isinstance(compounds, list):
            errors.append(f"Category '{category}' must map to a list, got {type(compounds).__name__}")
            continue

        if len(compounds) == 0:
            warnings.append(f"Category '{category}' is empty")

        for i, compound in enumerate(compounds):
            loc = f"'{category}'[{i}]"

            if not isinstance(compound, dict):
                errors.append(f"{loc}: compound must be a dict")
                continue

            # Name
            name = compound.get("Name")
            if not name or not isinstance(name, str):
                errors.append(f"{loc}: missing or invalid 'Name'")
                continue
            loc = f"'{category}' / '{name}'"

            # Peaks
            peaks = compound.get("Peaks")
            if not isinstance(peaks, list):
                errors.append(f"{loc}: missing or invalid 'Peaks' (must be list)")
                continue

            if len(peaks) == 0:
                errors.append(f"{loc}: has zero peaks")
                continue

            for j, peak in enumerate(peaks):
                ploc = f"{loc} peak[{j}]"

                if not isinstance(peak, dict):
                    errors.append(f"{ploc}: peak must be a dict")
                    continue

                # Wavenumber
                wn = peak.get("Wavenumber")
                if wn is None or not isinstance(wn, (int, float)):
                    errors.append(f"{ploc}: missing or invalid 'Wavenumber'")
                elif wn < 50 or wn > 4500:
                    warnings.append(f"{ploc}: Wavenumber {wn} outside typical range (50-4500 cm-1)")

                # Assignment
                assignment = peak.get("Assignment")
                if not assignment or not isinstance(assignment, str):
                    errors.append(f"{ploc}: missing or invalid 'Assignment'")

                # RelativeIntensity
                ri = peak.get("RelativeIntensity")
                if ri is None or not isinstance(ri, (int, float)):
                    errors.append(f"{ploc}: missing or invalid 'RelativeIntensity'")
                elif ri < 0 or ri > 1.0:
                    warnings.append(f"{ploc}: RelativeIntensity {ri} outside 0-1 range")

                total_peaks += 1

            total_compounds += 1

    return errors, warnings, total_compounds, total_peaks, len(data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_database.py <file1.json> [file2.json ...]")
        sys.exit(1)

    all_ok = True

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Validating: {path}")
        print(f"{'='*60}")

        result = validate_database(path)
        if len(result) == 2:
            errors, warnings = result
            n_compounds = n_peaks = n_categories = 0
        else:
            errors, warnings, n_compounds, n_peaks, n_categories = result

        if not errors:
            print(f"\n  PASS - {n_compounds} compounds, {n_peaks} peaks, {n_categories} categories")
        else:
            print(f"\n  FAIL - {len(errors)} error(s)")
            all_ok = False

        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")

    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
