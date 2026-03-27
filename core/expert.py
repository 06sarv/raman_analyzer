"""
core/expert.py
--------------
Rule-based expert system for interpreting Raman peaks.

Loads functional group rules from a JSON file where each entry is:
{
  "wavenumber_range_cm-1": "1600-1800",
  "vibrational_mode": "C=O stretch",
  "compound_functionality": "Carbonyl"
}

Also provides sample-condition diagnostics based on metadata.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# Rule loader
# ─────────────────────────────────────────────

def load_functional_group_rules(path: str) -> List[Dict]:
    """Load functional group rules from a JSON file. Returns [] on failure."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _parse_range(range_str: str) -> Tuple[float, float]:
    """Parse '1600-1800' or '1600–1800' into (1600.0, 1800.0)."""
    for sep in ["–", "-", "—"]:
        if sep in range_str:
            parts = range_str.split(sep, 1)
            return float(parts[0].strip()), float(parts[1].strip())
    val = float(range_str.strip())
    return val, val


# ─────────────────────────────────────────────
# Expert interpreter
# ─────────────────────────────────────────────

class ExpertInterpreter:
    def __init__(self, rules: List[Dict]):
        self._rules = rules
        # Pre-parse all ranges for speed
        self._parsed: List[Tuple[float, float, str, str]] = []
        for rule in rules:
            range_str = rule.get("wavenumber_range_cm-1", "")
            try:
                lo, hi = _parse_range(range_str)
                mode = rule.get("vibrational_mode", "")
                func = rule.get("compound_functionality", "")
                self._parsed.append((lo, hi, mode, func))
            except ValueError:
                continue

    def assign_functional_groups(
        self, peaks: List[float]
    ) -> List[Tuple[str, float]]:
        """
        For each peak, find ALL matching functional group rules.
        Returns list of (label, wavenumber) without duplicates.
        """
        results: List[Tuple[str, float]] = []
        seen = set()
        for peak in peaks:
            for lo, hi, mode, func in self._parsed:
                if lo <= peak <= hi:
                    label = f"{mode} ({func})"
                    key = (label, round(peak))
                    if key not in seen:
                        results.append((label, peak))
                        seen.add(key)
        return sorted(results, key=lambda x: x[1])

    def get_diagnostics(self, peaks: List[float], metadata: Dict) -> List[str]:
        """Return list of diagnostic messages based on peaks and sample metadata."""
        diagnostics: List[str] = []

        # Excitation-based warnings
        excitation = metadata.get("excitation", "")
        if excitation in ("UV", "Visible"):
            diagnostics.append(
                "⚠️ UV/Visible excitation may induce significant fluorescence — "
                "consider switching to NIR."
            )
        elif excitation == "NIR":
            diagnostics.append(
                "✅ NIR excitation is well-suited for biological/organic samples "
                "(reduced fluorescence)."
            )

        # Amorphous vs crystalline
        if metadata.get("crystalline") == "No":
            diagnostics.append(
                "🔍 Amorphous/disordered structure expected — peaks may appear broad."
            )

        # Liquid sample
        if metadata.get("sample_state") == "Liquid":
            diagnostics.append(
                "💧 Liquid sample: watch for broad solvent peaks overlapping analyte bands."
            )

        # Low-wavenumber mineral/inorganic indicator
        if any(p < 500 for p in peaks):
            diagnostics.append(
                "🪨 Peaks below 500 cm⁻¹ suggest minerals, inorganic salts, or "
                "metal-ligand vibrations."
            )

        # Peak crowding
        sorted_peaks = sorted(peaks)
        for i in range(len(sorted_peaks) - 1):
            if abs(sorted_peaks[i] - sorted_peaks[i + 1]) < 15:
                diagnostics.append(
                    "🔍 Closely spaced peaks detected — consider deconvolution "
                    "for improved resolution."
                )
                break

        # High-wavenumber C–H region
        ch_peaks = [p for p in peaks if 2800 <= p <= 3100]
        if ch_peaks:
            diagnostics.append(
                f"🧪 C–H stretching region ({min(ch_peaks):.0f}–"
                f"{max(ch_peaks):.0f} cm⁻¹): organic compound likely present."
            )

        return diagnostics
