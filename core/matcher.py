"""
core/matcher.py
---------------
Multi-strategy compound matching engine.

Strategies combined into a single confidence score (0–100):
  1. Peak overlap score  – how many reference peaks are matched
  2. Rarity bonus        – rare peaks (shared by few compounds) score higher
  3. Cosine similarity   – full spectrum vector similarity against a binned
                           reference built from the compound's known peaks
  4. Penalty             – unmatched reference peaks reduce confidence

All strategies are normalised and blended with configurable weights.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.spectrum import make_feature_vector, cosine_similarity, RANGE_START, RANGE_END, BIN_SIZE


# ─────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────

@dataclass
class MatchResult:
    compound: str
    group: str
    confidence: float          # 0–100
    matched_peaks: List[float] # observed peaks that matched
    unmatched_ref_peaks: int   # reference peaks with no match
    cosine_score: float
    peak_score: float
    pubchem_link: str = ""

    def to_dict(self) -> dict:
        return {
            "Compound": self.compound,
            "Group": self.group,
            "Confidence (%)": round(self.confidence, 1),
            "Matched Peaks": len(self.matched_peaks),
            "Unmatched Ref Peaks": self.unmatched_ref_peaks,
            "Cosine Score": round(self.cosine_score, 3),
            "PubChem Link": self.pubchem_link,
        }


# ─────────────────────────────────────────────
# Rarity index
# ─────────────────────────────────────────────

def _build_rarity_index(database: Dict) -> Dict[int, float]:
    """
    For each rounded wavenumber that appears in the DB, compute how many
    compounds share it (within ±30 cm⁻¹ buckets).
    Returns {bucket_wavenumber: count}.
    """
    bucket_size = 30
    counts: Dict[int, int] = {}
    for compounds in database.values():
        for compound in compounds:
            seen = set()
            for peak_entry in compound.get("Peaks", []):
                wn = peak_entry.get("Wavenumber")
                if wn is not None:
                    bucket = int(round(wn / bucket_size) * bucket_size)
                    if bucket not in seen:
                        counts[bucket] = counts.get(bucket, 0) + 1
                        seen.add(bucket)
    return counts


# ─────────────────────────────────────────────
# Reference spectrum builder
# ─────────────────────────────────────────────

def _compound_feature_vector(compound: dict) -> np.ndarray:
    """Build a synthetic binned feature vector from a compound's reference peaks."""
    bins = np.arange(RANGE_START, RANGE_END + BIN_SIZE, BIN_SIZE)
    vec = np.zeros(len(bins) - 1)
    for peak_entry in compound.get("Peaks", []):
        wn = peak_entry.get("Wavenumber")
        intensity = peak_entry.get("RelativeIntensity", 1.0)  # normalised 0–1
        if wn is not None:
            for i in range(len(bins) - 1):
                if bins[i] <= wn < bins[i + 1]:
                    vec[i] = max(vec[i], float(intensity))
                    break
    return vec


# ─────────────────────────────────────────────
# Main matcher
# ─────────────────────────────────────────────

class CompoundMatcher:
    """
    Match an observed spectrum against a compound database using a
    multi-strategy confidence score.
    """

    # Blend weights (must sum to 1.0)
    W_PEAK = 0.45
    W_RARITY = 0.20
    W_COSINE = 0.25
    W_PENALTY = 0.10   # subtracted

    def __init__(
        self,
        database: Dict,
        tolerance: float = 30.0,
        min_matches: int = 1,
    ):
        self.database = database
        self.tolerance = tolerance
        self.min_matches = min_matches
        self._rarity_index = _build_rarity_index(database)
        self._total_compounds = sum(len(v) for v in database.values())

    def _peak_overlap(
        self,
        observed: List[float],
        ref_peaks: List[dict],
    ) -> Tuple[List[float], int, float]:
        """
        Returns (matched_observed, unmatched_ref_count, raw_overlap_score 0-1).
        """
        matched_obs: List[float] = []
        unmatched_ref = 0
        for ref_peak in ref_peaks:
            ref_wn = ref_peak.get("Wavenumber")
            if ref_wn is None:
                continue
            hit = any(abs(obs - ref_wn) <= self.tolerance for obs in observed)
            if hit:
                # record closest observed peak
                closest = min(observed, key=lambda o: abs(o - ref_wn))
                if closest not in matched_obs:
                    matched_obs.append(closest)
            else:
                unmatched_ref += 1

        n_ref = len(ref_peaks)
        score = len(matched_obs) / n_ref if n_ref > 0 else 0.0
        return matched_obs, unmatched_ref, score

    def _rarity_score(self, matched_obs: List[float]) -> float:
        """
        Average rarity bonus for matched peaks.
        Peaks shared by fewer compounds get a higher bonus.
        """
        if not matched_obs:
            return 0.0
        bucket_size = 30
        bonuses = []
        for wn in matched_obs:
            bucket = int(round(wn / bucket_size) * bucket_size)
            count = self._rarity_index.get(bucket, 1)
            # rarity bonus: peaks shared by 1 compound → 1.0, many → near 0
            bonus = 1.0 / count if count > 0 else 0.0
            bonuses.append(bonus)
        # Normalise so max possible = 1.0
        max_bonus = 1.0 / 1.0  # single compound = 1.0
        return min(np.mean(bonuses) / max_bonus, 1.0)

    def match(
        self,
        observed_peaks: List[float],
        observed_feature_vector: np.ndarray,
    ) -> List[MatchResult]:
        """Run matching and return sorted list of MatchResult."""
        results: List[MatchResult] = []

        for group, compounds in self.database.items():
            for compound in compounds:
                ref_peaks = compound.get("Peaks", [])
                if not ref_peaks:
                    continue

                matched_obs, unmatched_ref, peak_score = self._peak_overlap(
                    observed_peaks, ref_peaks
                )

                if len(matched_obs) < self.min_matches:
                    continue

                rarity_score = self._rarity_score(matched_obs)

                # Cosine similarity against synthetic reference vector
                ref_vec = _compound_feature_vector(compound)
                cos_score = cosine_similarity(observed_feature_vector, ref_vec)

                # Penalty for unmatched reference peaks (normalised)
                penalty = (unmatched_ref / len(ref_peaks)) if ref_peaks else 0.0

                # Blend into final confidence 0–100
                confidence = (
                    self.W_PEAK * peak_score
                    + self.W_RARITY * rarity_score
                    + self.W_COSINE * cos_score
                    - self.W_PENALTY * penalty
                ) * 100
                confidence = max(0.0, min(100.0, confidence))

                results.append(MatchResult(
                    compound=compound.get("Name", "Unknown"),
                    group=group,
                    confidence=confidence,
                    matched_peaks=sorted(matched_obs),
                    unmatched_ref_peaks=unmatched_ref,
                    cosine_score=cos_score,
                    peak_score=peak_score,
                ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ─────────────────────────────────────────────
# Reference Library — full-spectrum similarity
# ─────────────────────────────────────────────

@dataclass
class SimilarityResult:
    """Result of a similarity search against the reference library."""
    label: str
    cosine_similarity: float
    wavenumbers: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "Reference": self.label,
            "Cosine Similarity": round(self.cosine_similarity, 4),
            "Similarity (%)": round(self.cosine_similarity * 100, 1),
        }


class ReferenceLibrary:
    """
    Store pre-computed feature vectors for known spectra and run
    fast nearest-neighbour search via cosine similarity.

    Usage:
        lib = ReferenceLibrary()
        lib.add_spectrum("Paracetamol", wavenumbers, intensities)
        lib.add_spectrum("Aspirin", wavenumbers2, intensities2)
        results = lib.search(query_vector, top_n=5)
    """

    def __init__(self):
        self._labels: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._spectra: List[Tuple[np.ndarray, np.ndarray]] = []  # raw wn, intensities

    def add_spectrum(
        self,
        label: str,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
    ) -> None:
        """Preprocess and store a reference spectrum."""
        from core.spectrum import preprocess
        wn, processed = preprocess(wavenumbers, intensities)
        fvec = make_feature_vector(wn, processed)
        self._labels.append(label)
        self._vectors.append(fvec)
        self._spectra.append((wn, processed))

    def add_precomputed(
        self,
        label: str,
        feature_vector: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
    ) -> None:
        """Store an already-computed feature vector."""
        self._labels.append(label)
        self._vectors.append(feature_vector)
        self._spectra.append((wavenumbers, intensities))

    @property
    def size(self) -> int:
        return len(self._labels)

    def search(
        self,
        query_vector: np.ndarray,
        top_n: int = 5,
        exclude_label: Optional[str] = None,
    ) -> List[SimilarityResult]:
        """
        Find the top-N most similar spectra by cosine similarity.
        Optionally exclude a label (e.g. searching for similar spectra
        among other uploaded files, excluding self).
        """
        if not self._vectors:
            return []

        scores = []
        for i, (label, ref_vec) in enumerate(zip(self._labels, self._vectors)):
            if exclude_label and label == exclude_label:
                continue
            sim = cosine_similarity(query_vector, ref_vec)
            wn, ints = self._spectra[i]
            scores.append(SimilarityResult(
                label=label,
                cosine_similarity=sim,
                wavenumbers=wn,
                intensities=ints,
            ))

        scores.sort(key=lambda s: s.cosine_similarity, reverse=True)
        return scores[:top_n]

