"""
core/grouping.py
----------------
Spectrum grouping and mixture analysis for rover Raman data.

Groups similar spectra by cosine similarity, creates representative spectra,
and identifies compound mixtures.

Configuration:
- Similarity threshold: ≥ 0.95 (cosine)
- Min group size: ≥ 3 spectra
- Confidence threshold: ≥ 75% (for valid mixture component)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from core.spectrum import cosine_similarity


@dataclass
class GroupedSpectrumResult:
    """Result of grouping analysis for a set of spectra."""
    group_id: int
    member_labels: List[str]  # spectrum labels in this group
    group_size: int
    avg_similarity: float  # average cosine sim within group
    representative_wn: np.ndarray
    representative_intensity: np.ndarray
    compound_matches: List[Dict]  # top matches for this group
    mixture_detected: bool  # True if 2+ distinct compounds above threshold
    mixture_compounds: List[str]  # sorted compound names in mixture

    def to_dict(self) -> dict:
        compounds_str = ", ".join(self.mixture_compounds) if self.mixture_compounds else "None"
        return {
            "Group ID": self.group_id,
            "Members": len(self.member_labels),
            "Member Labels": ", ".join(self.member_labels),
            "Avg Similarity": round(self.avg_similarity, 3),
            "Top Match": self.compound_matches[0]["compound"] if self.compound_matches else "N/A",
            "Top Confidence (%)": round(self.compound_matches[0]["confidence"], 1) if self.compound_matches else 0,
            "Mixture Detected": "Yes" if self.mixture_detected else "No",
            "Mixture Compounds": compounds_str,
        }


class SpectrumGrouper:
    """
    Cluster uploaded spectra by cosine similarity.

    Configuration (from guide):
    - similarity_threshold: 0.95
    - min_group_size: 3
    - confidence_threshold: 0.75 (75%)
    """

    SIMILARITY_THRESHOLD = 0.95
    MIN_GROUP_SIZE = 3
    CONFIDENCE_THRESHOLD = 0.75

    def __init__(self):
        pass

    def cluster_spectra(
        self,
        labels: List[str],
        feature_vectors: List[np.ndarray],
    ) -> List[List[int]]:
        """
        Cluster spectra indices into groups using single-linkage clustering
        based on cosine similarity threshold.

        Returns list of groups, where each group is a list of indices.
        """
        n = len(labels)
        if n == 0:
            return []

        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue

            # Start a new group with spectrum i
            group = [i]
            visited[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                # Find all unvisited spectra similar to current
                for j in range(n):
                    if visited[j]:
                        continue
                    sim = cosine_similarity(
                        feature_vectors[current],
                        feature_vectors[j],
                    )
                    if sim >= self.SIMILARITY_THRESHOLD:
                        visited[j] = True
                        group.append(j)
                        queue.append(j)

            if len(group) >= self.MIN_GROUP_SIZE:
                groups.append(sorted(group))

        return groups

    def get_representative_spectrum(
        self,
        indices: List[int],
        wavenumbers_list: List[np.ndarray],
        intensities_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a representative spectrum for a group by averaging.
        Assumes all spectra share the same wavenumber axis.
        """
        if not indices:
            return np.array([]), np.array([])

        # Use first spectrum's wavenumber axis
        wn = wavenumbers_list[indices[0]]

        # Average intensities
        avg_intensity = np.zeros_like(intensities_list[0], dtype=float)
        for idx in indices:
            avg_intensity += intensities_list[idx]
        avg_intensity /= len(indices)

        return wn, avg_intensity

    def analyze_groups(
        self,
        groups: List[List[int]],
        labels: List[str],
        feature_vectors: List[np.ndarray],
        matcher_func,  # function(peaks, feature_vec) -> List[MatchResult]
        wavenumbers_list: List[np.ndarray],
        intensities_list: List[np.ndarray],
        all_results: List[Dict],  # original analysis results per spectrum
    ) -> List[GroupedSpectrumResult]:
        """
        Run compound matching on each group's representative spectrum.
        Return grouped analysis results.
        """
        grouped_results = []

        for group_id, indices in enumerate(groups):
            # Get representative spectrum
            rep_wn, rep_int = self.get_representative_spectrum(
                indices, wavenumbers_list, intensities_list
            )

            # Compute average similarity within group
            within_sim = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = cosine_similarity(
                        feature_vectors[indices[i]],
                        feature_vectors[indices[j]],
                    )
                    within_sim.append(sim)
            avg_sim = np.mean(within_sim) if within_sim else 1.0

            # Get peaks for representative spectrum
            rep_peaks_wn = []
            if all_results:
                # Merge peaks from all members in group
                peak_set = set()
                for idx in indices:
                    if idx < len(all_results):
                        for p in all_results[idx].get("peaks_wn", []):
                            peak_set.add(round(float(p), 1))
                rep_peaks_wn = sorted(list(peak_set))

            # Run matching on representative
            rep_feature_vec = self._make_feature_vector(rep_wn, rep_int)
            matches = matcher_func(rep_peaks_wn, rep_feature_vec)

            # Filter matches by confidence threshold
            valid_matches = [
                m for m in matches
                if m.confidence >= (self.CONFIDENCE_THRESHOLD * 100)
            ]

            # Detect mixture
            mixture_detected = len(valid_matches) >= 2
            mixture_compounds = sorted(list(set(m.compound for m in valid_matches)))

            result = GroupedSpectrumResult(
                group_id=group_id,
                member_labels=[labels[i] for i in indices],
                group_size=len(indices),
                avg_similarity=float(avg_sim),
                representative_wn=rep_wn,
                representative_intensity=rep_int,
                compound_matches=[
                    {
                        "compound": m.compound,
                        "confidence": m.confidence,
                        "group": m.group,
                    }
                    for m in valid_matches[:3]  # top 3
                ],
                mixture_detected=mixture_detected,
                mixture_compounds=mixture_compounds,
            )
            grouped_results.append(result)

        return grouped_results

    @staticmethod
    def _make_feature_vector(
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        range_start: int = 100,
        range_end: int = 4000,
        bin_size: int = 50,
    ) -> np.ndarray:
        """Create feature vector (binned spectrum)."""
        bins = np.arange(range_start, range_end + bin_size, bin_size)
        vec = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (wavenumbers >= bins[i]) & (wavenumbers < bins[i + 1])
            if np.any(mask):
                vec[i] = np.max(intensities[mask])
        return vec
