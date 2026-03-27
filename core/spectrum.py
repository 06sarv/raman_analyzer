"""
core/spectrum.py
----------------
All signal processing logic: baseline correction, despiking,
peak detection, peak fitting, and feature vector creation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ─────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────

@dataclass
class Peak:
    wavenumber: float
    intensity: float
    fwhm: float = np.nan
    asymmetry: float = np.nan
    shape: str = "Unknown"
    functional_group: str = "Unassigned"


@dataclass
class Spectrum:
    wavenumbers: np.ndarray
    intensities_raw: np.ndarray
    intensities: np.ndarray = field(default=None)   # processed
    peaks: List[Peak] = field(default_factory=list)
    label: str = "Spectrum"

    def __post_init__(self):
        if self.intensities is None:
            self.intensities = self.intensities_raw.copy()


# ─────────────────────────────────────────────
# Signal pre-processing
# ─────────────────────────────────────────────

def despike(intensities: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove cosmic ray spikes via median filter."""
    return medfilt(intensities.astype(float), kernel_size=kernel_size)


def baseline_als(
    intensities: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    Peeling the fluorescence carpet from under the peaks.

    lam  : smoothness (higher → smoother baseline)
    p    : asymmetry (0.001–0.1 for Raman)
    """
    y = intensities.astype(float)
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float)
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, dtype=float)
        Z = (W + H).tocsc()          # CSC format required by spsolve
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def normalize_minmax(intensities: np.ndarray) -> np.ndarray:
    mn, mx = intensities.min(), intensities.max()
    if mx == mn:
        return np.zeros_like(intensities)
    return (intensities - mn) / (mx - mn)


def preprocess(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    do_despike: bool = True,
    do_baseline: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full pre-processing pipeline. Returns (wavenumbers, processed_intensities)."""
    y = intensities.astype(float)
    if do_despike:
        y = despike(y)
    if do_baseline:
        baseline = baseline_als(y)
        y = y - baseline
        y = np.clip(y, 0, None)          # no negative intensities
    y = normalize_minmax(y)
    return wavenumbers, y


# ─────────────────────────────────────────────
# Peak detection & fitting
# ─────────────────────────────────────────────

def _lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)


def _gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def detect_peaks(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    prominence_factor: float = 0.4,
    min_distance: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks using dynamic prominence based on signal std."""
    prominence = np.std(intensities) * prominence_factor
    idx, _ = find_peaks(intensities, prominence=prominence, distance=min_distance)
    return wavenumbers[idx], intensities[idx]


def fit_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    center: float,
    window: float = 25.0,
) -> Tuple[str, float, float]:
    """
    Fit a Lorentzian and Gaussian to a peak window.
    Returns (shape_type, fwhm, asymmetry).
    """
    mask = (wavenumbers > center - window) & (wavenumbers < center + window)
    x, y = wavenumbers[mask], intensities[mask]
    if len(x) < 5:
        return "Unknown", np.nan, np.nan

    shape, fwhm, asym = "Unknown", np.nan, np.nan
    try:
        popt_l, _ = curve_fit(_lorentzian, x, y, p0=[max(y), center, 5], maxfev=2000)
        res_l = np.sum((_lorentzian(x, *popt_l) - y) ** 2)
    except Exception:
        res_l = np.inf

    try:
        popt_g, _ = curve_fit(_gaussian, x, y, p0=[max(y), center, 5], maxfev=2000)
        res_g = np.sum((_gaussian(x, *popt_g) - y) ** 2)
    except Exception:
        res_g = np.inf

    if res_l < res_g and res_l < np.inf:
        shape = "Lorentzian"
        fwhm = abs(2 * popt_l[2])
    elif res_g < np.inf:
        shape = "Gaussian"
        fwhm = abs(2.355 * popt_g[2])

    # Asymmetry: ratio of right half-width to left half-width
    half_max = max(y) / 2
    above = x[y >= half_max]
    if len(above) >= 2:
        left_hw = center - above[0]
        right_hw = above[-1] - center
        if left_hw > 0:
            asym = right_hw / left_hw

    return shape, fwhm, asym


# ─────────────────────────────────────────────
# Feature vector for ML / cosine matching
# ─────────────────────────────────────────────

BIN_SIZE = 50          # cm⁻¹ per bin
RANGE_START = 100
RANGE_END = 4000


def make_feature_vector(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
) -> np.ndarray:
    """
    Bin the spectrum into fixed-width bins, taking the max intensity per bin.
    Produces a consistent-length feature vector regardless of instrument resolution.
    """
    bins = np.arange(RANGE_START, RANGE_END + BIN_SIZE, BIN_SIZE)
    vec = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (wavenumbers >= bins[i]) & (wavenumbers < bins[i + 1])
        if np.any(mask):
            vec[i] = np.max(intensities[mask])
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
