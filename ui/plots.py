"""
ui/plots.py
-----------
All matplotlib/plotly visualisation helpers for the Streamlit UI.
"""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_PALETTE = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea",
    "#ea580c", "#0891b2", "#65a30d", "#db2777",
]


def _base_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Normalised Intensity", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    return fig, ax


def plot_single(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    peaks_wn: np.ndarray,
    peaks_int: np.ndarray,
    label: str = "Spectrum",
    annotate: bool = True,
) -> plt.Figure:
    """Single spectrum with annotated peaks."""
    fig, ax = _base_fig()
    ax.plot(wavenumbers, intensities, color=_PALETTE[0], linewidth=1.5, label=label)
    ax.scatter(peaks_wn, peaks_int, color="#dc2626", zorder=6, s=40, label="Peaks")

    if annotate:
        for wn, intensity in zip(peaks_wn, peaks_int):
            ax.annotate(
                f"{wn:.0f}",
                xy=(wn, intensity),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#dc2626",
            )

    ax.invert_xaxis()
    ax.set_title(label, fontsize=13, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_overlay(spectra: List[Dict[str, Any]]) -> plt.Figure:
    """
    Overlay multiple spectra.
    Each dict: {"wavenumbers", "intensities", "peaks_wn", "peaks_int", "label"}
    """
    fig, ax = _base_fig(figsize=(13, 5))
    for i, s in enumerate(spectra):
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(s["wavenumbers"], s["intensities"], color=color, linewidth=1.4,
                label=s["label"], alpha=0.85)
        ax.scatter(s["peaks_wn"], s["peaks_int"], color=color, s=35, zorder=5)
    ax.invert_xaxis()
    ax.set_title("Spectra Overlay", fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    return fig


def plot_stacked(spectra: List[Dict[str, Any]]) -> plt.Figure:
    """
    Stacked view — each spectrum offset vertically for clarity.
    """
    n = len(spectra)
    fig, axes = plt.subplots(nrows=n, figsize=(13, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (s, ax) in enumerate(zip(spectra, axes)):
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(s["wavenumbers"], s["intensities"], color=color, linewidth=1.4)
        ax.scatter(s["peaks_wn"], s["peaks_int"], color="#dc2626", s=30, zorder=5)
        ax.set_ylabel(s["label"], fontsize=8)
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    axes[0].set_title("Spectra Stacked View", fontsize=13, pad=10)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    return fig


def plot_confidence_bar(matches: List[Dict]) -> plt.Figure:
    """Horizontal bar chart of top compound matches by confidence."""
    top = matches[:10]
    names = [m["Compound"] for m in top]
    confs = [m["Confidence (%)"] for m in top]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(names))))
    bars = ax.barh(names[::-1], confs[::-1], color=_PALETTE[0], alpha=0.85)

    # Colour bars by confidence level
    for bar, conf in zip(bars, confs[::-1]):
        if conf >= 60:
            bar.set_color("#16a34a")
        elif conf >= 35:
            bar.set_color("#ea580c")
        else:
            bar.set_color("#dc2626")

    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(60, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.6, label="High (≥60%)")
    ax.axvline(35, color="#ea580c", linestyle="--", linewidth=0.8, alpha=0.6, label="Medium (≥35%)")
    ax.legend(fontsize=8)
    ax.set_title("Top Compound Matches", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_comparison(
    query_wn: np.ndarray,
    query_int: np.ndarray,
    query_label: str,
    ref_wn: np.ndarray,
    ref_int: np.ndarray,
    ref_label: str,
    similarity: float = 0.0,
) -> plt.Figure:
    """Side-by-side overlay of query spectrum vs reference match."""
    fig, ax = _base_fig(figsize=(13, 5))
    ax.plot(query_wn, query_int, color=_PALETTE[0], linewidth=1.5,
            label=f"Query: {query_label}", alpha=0.9)
    ax.plot(ref_wn, ref_int, color=_PALETTE[2], linewidth=1.5,
            label=f"Match: {ref_label}", alpha=0.7, linestyle="--")
    ax.invert_xaxis()
    title = f"Spectrum Comparison (Similarity: {similarity*100:.1f}%)"
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    return fig

