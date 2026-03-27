"""
app.py
------
Raman Spectrum Analyser — Streamlit Application
================================================

Architecture:
  core/spectrum.py   → signal processing (baseline, peaks, features)
  core/matcher.py    → multi-strategy compound matching
  core/expert.py     → rule-based functional group interpretation
  core/ai.py         → Google Gemini integration
  utils/pubchem.py   → PubChem REST API
  utils/database.py  → JSON database loader
  utils/report.py    → PDF report generator
  ui/plots.py        → all matplotlib visualisations

Run:
  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Must be FIRST Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Spectrum Analyser",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal modules ─────────────────────────────────────────────────────────
from core.spectrum import preprocess, detect_peaks, fit_peak, make_feature_vector, Peak
from core.matcher import CompoundMatcher, MatchResult, ReferenceLibrary
from core.expert import ExpertInterpreter, load_functional_group_rules
from core.ai import GeminiAI
from core.grouping import SpectrumGrouper
from utils.database import load_database
from utils.pubchem import fetch_pubchem
from utils.report import generate_report
from ui.plots import plot_single, plot_overlay, plot_stacked, plot_confidence_bar, plot_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource initialisers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading compound database…")
def _load_db(db_paths_key: str) -> Tuple[Dict, List[str], List[str]]:
    paths = db_paths_key.split("|")
    return load_database(paths)


@st.cache_resource(show_spinner="Configuring AI…")
def _init_ai(api_key: str) -> Optional[GeminiAI]:
    if not api_key:
        return None
    try:
        return GeminiAI(api_key)
    except Exception as e:
        st.warning(f"Gemini AI could not be initialised: {e}")
        return None


@st.cache_data(show_spinner="Fetching PubChem data…")
def _pubchem_cached(name: str):
    result = fetch_pubchem(name)
    if result is None:
        return None
    return {
        "cid": result.cid,
        "iupac_name": result.iupac_name,
        "molecular_formula": result.molecular_formula,
        "molecular_weight": result.molecular_weight,
        "canonical_smiles": result.canonical_smiles,
        "description": result.description,
        "url": result.pubchem_url,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-spectrum analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_spectrum(
    wavenumbers: np.ndarray,
    intensities_raw: np.ndarray,
    label: str,
    metadata: Dict,
    matcher: CompoundMatcher,
    expert: ExpertInterpreter,
    prominence_factor: float,
    min_distance: int,
) -> Dict:
    """
    Full analysis pipeline for a single spectrum.
    Returns a results dict with all derived data.
    """
    # 1. Pre-process
    wn, intensities = preprocess(wavenumbers, intensities_raw)

    # 2. Detect peaks
    peaks_wn, peaks_int = detect_peaks(wn, intensities, prominence_factor, min_distance)

    # 3. Fit each peak
    fitted_peaks: List[Peak] = []
    for pw, pi in zip(peaks_wn, peaks_int):
        shape, fwhm, asym = fit_peak(wn, intensities, pw)
        fitted_peaks.append(Peak(
            wavenumber=pw,
            intensity=pi,
            fwhm=fwhm,
            asymmetry=asym,
            shape=shape,
        ))

    # 4. Feature vector (for cosine matching)
    fvec = make_feature_vector(wn, intensities)

    # 5. Expert functional group assignment
    functional_groups = expert.assign_functional_groups(peaks_wn.tolist())
    # Attach to peaks
    fg_map = {round(wn_): lbl for lbl, wn_ in functional_groups}
    for p in fitted_peaks:
        p.functional_group = fg_map.get(round(p.wavenumber), "Unassigned")

    # 6. Diagnostics
    diagnostics = expert.get_diagnostics(peaks_wn.tolist(), metadata)

    # 7. Compound matching
    matches: List[MatchResult] = matcher.match(peaks_wn.tolist(), fvec)

    return {
        "label": label,
        "wavenumbers": wn,
        "intensities": intensities,
        "peaks_wn": peaks_wn,
        "peaks_int": peaks_int,
        "fitted_peaks": fitted_peaks,
        "functional_groups": functional_groups,
        "diagnostics": diagnostics,
        "matches": matches,
        "feature_vector": fvec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _peak_dataframe(fitted_peaks: List[Peak]) -> pd.DataFrame:
    rows = [{
        "Wavenumber (cm⁻¹)": f"{p.wavenumber:.1f}",
        "Intensity": f"{p.intensity:.4f}",
        "FWHM (cm⁻¹)": f"{p.fwhm:.2f}" if not np.isnan(p.fwhm) else "—",
        "Asymmetry": f"{p.asymmetry:.3f}" if not np.isnan(p.asymmetry) else "—",
        "Shape": p.shape,
        "Functional Group": p.functional_group,
    } for p in fitted_peaks]
    return pd.DataFrame(rows)


def _match_dataframe(matches: List[MatchResult], max_rows: int = 20) -> pd.DataFrame:
    rows = [m.to_dict() for m in matches[:max_rows]]
    return pd.DataFrame(rows)


def _confidence_badge(conf: float) -> str:
    if conf >= 60:
        return f"HIGH {conf:.1f}%"
    elif conf >= 35:
        return f"MED {conf:.1f}%"
    return f"LOW {conf:.1f}%"


def _get_api_key_default() -> str:
    """Safely retrieve the API key from secrets or env without crashing."""
    try:
        return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def _raman_shift_from_wavelength_nm(
    wavelengths_nm: np.ndarray,
    laser_wavelength_nm: float,
) -> np.ndarray:
    """Convert measured wavelengths to Raman shift in cm^-1."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if laser_wavelength_nm <= 0:
        raise ValueError("Laser wavelength must be positive.")
    if np.any(wavelengths_nm <= 0):
        raise ValueError("Computed wavelengths must be positive.")
    return (1.0 / laser_wavelength_nm - 1.0 / wavelengths_nm) * 1e7


def _wavelength_from_raman_shift_cm1(
    raman_shift_cm1: np.ndarray,
    laser_wavelength_nm: float,
) -> np.ndarray:
    """Convert Raman shift in cm^-1 to measured wavelength in nm."""
    raman_shift_cm1 = np.asarray(raman_shift_cm1, dtype=float)
    if laser_wavelength_nm <= 0:
        raise ValueError("Laser wavelength must be positive.")
    denominator = 1.0 / laser_wavelength_nm - raman_shift_cm1 / 1e7
    if np.any(denominator <= 0):
        raise ValueError("Raman shift range is incompatible with the laser wavelength.")
    return 1.0 / denominator


def _extract_pixel_columns(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """
    Extract CCD pixel columns such as pixel_0_adc, pixel_1_adc, ... in numeric order.
    Falls back to numeric columns after timestamp/frame if needed.
    """
    pixel_columns: List[Tuple[int, str]] = []
    for col in df.columns:
        match = re.fullmatch(r"pixel_(\d+)_adc", str(col).strip())
        if match:
            pixel_columns.append((int(match.group(1)), col))

    if pixel_columns:
        return sorted(pixel_columns, key=lambda item: item[0])

    fallback_columns = [
        col for col in df.columns
        if str(col).strip().lower() not in {"timestamp", "frame"}
    ]
    extracted: List[Tuple[int, str]] = []
    for idx, col in enumerate(fallback_columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            extracted.append((idx, col))
    return extracted


def convert_raw_ccd_to_raman_shift(
    df: pd.DataFrame,
    calibration_mode: str,
    laser_wavelength_nm: float,
    nominal_shift_start_cm1: float,
    nominal_shift_end_cm1: float,
    anchor_pixel: int,
    anchor_wavelength_nm: float,
    nm_per_pixel: float,
    aggregation: str = "Mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse a raw CCD frame stack into a single spectrum, calibrate
    CCD pixel positions to wavelength, then convert to Raman shift.
    """
    pixel_columns = _extract_pixel_columns(df)
    if not pixel_columns:
        raise ValueError("No CCD pixel columns found.")

    pixel_df = df[[col for _, col in pixel_columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_columns = [col for col in pixel_df.columns if pixel_df[col].notna().any()]
    if not valid_columns:
        raise ValueError("CCD file has no usable intensity values.")
    pixel_df = pixel_df[valid_columns].dropna(axis=0, how="all")
    if pixel_df.empty:
        raise ValueError("CCD file has no usable intensity values.")

    filtered_pixel_columns = [
        (idx, col) for idx, col in pixel_columns if col in valid_columns
    ]
    pixel_indices = np.array([idx for idx, _ in filtered_pixel_columns], dtype=float)

    if aggregation == "Median":
        intensities = pixel_df.median(axis=0).to_numpy(dtype=float)
    else:
        intensities = pixel_df.mean(axis=0).to_numpy(dtype=float)

    if calibration_mode == "Estimated Raman Range":
        shift_start = float(nominal_shift_start_cm1)
        shift_end = float(nominal_shift_end_cm1)
        if shift_start == shift_end:
            raise ValueError("Estimated Raman range start and end must differ.")
        raman_shift = np.linspace(shift_start, shift_end, len(pixel_indices))
        wavelengths_nm = _wavelength_from_raman_shift_cm1(raman_shift, laser_wavelength_nm)
    else:
        wavelengths_nm = anchor_wavelength_nm + (pixel_indices - anchor_pixel) * nm_per_pixel
        raman_shift = _raman_shift_from_wavelength_nm(wavelengths_nm, laser_wavelength_nm)

    valid_mask = np.isfinite(raman_shift) & np.isfinite(intensities)
    raman_shift = raman_shift[valid_mask]
    intensities = intensities[valid_mask]

    if len(raman_shift) < 2:
        raise ValueError("Not enough calibrated CCD points after conversion.")

    order = np.argsort(raman_shift)
    return raman_shift[order], intensities[order]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict:
    st.sidebar.title("Configuration")

    # ── Database ──────────────────────────────────────────────
    st.sidebar.header("Compound Database")
    uploaded_dbs = st.sidebar.file_uploader(
        "Upload JSON database(s)",
        type=["json"],
        accept_multiple_files=True,
        help="JSON files mapping category → list of compounds with peak data.",
    )

    fg_db_file = st.sidebar.file_uploader(
        "Upload Functional Group Rules (JSON)",
        type=["json"],
        help="List of {wavenumber_range_cm-1, vibrational_mode, compound_functionality}.",
    )

    # ── API key ───────────────────────────────────────────────
    st.sidebar.header("AI Settings")
    api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        value=_get_api_key_default(),
        help="Your Gemini API key. Set GEMINI_API_KEY in secrets.toml or env.",
    )

    # ── Detection parameters ──────────────────────────────────
    st.sidebar.header("Peak Detection")
    prominence_factor = st.sidebar.slider(
        "Prominence factor",
        min_value=0.1, max_value=2.0, value=0.4, step=0.05,
        help="Higher = fewer, stronger peaks detected.",
    )
    min_distance = st.sidebar.slider(
        "Min peak distance (pts)",
        min_value=3, max_value=50, value=10, step=1,
        help="Minimum number of data points between two peaks.",
    )
    tolerance = st.sidebar.slider(
        "Match tolerance (cm⁻¹)",
        min_value=5, max_value=80, value=30, step=5,
        help="How close an observed peak must be to a reference peak to count as a match.",
    )

    # ── Sample metadata ───────────────────────────────────────
    st.sidebar.header("Sample Metadata")
    excitation = st.sidebar.selectbox(
        "Excitation laser", ["NIR", "Visible", "UV"], index=0
    )
    sample_state = st.sidebar.selectbox(
        "Sample state", ["Solid", "Liquid", "Gas"], index=0
    )
    crystalline = st.sidebar.selectbox("Crystalline?", ["Yes", "No"], index=0)

    # ── Raw CCD calibration ──────────────────────────────────
    st.sidebar.header("Raw CCD Calibration")
    st.sidebar.caption(
        "Used only for raw CCD CSV uploads. You can estimate a Raman-shift span "
        "across the CCD or enter a manual wavelength calibration."
    )
    calibration_mode = st.sidebar.selectbox(
        "Calibration mode",
        ["Estimated Raman Range", "Manual Wavelength Calibration"],
        index=0,
        help="Estimated mode spreads the CCD pixels across a Raman-shift window. Manual mode uses wavelength calibration values.",
    )
    laser_wavelength_nm = st.sidebar.number_input(
        "Laser wavelength (nm)",
        min_value=100.0,
        max_value=2000.0,
        value=785.0,
        step=1.0,
        help="Laser wavelength used to convert calibrated wavelength to Raman shift.",
    )
    nominal_shift_start_cm1 = st.sidebar.number_input(
        "Estimated Raman start (cm⁻¹)",
        min_value=-5000.0,
        max_value=50000.0,
        value=100.0,
        step=10.0,
        help="Used in estimated mode for the first side of the CCD span.",
    )
    nominal_shift_end_cm1 = st.sidebar.number_input(
        "Estimated Raman end (cm⁻¹)",
        min_value=-5000.0,
        max_value=50000.0,
        value=2000.0,
        step=10.0,
        help="Used in estimated mode for the opposite side of the CCD span.",
    )
    anchor_pixel = st.sidebar.number_input(
        "Reference pixel",
        min_value=0,
        max_value=100000,
        value=20,
        step=1,
        help="Used only in manual wavelength calibration mode.",
    )
    anchor_wavelength_nm = st.sidebar.number_input(
        "Reference wavelength (nm)",
        min_value=1.0,
        max_value=5000.0,
        value=800.0,
        step=0.1,
        help="Used only in manual wavelength calibration mode.",
    )
    nm_per_pixel = st.sidebar.number_input(
        "Wavelength step (nm/pixel)",
        min_value=-100.0,
        max_value=100.0,
        value=0.08,
        step=0.01,
        format="%.4f",
        help="Used only in manual wavelength calibration mode.",
    )
    ccd_aggregation = st.sidebar.selectbox(
        "CCD frame aggregation",
        ["Mean", "Median"],
        index=0,
        help="How to collapse multiple CCD frames into one spectrum.",
    )

    # ── Plot style ────────────────────────────────────────────
    st.sidebar.header("Visualisation")
    plot_type = st.sidebar.radio(
        "Multi-spectrum plot style", ["Overlay", "Stacked"], horizontal=True
    )
    annotate_peaks = st.sidebar.checkbox("Annotate peak wavenumbers", value=True)

    return {
        "uploaded_dbs": uploaded_dbs,
        "fg_db_file": fg_db_file,
        "api_key": api_key,
        "prominence_factor": prominence_factor,
        "min_distance": min_distance,
        "tolerance": tolerance,
        "metadata": {
            "excitation": excitation,
            "sample_state": sample_state,
            "crystalline": crystalline,
        },
        "raw_ccd": {
            "calibration_mode": calibration_mode,
            "laser_wavelength_nm": laser_wavelength_nm,
            "nominal_shift_start_cm1": nominal_shift_start_cm1,
            "nominal_shift_end_cm1": nominal_shift_end_cm1,
            "anchor_pixel": anchor_pixel,
            "anchor_wavelength_nm": anchor_wavelength_nm,
            "nm_per_pixel": nm_per_pixel,
            "aggregation": ccd_aggregation,
        },
        "plot_type": plot_type,
        "annotate_peaks": annotate_peaks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tabs: Results for a single spectrum
# ─────────────────────────────────────────────────────────────────────────────

def render_spectrum_results(
    result: Dict,
    ai: Optional[GeminiAI],
    metadata: Dict,
    annotate_peaks: bool,
    ref_lib: Optional[ReferenceLibrary] = None,
):
    label = result["label"]
    matches = result["matches"]
    fitted_peaks = result["fitted_peaks"]
    functional_groups = result["functional_groups"]
    diagnostics = result["diagnostics"]

    tab_spectrum, tab_peaks, tab_matches, tab_sim, tab_ai, tab_export = st.tabs([
        "Spectrum", "Peaks", "Matches", "Similarity", "AI Analysis", "Export",
    ])

    # ── Tab 1: Spectrum ───────────────────────────────────────
    with tab_spectrum:
        fig_spectrum = plot_single(
            result["wavenumbers"], result["intensities"],
            result["peaks_wn"], result["peaks_int"],
            label=label, annotate=annotate_peaks,
        )
        st.pyplot(fig_spectrum, use_container_width=True)
        plt.close(fig_spectrum)

        if diagnostics:
            st.markdown("**Diagnostics**")
            for d in diagnostics:
                st.info(d)

    # ── Tab 2: Peaks ──────────────────────────────────────────
    with tab_peaks:
        peak_df = _peak_dataframe(fitted_peaks)
        if not peak_df.empty:
            st.dataframe(peak_df, use_container_width=True, hide_index=True)
            # CSV download
            st.download_button(
                "Download Peaks CSV",
                data=peak_df.to_csv(index=False),
                file_name=f"{label}_peaks.csv",
                mime="text/csv",
            )
        else:
            st.warning("No peaks detected. Try lowering the prominence factor.")

        if functional_groups:
            st.markdown("**Identified Functional Groups**")
            fg_df = pd.DataFrame(
                [
                    {
                        "Functional Group": fg_name,
                        "Wavenumber (cm⁻¹)": round(float(fg_wn), 1),
                    }
                    for fg_name, fg_wn in functional_groups
                ]
            )
            fg_df = fg_df.sort_values("Wavenumber (cm⁻¹)").reset_index(drop=True)
            st.table(fg_df)

    # ── Tab 3: Matches ────────────────────────────────────────
    with tab_matches:
        if not matches:
            st.warning("No compound matches found. Try increasing the tolerance or uploading a larger database.")
        else:
            match_df = _match_dataframe(matches)
            st.dataframe(
                match_df.style.background_gradient(
                    subset=["Confidence (%)"], cmap="RdYlGn", vmin=0, vmax=100
                ),
                use_container_width=True, hide_index=True,
            )

            # Confidence bar chart
            st.pyplot(plot_confidence_bar(match_df.to_dict("records")), use_container_width=True)

            # CSV download
            st.download_button(
                "Download Matches CSV",
                data=match_df.to_csv(index=False),
                file_name=f"{label}_matches.csv",
                mime="text/csv",
            )

            # PubChem details for top match
            top = matches[0]
            with st.expander(f"PubChem details - {top.compound}"):
                pc_data = _pubchem_cached(top.compound)
                if pc_data:
                    c1, c2 = st.columns(2)
                    c1.metric("CID", pc_data["cid"])
                    c1.markdown(f"**Formula:** `{pc_data['molecular_formula']}`")
                    c1.markdown(f"**Mol. Weight:** {pc_data['molecular_weight']} g/mol")
                    c2.markdown(f"**IUPAC Name:** {pc_data['iupac_name']}")
                    c2.markdown(f"**SMILES:** `{pc_data['canonical_smiles']}`")
                    if pc_data.get("description"):
                        st.markdown(pc_data["description"])
                    st.markdown(f"[Open in PubChem ↗]({pc_data['url']})")
                else:
                    st.info("No PubChem record found for this compound.")

    # ── Tab 4: Similarity ─────────────────────────────────────
    with tab_sim:
        if not ref_lib or ref_lib.size <= 1:
            st.info("Upload multiple spectra to enable cross-spectrum similarity search.")
        else:
            st.markdown("### Spectrum Similarity Search")
            st.caption(
                "Comparing the full spectrum (feature vector) of this sample against all "
                "other uploaded spectra in the current session."
            )
            sim_results = ref_lib.search(result["feature_vector"], top_n=5, exclude_label=label)

            if not sim_results:
                st.warning("No other spectra available for comparison.")
            else:
                sim_df = pd.DataFrame([s.to_dict() for s in sim_results])
                st.table(sim_df)

                top_sim = sim_results[0]
                st.markdown(f"**Best Match:** {top_sim.label}")
                fig_comp = plot_comparison(
                    query_wn=result["wavenumbers"],
                    query_int=result["intensities"],
                    query_label=label,
                    ref_wn=top_sim.wavenumbers,
                    ref_int=top_sim.intensities,
                    ref_label=top_sim.label,
                    similarity=top_sim.cosine_similarity,
                )
                st.pyplot(fig_comp, use_container_width=True)
                plt.close(fig_comp)

    # ── Tab 5: AI Analysis ────────────────────────────────────
    with tab_ai:
        if ai is None:
            st.warning("Gemini API key not configured. Add it in the sidebar.")
        else:
            # AI compound summary for top match
            if matches:
                top = matches[0]
                with st.expander(
                    f"AI Summary - {top.compound} ({_confidence_badge(top.confidence)})",
                    expanded=True,
                ):
                    with st.spinner("Generating AI summary…"):
                        summary = ai.generate_compound_summary(top.compound, top.group)
                    st.markdown(summary)
                    st.session_state[f"ai_summary_{label}"] = summary

            # AI compound predictions
            st.markdown("---")
            st.markdown("### AI Compound Prediction")
            st.caption(
                "Gemini analyses your detected peaks and functional groups to suggest "
                "additional plausible compounds beyond the database matching."
            )
            if st.button("Run AI Prediction", key=f"ai_pred_btn_{label}"):
                with st.spinner("Asking Gemini…"):
                    predictions = ai.predict_compounds(
                        result["peaks_wn"].tolist(),
                        functional_groups,
                        diagnostics,
                        metadata,
                    )
                st.session_state[f"ai_pred_result_{label}"] = predictions

            predictions = st.session_state.get(f"ai_pred_result_{label}", [])
            if predictions:
                for i, p in enumerate(predictions, 1):
                    conf_label = p.get("confidence", "")
                    icon = "[HIGH]" if conf_label == "High" else ("[MED]" if conf_label == "Medium" else "[LOW]")
                    st.markdown(
                        f"**{i}. {p.get('compound', 'Unknown')}** {icon} {conf_label}"
                    )
                    st.caption(p.get("reasoning", ""))

    # ── Tab 5: Export ─────────────────────────────────────────
    with tab_export:
        st.markdown("### Generate PDF Report")
        st.caption("Includes spectrum plot, peaks table, matches, AI summary, and PubChem data.")

        if st.button("Generate PDF Report", key=f"pdf_{label}", type="primary"):
            peak_df_export = _peak_dataframe(fitted_peaks)
            match_df_export = _match_dataframe(matches)

            ai_summary = st.session_state.get(f"ai_summary_{label}", "")
            ai_predictions = st.session_state.get(f"ai_pred_result_{label}", [])

            pc_data = None
            if matches:
                pc_data = _pubchem_cached(matches[0].compound)

            fig_for_pdf = plot_single(
                result["wavenumbers"], result["intensities"],
                result["peaks_wn"], result["peaks_int"],
                label=label, annotate=True,
            )

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            with st.spinner("Building PDF…"):
                generate_report(
                    output_path=tmp_path,
                    spectrum_label=label,
                    fig=fig_for_pdf,
                    peak_df=peak_df_export,
                    match_df=match_df_export,
                    ai_summary=ai_summary,
                    pubchem_info=pc_data,
                    metadata=metadata,
                    ai_predictions=ai_predictions,
                )
            plt.close(fig_for_pdf)

            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()
            os.unlink(tmp_path)

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"{label}_raman_report.pdf",
                mime="application/pdf",
            )
            st.success("PDF ready!")


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Analysis & Mixture Detection
# ─────────────────────────────────────────────────────────────────────────────

def render_grouped_results(grouped_results):
    """Render grouped spectrum analysis results in table format."""
    if not grouped_results:
        st.info("No spectrum groups found. Try uploading more spectra or adjusting the similarity threshold.")
        return
    
    st.subheader("Spectrum Grouping & Mixture Detection")
    st.caption(
        "Spectra grouped by cosine similarity (≥0.95) with representative spectrum matching "
        "and mixture detection (≥2 compounds at ≥75% confidence)."
    )
    
    # Convert grouped results to DataFrame
    group_rows = []
    for group_result in grouped_results:
        top_compound = group_result.compound_matches[0]["compound"] if group_result.compound_matches else "N/A"
        top_confidence = group_result.compound_matches[0]["confidence"] if group_result.compound_matches else 0
        
        row = {
            "Group ID": group_result.group_id,
            "Members": group_result.group_size,
            "Member Labels": ", ".join(group_result.member_labels[:3]) + 
                            ("..." if len(group_result.member_labels) > 3 else ""),
            "Top Match": top_compound,
            "Confidence (%)": round(top_confidence * 100, 1),
            "Mixture Detected": "Yes" if group_result.mixture_detected else "No",
            "Mixture Compounds": ", ".join(group_result.mixture_compounds) if group_result.mixture_compounds else "—",
        }
        group_rows.append(row)
    
    group_df = pd.DataFrame(group_rows)
    
    # Display with conditional formatting
    st.table(group_df)
    
    # CSV download
    csv_data = group_df.to_csv(index=False)
    st.download_button(
        "Download Grouped Results CSV",
        data=csv_data,
        file_name="grouped_spectra_analysis.csv",
        mime="text/csv",
    )
    
    # Detailed view per group
    st.markdown("---")
    st.markdown("### Detailed Group Analysis")
    for i, group_result in enumerate(grouped_results, 1):
        with st.expander(f"Group {group_result.group_id} ({group_result.group_size} members)"):
            st.markdown(f"**All Matches:** {len(group_result.compound_matches)} compounds found")
            st.markdown(f"**Average Within-Group Similarity:** {round(group_result.avg_similarity, 3)}")
            
            if group_result.compound_matches:
                match_rows = []
                for match in group_result.compound_matches:
                    match_rows.append({
                        "Compound": match["compound"],
                        "Confidence (%)": round(match["confidence"] * 100, 1),
                        "Group": match["group"],
                    })
                match_df = pd.DataFrame(match_rows)
                st.table(match_df)
            
            if group_result.mixture_detected:
                st.info(f"🔬 **Mixture Detected:** {', '.join(group_result.mixture_compounds)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("Raman Spectrum Analyser")
    st.caption(
        "Upload a Raman CSV or raw CCD CSV, configure your database, and let the multi-strategy "
        "matching engine and AI identify your compounds."
    )

    cfg = render_sidebar()
    metadata = cfg["metadata"]

    # ── Load databases ────────────────────────────────────────
    db: Dict = {}
    db_msgs: List[str] = []

    # Load from user-uploaded files
    if cfg["uploaded_dbs"]:
        import json
        for uf in cfg["uploaded_dbs"]:
            try:
                data = json.load(uf)
                if isinstance(data, dict):
                    for cat, compounds in data.items():
                        db.setdefault(cat, []).extend(compounds)
                elif isinstance(data, list):
                    db.setdefault("Uncategorized", []).extend(data)
                db_msgs.append(f"Loaded: {uf.name}")
            except Exception as e:
                db_msgs.append(f"Error - {uf.name}: {e}")

    n_compounds = sum(len(v) for v in db.values())

    if n_compounds == 0:
        st.info(
            "No compound database loaded. Upload a JSON database in the sidebar to enable matching."
        )
    else:
        st.sidebar.success(f"{n_compounds} compounds across {len(db)} categories loaded.")

    # ── Load functional group rules ───────────────────────────
    fg_rules = []
    if cfg["fg_db_file"]:
        import json
        try:
            fg_rules = json.load(cfg["fg_db_file"])
        except Exception as e:
            st.warning(f"Could not load functional group rules: {e}")

    expert = ExpertInterpreter(fg_rules)
    matcher = CompoundMatcher(db, tolerance=cfg["tolerance"], min_matches=1)
    ai = _init_ai(cfg["api_key"]) if cfg["api_key"] else None

    # ── Initialisation messages ───────────────────────────────
    if db_msgs:
        with st.expander("Database load log", expanded=False):
            for m in db_msgs:
                st.write(m)

    # ── File upload ───────────────────────────────────────────
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload Raman CSV file(s)  (first column: wavenumber, remaining columns: intensity)",
        type=["csv"],
        accept_multiple_files=True,
    )
    uploaded_ccd_files = st.file_uploader(
        "Upload raw CCD CSV file(s)  (rows = frames, columns = CCD pixels)",
        type=["csv"],
        accept_multiple_files=True,
        help="Expected columns look like timestamp, frame, pixel_0_adc, pixel_1_adc, ... The app converts these into a Raman-style CSV before analysis.",
    )

    if not uploaded_files and not uploaded_ccd_files:
        st.markdown(
            """
            **How to use:**
            1. Upload your compound database JSON in the sidebar
            2. (Optional) Upload functional group rules JSON
            3. (Optional) Enter your Gemini API key for AI features
            4. Upload one or more Raman spectra CSV files or raw CCD CSV files here
            5. Raw CCD files are converted into Raman-shift CSVs, then analysed
            6. Explore peaks, matches, AI analysis, and download reports
            """
        )
        return

    # ── Analyse all files ─────────────────────────────────────
    all_results: List[Dict] = []
    converted_ccd_exports: List[Dict[str, object]] = []

    with st.status("Analysing spectra…", expanded=True) as status:
        for uf in uploaded_files:
            try:
                df = pd.read_csv(uf)
                if df.shape[1] < 2:
                    st.error(f"{uf.name}: needs at least 2 columns.")
                    continue
                wavenumbers = df.iloc[:, 0].values.astype(float)
                n_intensity_cols = df.shape[1] - 1
                for col_i in range(n_intensity_cols):
                    label = (
                        uf.name if n_intensity_cols == 1
                        else f"{uf.name} — col {col_i + 1}"
                    )
                    st.write(f"Processing: **{label}**")
                    intensities = df.iloc[:, col_i + 1].values.astype(float)
                    result = analyse_spectrum(
                        wavenumbers, intensities, label, metadata,
                        matcher, expert,
                        cfg["prominence_factor"], cfg["min_distance"],
                    )
                    all_results.append(result)
            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")

        for uf in uploaded_ccd_files:
            try:
                df = pd.read_csv(uf)
                st.write(f"Processing raw CCD: **{uf.name}**")
                wavenumbers, intensities = convert_raw_ccd_to_raman_shift(
                    df=df,
                    calibration_mode=cfg["raw_ccd"]["calibration_mode"],
                    laser_wavelength_nm=cfg["raw_ccd"]["laser_wavelength_nm"],
                    nominal_shift_start_cm1=cfg["raw_ccd"]["nominal_shift_start_cm1"],
                    nominal_shift_end_cm1=cfg["raw_ccd"]["nominal_shift_end_cm1"],
                    anchor_pixel=cfg["raw_ccd"]["anchor_pixel"],
                    anchor_wavelength_nm=cfg["raw_ccd"]["anchor_wavelength_nm"],
                    nm_per_pixel=cfg["raw_ccd"]["nm_per_pixel"],
                    aggregation=cfg["raw_ccd"]["aggregation"],
                )
                converted_df = pd.DataFrame({
                    "Raman Shift": wavenumbers,
                    uf.name.replace(".csv", ""): intensities,
                })
                converted_ccd_exports.append(
                    {
                        "name": uf.name,
                        "dataframe": converted_df,
                    }
                )
                result = analyse_spectrum(
                    wavenumbers, intensities, f"{uf.name} — raw CCD", metadata,
                    matcher, expert,
                    cfg["prominence_factor"], cfg["min_distance"],
                )
                all_results.append(result)
            except Exception as e:
                st.error(f"Error processing raw CCD file {uf.name}: {e}")
        status.update(label="Analysis complete", state="complete", expanded=False)

    if not all_results:
        st.error("No spectra could be processed.")
        return

    if converted_ccd_exports:
        st.markdown("---")
        st.subheader("Converted Raw CCD Spectra")
        st.caption(
            "Each raw CCD file is converted into a Raman-style table with "
            "Raman shift in the first column and one intensity column."
        )
        for export in converted_ccd_exports:
            converted_df = export["dataframe"]
            with st.expander(f"Converted CSV Preview - {export['name']}", expanded=False):
                st.dataframe(converted_df.head(25), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Converted Raman CSV",
                    data=converted_df.to_csv(index=False),
                    file_name=f"{export['name'].replace('.csv', '')}_converted_raman.csv",
                    mime="text/csv",
                    key=f"download_converted_{export['name']}",
                )

    # ── Build Reference Library from session uploads ──────────
    ref_lib = ReferenceLibrary()
    for r in all_results:
        ref_lib.add_precomputed(
            label=r["label"],
            feature_vector=r["feature_vector"],
            wavenumbers=r["wavenumbers"],
            intensities=r["intensities"],
        )

    # ── Spectrum Grouping & Mixture Detection (if ≥3 spectra) ──
    grouped_results = []
    if len(all_results) >= 3:
        st.markdown("---")
        st.subheader("Grouping Similar Spectra")
        
        # Prepare data for grouping
        labels = [r["label"] for r in all_results]
        feature_vectors = [r["feature_vector"] for r in all_results]
        wavenumbers_list = [r["wavenumbers"] for r in all_results]
        intensities_list = [r["intensities"] for r in all_results]
        
        try:
            with st.spinner("Clustering spectra by similarity…"):
                grouper = SpectrumGrouper()
                groups = grouper.cluster_spectra(labels, feature_vectors)
                grouped_results = grouper.analyze_groups(
                    groups, labels, feature_vectors, matcher.match,
                    wavenumbers_list, intensities_list, all_results
                )
            st.success(f"Found {len(grouped_results)} group(s) with ≥3 similar spectra")
        except Exception as e:
            st.warning(f"Grouping error: {e}")

    # ─────────────────────────────────────────────────────────────
    # Display grouped results (if any groups found)
    if grouped_results:
        st.markdown("---")
        render_grouped_results(grouped_results)

    # ── Multi-spectrum comparison plot ────────────────────────
    if len(all_results) > 1:
        st.markdown("---")
        st.subheader("Multi-Spectrum Comparison")
        spectra_plot_data = [{
            "wavenumbers": r["wavenumbers"],
            "intensities": r["intensities"],
            "peaks_wn": r["peaks_wn"],
            "peaks_int": r["peaks_int"],
            "label": r["label"],
        } for r in all_results]

        if cfg["plot_type"] == "Overlay":
            st.pyplot(plot_overlay(spectra_plot_data), use_container_width=True)
        else:
            st.pyplot(plot_stacked(spectra_plot_data), use_container_width=True)

    # ── Per-spectrum detailed results ─────────────────────────
    st.markdown("---")
    if len(all_results) == 1:
        render_spectrum_results(
            all_results[0], ai, metadata, cfg["annotate_peaks"], ref_lib
        )
    else:
        spectrum_labels = [r["label"] for r in all_results]
        selected = st.selectbox("Select spectrum to inspect:", spectrum_labels)
        result = next(r for r in all_results if r["label"] == selected)
        render_spectrum_results(result, ai, metadata, cfg["annotate_peaks"], ref_lib)


if __name__ == "__main__":
    main()
