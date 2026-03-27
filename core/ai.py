"""
core/ai.py
----------
Google Gemini integration.

Responsibilities:
  - generate_compound_summary()   : brief, structured Raman-focused description
  - predict_compounds_from_peaks(): structured JSON prediction from peaks + context
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


MODEL_NAME = "gemini-2.5-flash"

_SUMMARY_PROMPT = """You are an expert analytical chemist specialising in Raman spectroscopy.
Describe the compound '{name}' (chemical group: {group}) in the context of Raman spectroscopy.
Cover:
- Key Raman-active vibrational modes and their typical wavenumber positions
- Chemical structure and functional groups
- Common real-world applications
- Any spectral peculiarities worth noting

Be concise but technically precise. Use bullet points for the spectral modes section."""

_PREDICT_PROMPT = """You are an expert Raman spectroscopist. Analyse the following spectral data and
predict the most plausible chemical compounds present.

Detected peaks (cm⁻¹): {peaks}
Identified functional groups: {functional_groups}
Sample metadata: {metadata}
Diagnostics: {diagnostics}

Return ONLY a JSON array (no preamble, no markdown) where each item has exactly:
  "compound"  : string  (IUPAC or common name)
  "confidence": string  ("High" | "Medium" | "Low")
  "reasoning" : string  (one concise sentence linking peaks to the compound)

Example:
[
  {{"compound": "Calcite", "confidence": "High", "reasoning": "Strong peak at 1085 cm⁻¹ matches the ν1 CO3²⁻ symmetric stretch."}},
  {{"compound": "Quartz", "confidence": "Medium", "reasoning": "Peak near 464 cm⁻¹ is consistent with Si–O–Si bending."}}
]"""


class GeminiAI:
    def __init__(self, api_key: str):
        if not _GENAI_AVAILABLE:
            raise ImportError("google-generativeai is not installed.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(MODEL_NAME)

    def generate_compound_summary(self, name: str, group: str) -> str:
        """Return a Raman-focused summary of a compound."""
        prompt = _SUMMARY_PROMPT.format(name=name, group=group)
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini summary failed for {name}: {e}")
            return f"⚠️ AI summary unavailable: {e}"

    def predict_compounds(
        self,
        peaks: List[float],
        functional_groups: List[Tuple[str, float]],
        diagnostics: List[str],
        metadata: Dict,
    ) -> List[Dict]:
        """
        Ask Gemini to predict compounds based on spectral features.
        Returns a list of dicts: [{"compound", "confidence", "reasoning"}, ...]
        """
        fg_str = (
            "; ".join(f"{name} @ {wn:.0f} cm⁻¹" for name, wn in functional_groups)
            or "None identified"
        )
        diag_str = "; ".join(diagnostics) or "None"
        meta_str = (
            f"State={metadata.get('sample_state','?')}, "
            f"Crystalline={metadata.get('crystalline','?')}, "
            f"Excitation={metadata.get('excitation','?')}"
        )
        prompt = _PREDICT_PROMPT.format(
            peaks=sorted([round(p, 1) for p in peaks]),
            functional_groups=fg_str,
            metadata=meta_str,
            diagnostics=diag_str,
        )
        try:
            response = self._model.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array.")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini prediction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Gemini prediction failed: {e}")
            return []
