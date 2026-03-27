"""
utils/database.py
-----------------
Load and merge compound databases from local JSON files or URLs.

Expected JSON structure (dict of category → list of compounds):
{
  "Organics": [
    {
      "Name": "Ethanol",
      "Peaks": [
        {"Wavenumber": 880, "Assignment": "C–C stretch", "RelativeIntensity": 0.8},
        ...
      ]
    }
  ],
  ...
}

Also accepts a flat list (wrapped under "Uncategorized").
"""

from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Tuple, Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15


def _load_single(path_or_url: str) -> Tuple[Dict, Optional[str], Optional[str]]:
    """Load one JSON source. Returns (data_dict, success_msg, error_msg)."""
    try:
        if path_or_url.startswith(("http://", "https://")):
            r = requests.get(path_or_url, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            src = path_or_url
        else:
            if not os.path.isabs(path_or_url):
                base = os.path.dirname(os.path.abspath(__file__))
                full = os.path.join(base, "..", path_or_url)
            else:
                full = path_or_url
            if not os.path.exists(full):
                return {}, None, f"File not found: {path_or_url}"
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
            src = os.path.basename(path_or_url)

        # Normalise to dict-of-lists
        if isinstance(data, list):
            return {"Uncategorized": data}, f"Loaded: {src} (flat list → Uncategorized)", None
        elif isinstance(data, dict):
            return data, f"Loaded: {src}", None
        else:
            return {}, None, f"Unsupported JSON structure in {src}"

    except requests.exceptions.RequestException as e:
        return {}, None, f"Network error loading {path_or_url}: {e}"
    except json.JSONDecodeError as e:
        return {}, None, f"JSON parse error in {path_or_url}: {e}"
    except Exception as e:
        return {}, None, f"Unexpected error loading {path_or_url}: {e}"


def load_database(
    paths: List[str],
) -> Tuple[Dict, List[str], List[str]]:
    """
    Load and merge multiple compound database sources.

    Returns:
        merged_db   : Dict[category, List[compound_dict]]
        success_msgs: List[str]
        error_msgs  : List[str]
    """
    merged: Dict = {}
    success_msgs: List[str] = []
    error_msgs: List[str] = []

    for path in paths:
        if not path:
            continue
        data, ok, err = _load_single(path)
        if ok:
            success_msgs.append(ok)
        if err:
            error_msgs.append(err)
        for category, compounds in data.items():
            if isinstance(compounds, list):
                merged.setdefault(category, []).extend(compounds)

    return merged, success_msgs, error_msgs
