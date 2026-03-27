"""
utils/pubchem.py
----------------
Clean PubChem PUG-REST integration with structured results.
No third-party pubchempy dependency — pure requests.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Optional

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 10


@dataclass
class PubChemResult:
    cid: int
    name: str
    iupac_name: str = ""
    molecular_formula: str = ""
    molecular_weight: str = ""
    canonical_smiles: str = ""
    description: str = ""
    url: str = ""

    @property
    def pubchem_url(self) -> str:
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.cid}"


def fetch_pubchem(compound_name: str) -> Optional[PubChemResult]:
    """
    Fetch compound data from PubChem.
    Returns None if not found or on network error.
    """
    # 1. Resolve name → CID
    try:
        r = requests.get(
            f"{_BASE}/compound/name/{requests.utils.quote(compound_name)}/cids/JSON",
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        cid = cids[0]
    except Exception:
        return None

    # 2. Fetch properties
    props = {}
    try:
        prop_names = "IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES"
        r = requests.get(
            f"{_BASE}/compound/cid/{cid}/property/{prop_names}/JSON",
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        prop_list = r.json().get("PropertyTable", {}).get("Properties", [])
        if prop_list:
            props = prop_list[0]
    except Exception:
        pass

    # 3. Fetch description
    description = ""
    try:
        r = requests.get(
            f"{_BASE}/compound/cid/{cid}/description/JSON",
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        info_list = r.json().get("InformationList", {}).get("Information", [])
        for item in info_list:
            if "Description" in item:
                description = item["Description"]
                break
    except Exception:
        pass

    return PubChemResult(
        cid=cid,
        name=compound_name,
        iupac_name=props.get("IUPACName", ""),
        molecular_formula=props.get("MolecularFormula", ""),
        molecular_weight=str(props.get("MolecularWeight", "")),
        canonical_smiles=props.get("CanonicalSMILES", ""),
        description=description,
    )
