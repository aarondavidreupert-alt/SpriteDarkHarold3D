"""
FRM catalog — parses Fallout 2 critter FRM filenames into structured entries.

Naming scheme:
    [TT][NNNN][AA].FRM
      TT   2-letter type code   (HF/HM/HA/NF/NM/NA/MA…)
      NNNN variable-length character/armor name
      AA   2-letter animation code

See https://falloutmods.fandom.com/wiki/Critter_FRM_nomenclature_(naming_system)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ANIM_CODES: Dict[str, str] = {
    "AA": "Stand/Idle",
    "AB": "Walk",
    "AN": "Dodge",
    "AO": "Hit Front",
    "AP": "Hit Back",
    "AQ": "Punch",
    "AR": "Kick",
    "AS": "Throw",
    "AT": "Run",
    "AK": "Use Ground",
    "AL": "Use Middle",
    "BA": "Death: Fall Back",
    "BB": "Death: Fall Front",
    "BD": "Death: Big Hole",
    "BF": "Death: Chunks",
    "BG": "Death: Dancing",
    "BH": "Death: Electrify",
    "BI": "Death: Sliced",
    "BJ": "Death: Burned",
    "BK": "Death: Electrified",
    "BL": "Death: Exploded",
    "BM": "Death: Melted",
    "BN": "Death: Fire Dance",
    "DA": "Knife Idle",   "DB": "Knife Walk",   "DF": "Knife Thrust",
    "EA": "Melee Idle",   "EB": "Melee Walk",   "EF": "Melee Thrust",
    "FA": "Hammer Idle",  "FB": "Hammer Walk",
    "GA": "Spear Idle",   "GB": "Spear Walk",   "GF": "Spear Thrust",
    "HA": "Pistol Idle",  "HB": "Pistol Walk",  "HJ": "Pistol Shot",
    "IA": "SMG Idle",     "IB": "SMG Walk",     "IJ": "SMG Shot",
    "JA": "Rifle Idle",   "JB": "Rifle Walk",   "JJ": "Rifle Shot",
    "KA": "Heavy Idle",   "KB": "Heavy Walk",   "KJ": "Heavy Shot",
    "LA": "Minigun Idle", "LB": "Minigun Walk",
    "MA": "Rocket Idle",  "MB": "Rocket Walk",  "MJ": "Rocket Shot",
    "RA": "Dead: Fall Back", "RB": "Dead: Fall Front",
    "RD": "Dead: Big Hole",  "RF": "Dead: Chunks",
}

TYPE_CODES: Dict[str, str] = {
    "HA": "Hero Androgynous", "HF": "Hero Female", "HM": "Hero Male",
    "NA": "NPC Androgynous",  "NF": "NPC Female",  "NM": "NPC Male",
    "MA": "Monster",
}


@dataclass
class FrmEntry:
    path:       str
    filename:   str
    type_code:  str
    char_name:  str
    anim_code:  str
    type_label: str
    anim_label: str
    char_label: str


class FrmCatalog:
    """Walk a folder, parse all .frm filenames, expose filter/matrix queries."""

    def __init__(self):
        self.entries: List[FrmEntry] = []

    # ------------------------------------------------------------------

    def scan(self, folder: str):
        """Recursively scan *folder* for .frm files and parse names."""
        self.entries.clear()
        if not folder or not os.path.isdir(folder):
            return
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if not fn.upper().endswith(".FRM"):
                    continue
                stem = fn[:-4].upper()
                if len(stem) < 4:
                    continue
                anim_code = stem[-2:]
                type_code = stem[:2]
                char_name = stem[2:-2]
                self.entries.append(FrmEntry(
                    path=os.path.join(root, fn),
                    filename=fn,
                    type_code=type_code,
                    char_name=char_name,
                    anim_code=anim_code,
                    type_label=TYPE_CODES.get(type_code, type_code),
                    anim_label=ANIM_CODES.get(anim_code, anim_code),
                    char_label=char_name,
                ))

    # ------------------------------------------------------------------

    def unique_values(self, field: str) -> List[str]:
        return sorted({getattr(e, field) for e in self.entries})

    def filter(
        self,
        type_codes: Optional[List[str]] = None,
        char_names: Optional[List[str]] = None,
        anim_codes: Optional[List[str]] = None,
    ) -> List[FrmEntry]:
        result = self.entries
        if type_codes:
            tcs = set(type_codes)
            result = [e for e in result if e.type_code in tcs]
        if char_names:
            cs = set(char_names)
            result = [e for e in result if e.char_name in cs]
        if anim_codes:
            acs = set(anim_codes)
            result = [e for e in result if e.anim_code in acs]
        return result

    def as_matrix(
        self,
        row_field: str,
        col_field: str,
        entries: Optional[List[FrmEntry]] = None,
    ) -> Tuple[List[str], List[str], Dict[Tuple[str, str], FrmEntry]]:
        """
        (row_labels, col_labels, cell_map) where cell_map[(row_val, col_val)]
        is the first FrmEntry matching that intersection.
        """
        src = entries if entries is not None else self.entries
        rows = sorted({getattr(e, row_field) for e in src})
        cols = sorted({getattr(e, col_field) for e in src})
        cell_map: Dict[Tuple[str, str], FrmEntry] = {}
        for e in src:
            key = (getattr(e, row_field), getattr(e, col_field))
            if key not in cell_map:
                cell_map[key] = e
        return rows, cols, cell_map
