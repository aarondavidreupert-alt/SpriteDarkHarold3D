"""
Sausage Library — collection of VoxelCarver snapshots that can be
weight-averaged into a "master" voxel grid per bone.

Workflow:
  1. Build a VoxelCarver for a (character, animation) pair.
  2. SausageEntry.from_carver(carver, character, animation, weight)
  3. library.add(entry)  ⋯  library.add(entry2)  ⋯
  4. master_voxels = library.build_master(filter_character="HFPRIME")
"""

from __future__ import annotations

import os
import logging
import numpy as np
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SausageEntry
# ---------------------------------------------------------------------------

class SausageEntry:
    """Snapshot of a VoxelCarver: one (character, animation, weight) sample."""

    def __init__(self):
        self.character: str = ""
        self.animation: str = ""
        self.weight: float = 1.0
        self.bone_radii:   Dict[int, float]      = {}
        self.voxels:       Dict[int, np.ndarray] = {}
        self.grid_origins: Dict[int, np.ndarray] = {}
        self.voxel_sizes:  Dict[int, float]      = {}
        self.bone_lens:    Dict[int, float]      = {}

    # ------------------------------------------------------------------

    @classmethod
    def from_carver(cls, carver, character: str, animation: str,
                    weight: float = 1.0) -> "SausageEntry":
        """Snapshot a VoxelCarver into a library entry."""
        entry = cls()
        entry.character = str(character)
        entry.animation = str(animation)
        entry.weight    = float(weight)
        for jidx, s in carver.sausages.items():
            if s.voxels is None:
                continue
            entry.bone_radii[jidx]    = float(s.radius)
            entry.voxels[jidx]        = s.voxels.copy()
            entry.grid_origins[jidx]  = s.grid_origin.copy()
            entry.voxel_sizes[jidx]   = float(s.voxel_size)
            entry.bone_lens[jidx]     = float(s._bone_len_f0)
        return entry

    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save as a single .npz."""
        data: Dict[str, np.ndarray] = {
            "character": np.array(self.character),
            "animation": np.array(self.animation),
            "weight":    np.array(self.weight, dtype=np.float32),
        }
        joint_ids = np.array(sorted(self.voxels.keys()), dtype=np.int32)
        data["joint_ids"] = joint_ids
        for jidx in joint_ids:
            p = f"j{int(jidx)}_"
            data[p + "voxels"]      = self.voxels[int(jidx)]
            data[p + "grid_origin"] = self.grid_origins[int(jidx)]
            data[p + "voxel_size"]  = np.array(self.voxel_sizes[int(jidx)], dtype=np.float64)
            data[p + "bone_len"]    = np.array(self.bone_lens[int(jidx)], dtype=np.float64)
            data[p + "radius"]      = np.array(self.bone_radii[int(jidx)], dtype=np.float64)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "SausageEntry":
        """Inverse of save()."""
        d = np.load(path, allow_pickle=False)
        entry = cls()
        entry.character = str(d["character"].item()) if "character" in d.files else ""
        entry.animation = str(d["animation"].item()) if "animation" in d.files else ""
        entry.weight    = float(d["weight"]) if "weight" in d.files else 1.0
        joint_ids = d["joint_ids"]
        for jidx in joint_ids:
            p = f"j{int(jidx)}_"
            entry.voxels[int(jidx)]       = d[p + "voxels"].astype(bool)
            entry.grid_origins[int(jidx)] = d[p + "grid_origin"].copy()
            entry.voxel_sizes[int(jidx)]  = float(d[p + "voxel_size"])
            entry.bone_lens[int(jidx)]    = float(d[p + "bone_len"])
            entry.bone_radii[int(jidx)]   = float(d[p + "radius"])
        return entry

    # ------------------------------------------------------------------

    @property
    def n_bones(self) -> int:
        return len(self.voxels)


# ---------------------------------------------------------------------------
# SausageLibrary
# ---------------------------------------------------------------------------

class SausageLibrary:
    """Collection of SausageEntry, with weighted averaging into a master grid."""

    def __init__(self):
        self.entries: List[SausageEntry] = []

    def add(self, entry: SausageEntry):
        self.entries.append(entry)

    def remove(self, idx: int):
        if 0 <= idx < len(self.entries):
            self.entries.pop(idx)

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------

    def build_master(
        self,
        filter_character: Optional[str] = None,
        resolution: int = 32,
        threshold: float = 0.3,
    ) -> Dict[int, np.ndarray]:
        """
        Weighted-average voxel grids into a single master grid per bone.

        For each joint_idx:
          1. Collect all entries that have this joint
          2. Resample each grid to (resolution, resolution, resolution) via zoom
          3. Weighted sum: Σ weight * resampled
          4. Normalise: / Σ weight
          5. Threshold: density >= threshold → True

        Returns: {joint_idx → (R, R, R) bool}
        """
        try:
            from scipy.ndimage import zoom
        except ImportError as exc:
            raise ImportError("scipy required: pip install scipy") from exc

        entries = [e for e in self.entries
                   if filter_character is None or e.character == filter_character]
        if not entries:
            return {}

        all_joints: set = set()
        for e in entries:
            all_joints.update(e.voxels.keys())

        master: Dict[int, np.ndarray] = {}
        for jidx in sorted(all_joints):
            relevant = [(e.weight, e.voxels[jidx])
                        for e in entries if jidx in e.voxels]
            if not relevant:
                continue
            acc = np.zeros((resolution, resolution, resolution), dtype=np.float32)
            total_w = 0.0
            for w, vox in relevant:
                src = vox.astype(np.float32)
                if src.shape[0] == 0:
                    continue
                factor = resolution / src.shape[0]
                resampled = zoom(src, factor, order=1)
                resampled = resampled[:resolution, :resolution, :resolution]
                pad = [(0, max(0, resolution - resampled.shape[i])) for i in range(3)]
                resampled = np.pad(resampled, pad)
                acc += w * resampled
                total_w += w
            if total_w <= 0.0:
                continue
            master[jidx] = (acc / total_w) >= threshold
        return master

    # ------------------------------------------------------------------

    def save_library(self, dir_path: str):
        """Save all entries as individual .npz files in dir_path."""
        os.makedirs(dir_path, exist_ok=True)
        for i, e in enumerate(self.entries):
            fname = f"{i:03d}_{e.character}_{e.animation}.npz"
            # sanitise
            fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)
            e.save(os.path.join(dir_path, fname))

    def load_library(self, dir_path: str):
        """Load all .npz files in dir_path, append to self.entries."""
        if not os.path.isdir(dir_path):
            return
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".npz"):
                continue
            try:
                self.entries.append(SausageEntry.load(os.path.join(dir_path, fname)))
            except Exception as exc:
                _logger.warning("Failed to load %s: %s", fname, exc)
