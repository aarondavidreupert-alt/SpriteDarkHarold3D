"""
PoseLibrary — stores per-character 3D skeletons, matches poses across
characters via cosine similarity, averages them with confidence weighting,
and detects/replaces outlier landmarks using Mahalanobis distance.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CharacterEntry:
    name: str
    category: str
    skeleton_sequence: np.ndarray        # (N, 33, 3)
    confidences: np.ndarray              # (N, 33)  — per-landmark confidence
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Translate to hip centre origin, scale by shoulder width."""
    hip = (skeleton[23] + skeleton[24]) / 2
    centred = skeleton - hip
    shoulder_w = np.linalg.norm(skeleton[12] - skeleton[11])
    if shoulder_w < 1e-6:
        return centred
    return centred / shoulder_w


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _mahalanobis_distances(samples: np.ndarray) -> np.ndarray:
    """samples: (M, 3).  Returns (M,) distances from the mean."""
    if len(samples) < 2:
        return np.zeros(len(samples))
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    try:
        cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(3))
    except np.linalg.LinAlgError:
        return np.zeros(len(samples))
    diffs = samples - mean
    return np.sqrt(np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs))


class PoseLibrary:
    """
    Stores and manages 3D skeleton data for multiple characters.

    Typical workflow
    ----------------
    lib = PoseLibrary()
    lib.add_character("Vault Dweller", "humanoid", skeleton_seq, confidences)
    lib.add_character("Raider", "humanoid", skeleton_seq2, confidences2)
    master = lib.compute_master_skeleton("humanoid")
    """

    def __init__(self):
        self.characters: List[CharacterEntry] = []

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def add_character(
        self,
        name: str,
        category: str,
        skeleton_sequence: np.ndarray,
        confidences: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> int:
        self.characters.append(CharacterEntry(name, category, skeleton_sequence, confidences, color))
        return len(self.characters) - 1

    def remove_character(self, index: int):
        self.characters.pop(index)

    def get_by_category(self, category: str) -> List[Tuple[int, CharacterEntry]]:
        return [(i, c) for i, c in enumerate(self.characters) if c.category == category]

    # ------------------------------------------------------------------
    # Pose matching
    # ------------------------------------------------------------------

    def match_poses(
        self, idx_a: int, idx_b: int, threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """Return matched (frame_a, frame_b, similarity) pairs above threshold."""
        ca = self.characters[idx_a]
        cb = self.characters[idx_b]
        matches = []
        for fa, skel_a in enumerate(ca.skeleton_sequence):
            norm_a = _normalize_skeleton(skel_a)
            best_score, best_fb = -1.0, -1
            for fb, skel_b in enumerate(cb.skeleton_sequence):
                norm_b = _normalize_skeleton(skel_b)
                s = _cosine_similarity(norm_a, norm_b)
                if s > best_score:
                    best_score, best_fb = s, fb
            if best_score >= threshold:
                matches.append((fa, best_fb, best_score))
        return matches

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------

    def average_poses(
        self, poses: np.ndarray, confidences: np.ndarray,
        outlier_threshold: float = 2.5,
    ) -> np.ndarray:
        """
        Confidence-weighted average of multiple skeleton poses,
        with Mahalanobis-based outlier exclusion per landmark.

        Parameters
        ----------
        poses       : (M, 33, 3)
        confidences : (M, 33)

        Returns
        -------
        (33, 3) averaged skeleton
        """
        M = poses.shape[0]
        w = confidences / (confidences.sum(axis=0, keepdims=True) + 1e-10)  # (M, 33)
        averaged = np.einsum("mi,mij->ij", w, poses)  # (33, 3)

        for lm_idx in range(33):
            samples = poses[:, lm_idx, :]   # (M, 3)
            dists = _mahalanobis_distances(samples)
            outliers = dists > outlier_threshold
            if outliers.any() and (~outliers).sum() >= 1:
                clean = samples[~outliers]
                clean_w = confidences[~outliers, lm_idx]
                clean_w = clean_w / (clean_w.sum() + 1e-10)
                averaged[lm_idx] = np.einsum("m,mj->j", clean_w, clean)

        return averaged

    def compute_master_skeleton(
        self,
        category: str,
        outlier_threshold: float = 2.5,
    ) -> Optional[np.ndarray]:
        """
        Build a per-frame master skeleton by averaging all characters
        of the given category that have matching pose frames.

        Returns (N, 33, 3) or None if no characters match.
        """
        members = self.get_by_category(category)
        if not members:
            return None

        # Use the first character as reference; match all others to it
        ref_idx, ref_char = members[0]
        n_frames = ref_char.skeleton_sequence.shape[0]
        master = []

        for frame_idx in range(n_frames):
            frame_skels, frame_confs = [], []
            frame_skels.append(ref_char.skeleton_sequence[frame_idx])
            frame_confs.append(ref_char.confidences[frame_idx])

            for other_idx, other_char in members[1:]:
                matches = self.match_poses(ref_idx, other_idx)
                # Find best match for this frame
                best = next(
                    (m for m in matches if m[0] == frame_idx),
                    None,
                )
                if best is not None:
                    _, matched_fb, _ = best
                    frame_skels.append(other_char.skeleton_sequence[matched_fb])
                    frame_confs.append(other_char.confidences[matched_fb])

            poses_arr = np.stack(frame_skels)
            confs_arr = np.stack(frame_confs)
            master.append(self.average_poses(poses_arr, confs_arr, outlier_threshold))

        return np.array(master)   # (N, 33, 3)

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def find_outlier_landmarks(
        self, frame_idx: int, category: str, threshold: float = 2.5
    ) -> Dict[int, List[int]]:
        """
        Return {char_idx: [outlier_landmark_indices]} for a given frame.
        """
        members = self.get_by_category(category)
        if not members:
            return {}

        # Collect per-landmark positions across characters
        lm_samples = {}   # lm_idx → list of (char_idx, position)
        for ch_idx, ch in members:
            if frame_idx >= ch.skeleton_sequence.shape[0]:
                continue
            skel = ch.skeleton_sequence[frame_idx]
            for lm_idx, pos in enumerate(skel):
                lm_samples.setdefault(lm_idx, []).append((ch_idx, pos))

        result: Dict[int, List[int]] = {}
        for lm_idx, entries in lm_samples.items():
            if len(entries) < 3:
                continue
            char_ids = [e[0] for e in entries]
            positions = np.array([e[1] for e in entries])
            dists = _mahalanobis_distances(positions)
            for char_id, dist in zip(char_ids, dists):
                if dist > threshold:
                    result.setdefault(char_id, []).append(lm_idx)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "characters": [
                {
                    "name": c.name,
                    "category": c.category,
                    "skeleton_sequence": c.skeleton_sequence.tolist(),
                    "confidences": c.confidences.tolist(),
                    "color": list(c.color),
                }
                for c in self.characters
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.characters = []
        for c in data["characters"]:
            self.characters.append(CharacterEntry(
                name=c["name"],
                category=c["category"],
                skeleton_sequence=np.array(c["skeleton_sequence"]),
                confidences=np.array(c["confidences"]),
                color=tuple(c.get("color", [1.0, 1.0, 1.0])),
            ))
