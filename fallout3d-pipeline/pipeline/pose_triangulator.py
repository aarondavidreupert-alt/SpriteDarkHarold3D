"""
PoseTriangulator — MediaPipe pose detection across 6 isometric views,
with flip correction, per-landmark confidence weighting, and triangulation.
"""

import os
import math
import numpy as np
import cv2

from .isometric_camera_setup5 import IsometricCameraSetup

# MediaPipe is optional at import time so the pipeline can load without a GPU.
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# -----------------------------------------------------------------------
# Model paths (Tasks API)
# -----------------------------------------------------------------------

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.dirname(os.path.dirname(_PIPELINE_DIR))
_MODEL_DIR    = os.path.join(_REPO_ROOT, "models")
_MODEL_PATH   = os.path.join(_MODEL_DIR, "pose_landmarker_heavy.task")
_MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)


# -----------------------------------------------------------------------
# Skeleton connectivity
# -----------------------------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (23, 25), (25, 27), (24, 26), (26, 28),   # legs
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),           # face
]

LANDMARK_BODY_PART = {**{i: "face" for i in range(11)},
                      **{i: "torso" for i in [11, 12, 23, 24]},
                      **{i: "arms" for i in [13, 14, 15, 16]},
                      **{i: "legs" for i in [25, 26, 27, 28]}}

PART_COLORS = {
    "face":  (255, 0, 255),
    "torso": (0, 255, 0),
    "arms":  (0, 0, 255),
    "legs":  (255, 255, 0),
}

# Pairs that get swapped on a left/right flip — kept for backward compat
_FLIP_PAIRS = [(11, 12), (13, 14), (15, 16), (23, 24),
               (7, 8), (9, 10), (1, 4), (2, 5), (3, 6)]

# Per-direction expected pixel-space vector from right_shoulder (LM12) → left_shoulder (LM11).
# x: rightward, y: downward. Derived from Fallout 2 isometric camera geometry.
DIR_EXPECTED_RL = [
    np.array([-1.0,  0.0]),  # Dir 1 (idx 0) NE — back to camera
    np.array([ 0.0, -1.0]),  # Dir 2 (idx 1) E  — right profile
    np.array([ 1.0,  0.0]),  # Dir 3 (idx 2) SE — facing camera
    np.array([ 1.0,  0.0]),  # Dir 4 (idx 3) SW — facing camera
    np.array([ 0.0,  1.0]),  # Dir 5 (idx 4) W  — left profile
    np.array([-1.0,  0.0]),  # Dir 6 (idx 5) NW — back to camera
]

_FLIP_PAIRS_UPPER = [(11,12),(13,14),(15,16),(7,8),(9,10),(1,4),(2,5),(3,6)]
_FLIP_PAIRS_LOWER = [(23,24),(25,26),(27,28),(29,30),(31,32)]

# Anatomical side per MediaPipe 33-landmark schema
LANDMARK_SIDE: dict = {}
for _i in [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
    LANDMARK_SIDE[_i] = "left"
for _i in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
    LANDMARK_SIDE[_i] = "right"
for _i in range(11):  # 0-10 face/mid-body
    LANDMARK_SIDE[_i] = "center"

_SIDE_DOT_COLORS = {
    "left":   (220,  80,  80),  # red
    "right":  ( 80,  80, 220),  # blue
    "center": (200, 200, 200),  # grey
}


def _normalized_to_pixel(nx, ny, w, h):
    px = min(math.floor(nx * w), w - 1)
    py = min(math.floor(ny * h), h - 1)
    return max(0, px), max(0, py)


class PoseTriangulator:
    """
    Detects 2D MediaPipe poses in each of the 6 isometric views,
    corrects left/right flips per perspective, and triangulates to 3D.

    Usage
    -----
    t = PoseTriangulator()
    t.load_animation_sequence(numpy_array)   # shape (6, N, H, W, 3)
    t.detect_poses_sequence()
    skeleton_3d = t.triangulate_sequence()   # shape (N, 33, 3)
    """

    def __init__(self, image_size=(400, 400)):
        self.camera_setup = IsometricCameraSetup(
            radius=2.0, height=1.5, focal_length=500,
            image_size=image_size, subject_height=0.3,
        )
        self.frames: np.ndarray | None = None          # (6, N, H, W, 3)
        self.poses_sequence: np.ndarray | None = None  # (N, 6, 33, 3)
        self.pad_pixels: int = 40   # black border added before MediaPipe detection

        self._pose_detector = None  # lazy-initialised

    # ------------------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------------------

    def _get_detector(self):
        if self._pose_detector is None:
            if not _MP_AVAILABLE:
                raise RuntimeError("mediapipe is not installed.")

            if not os.path.exists(_MODEL_PATH):
                import urllib.request
                os.makedirs(_MODEL_DIR, exist_ok=True)
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

            from mediapipe.tasks import python as _mptasks
            from mediapipe.tasks.python import vision as _mpvision

            options = _mpvision.PoseLandmarkerOptions(
                base_options=_mptasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=_mpvision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_detector = _mpvision.PoseLandmarker.create_from_options(options)
        return self._pose_detector

    # ------------------------------------------------------------------
    # Padding helper
    # ------------------------------------------------------------------

    def _pad_frame(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Add black border so MediaPipe has context pixels around tight crops.

        Returns (padded_image, pad_top, pad_left) so callers can subtract the
        offset from detected coordinates to get back into original image space.
        """
        p = self.pad_pixels
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        return padded, p, p

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_animation_sequence(self, array: np.ndarray):
        """Accept array of shape (6, N, H, W, 3) or (6, N, H, W)."""
        self.frames = array
        self.poses_sequence = None

    # ------------------------------------------------------------------
    # Flip correction
    # ------------------------------------------------------------------

    def _correct_flip(self, pose_2d: np.ndarray, perspective_idx: int) -> np.ndarray:
        """
        Independently correct left/right flip for upper and lower body.
        Uses direction-specific expected shoulder/hip vectors for Fallout 2
        isometric perspective. Upper and lower body are checked separately
        because mid-stride poses can have correct shoulders but flipped hips.
        """
        expected  = DIR_EXPECTED_RL[perspective_idx % 6]
        corrected = pose_2d.copy()

        # --- Upper body ---
        left_sh, right_sh = corrected[11], corrected[12]
        if not (np.all(left_sh == 0) or np.all(right_sh == 0)):
            vec = left_sh[:2] - right_sh[:2]   # right → left
            if np.dot(vec, expected) < 0:
                for a, b in _FLIP_PAIRS_UPPER:
                    corrected[a], corrected[b] = corrected[b].copy(), corrected[a].copy()

        # --- Lower body (independent check) ---
        left_hip, right_hip = corrected[23], corrected[24]
        if not (np.all(left_hip == 0) or np.all(right_hip == 0)):
            vec = left_hip[:2] - right_hip[:2]  # right → left
            if np.dot(vec, expected) < 0:
                for a, b in _FLIP_PAIRS_LOWER:
                    corrected[a], corrected[b] = corrected[b].copy(), corrected[a].copy()

        return corrected

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_poses_sequence(self, progress_cb=None):
        """Run MediaPipe on every (frame, perspective) pair.

        Parameters
        ----------
        progress_cb : callable(int, int) | None
            Called with (current_frame, total_frames) for progress reporting.
        """
        detector = self._get_detector()
        n_perspectives, n_frames = self.frames.shape[0], self.frames.shape[1]
        all_poses = []

        for frame_idx in range(n_frames):
            if progress_cb:
                progress_cb(frame_idx, n_frames)

            frame_poses = []
            for persp_idx in range(n_perspectives):
                img = self.frames[persp_idx, frame_idx]
                h, w = img.shape[:2]

                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]

                img_padded, pad_top, pad_left = self._pad_frame(img)
                h_pad, w_pad = img_padded.shape[:2]

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(img_padded),
                )
                result = detector.detect(mp_image)

                if result.pose_landmarks:
                    lms = np.array([
                        [*_normalized_to_pixel(lm.x, lm.y, w_pad, h_pad), lm.z]
                        for lm in result.pose_landmarks[0]
                    ], dtype=float)
                    # Subtract padding to restore original image coordinates
                    lms[:, 0] = np.clip(lms[:, 0] - pad_left, 0, w - 1)
                    lms[:, 1] = np.clip(lms[:, 1] - pad_top,  0, h - 1)
                    lms = self._correct_flip(lms, persp_idx)
                else:
                    lms = np.zeros((33, 3))

                frame_poses.append(lms)

            all_poses.append(np.array(frame_poses))  # (6, 33, 3)

        self.poses_sequence = np.array(all_poses)     # (N, 6, 33, 3)
        if progress_cb:
            progress_cb(n_frames, n_frames)

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    def triangulate_sequence(self) -> np.ndarray:
        """Triangulate each landmark across all views for every frame.

        Returns
        -------
        np.ndarray of shape (N, 33, 3)
        """
        if self.poses_sequence is None:
            raise ValueError("Run detect_poses_sequence() first.")

        out = []
        for frame_poses in self.poses_sequence:   # (6, 33, 3)
            frame_3d = []
            for lm_idx in range(33):
                views, weights = {}, {}
                for v_idx, pose in enumerate(frame_poses):
                    lm = pose[lm_idx]
                    if not np.all(lm == 0):
                        views[f"ISO-{v_idx + 1}"] = lm[:2]
                        weights[f"ISO-{v_idx + 1}"] = max(0.05, float(lm[2]))
                if len(views) >= 2:
                    pt3d = self.camera_setup.triangulate_point(views, weights)
                else:
                    pt3d = np.zeros(3)
                frame_3d.append(pt3d)
            out.append(np.array(frame_3d))

        return np.array(out)    # (N, 33, 3)

    # ------------------------------------------------------------------
    # Manual landmark correction
    # ------------------------------------------------------------------

    def set_landmark(self, frame_idx: int, view_idx: int, lm_idx: int, xy: np.ndarray):
        """Override a single 2D landmark position (in pixel coords)."""
        if self.poses_sequence is None:
            return
        self.poses_sequence[frame_idx, view_idx, lm_idx, :2] = xy

    def get_backprojection_error(self, frame_idx: int, skeleton_3d: np.ndarray) -> np.ndarray:
        """Return per-landmark reprojection error (mean over all views), shape (33,)."""
        bp_views = self.camera_setup.back_project_points(skeleton_3d)  # list of 6 × (33, 3)
        errors = np.zeros(33)
        for v_idx in range(6):
            orig = self.poses_sequence[frame_idx, v_idx]   # (33, 3)
            bp = bp_views[v_idx]                            # (33, 3)
            for lm_idx in range(33):
                if not np.all(orig[lm_idx] == 0):
                    errors[lm_idx] += np.linalg.norm(orig[lm_idx, :2] - bp[lm_idx, :2])
        return errors / 6

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def overlay_pose_on_image(
        self, image: np.ndarray, pose_2d: np.ndarray,
        frame_idx: int = 0, perspective_idx: int = 0
    ) -> np.ndarray:
        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        connections = [
            (11, 12, "torso"), (12, 24, "torso"), (24, 23, "torso"), (23, 11, "torso"),
            (11, 13, "arms"), (13, 15, "arms"), (12, 14, "arms"), (14, 16, "arms"),
            (23, 25, "legs"), (25, 27, "legs"), (24, 26, "legs"), (26, 28, "legs"),
        ]
        for s, e, part in connections:
            if not (np.all(pose_2d[s] == 0) or np.all(pose_2d[e] == 0)):
                cv2.line(img, tuple(pose_2d[s, :2].astype(int)),
                         tuple(pose_2d[e, :2].astype(int)), PART_COLORS[part], 2)
        for i, pt in enumerate(pose_2d):
            if not np.all(pt == 0):
                dot_color = _SIDE_DOT_COLORS[LANDMARK_SIDE.get(i, "center")]
                cv2.circle(img, tuple(pt[:2].astype(int)), 3, dot_color, -1)

        cv2.putText(img, f"F{frame_idx} V{perspective_idx + 1}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    # ------------------------------------------------------------------
    # Bidirectional detection
    # ------------------------------------------------------------------

    def detect_poses_bidirectional(self, progress_cb=None):
        """Run detection forward + backward through frames and merge results.

        Frames where only one pass detects a pose use that result;
        frames where both passes detect average them for stability.
        """
        if self.frames is None:
            raise ValueError("Load frames with load_animation_sequence() first.")
        n_frames = self.frames.shape[1]

        def _fwd_cb(f, t):
            if progress_cb:
                progress_cb(f, n_frames * 2)

        self.detect_poses_sequence(progress_cb=_fwd_cb)
        fwd = self.poses_sequence.copy()

        self.frames = self.frames[:, ::-1].copy()

        def _bwd_cb(f, t):
            if progress_cb:
                progress_cb(n_frames + f, n_frames * 2)

        self.detect_poses_sequence(progress_cb=_bwd_cb)
        bwd = self.poses_sequence[::-1].copy()
        self.frames = self.frames[:, ::-1].copy()  # restore original order

        fwd_zero = np.all(fwd == 0, axis=-1, keepdims=True)
        bwd_zero = np.all(bwd == 0, axis=-1, keepdims=True)
        self.poses_sequence = np.where(
            ~fwd_zero & ~bwd_zero,
            (fwd + bwd) / 2,
            np.where(~fwd_zero, fwd, bwd),
        )

        if progress_cb:
            progress_cb(n_frames * 2, n_frames * 2)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_animation_data(self, skeleton_3d: np.ndarray, path: str = "animation_data.json"):
        import json
        data = {
            "metadata": {
                "total_frames": int(skeleton_3d.shape[0]),
                "perspectives": 6,
                "image_size": list(self.camera_setup.image_size),
                "landmarks_per_pose": 33,
            },
            "poses_2d": self.poses_sequence.tolist() if self.poses_sequence is not None else [],
            "poses_3d": skeleton_3d.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
