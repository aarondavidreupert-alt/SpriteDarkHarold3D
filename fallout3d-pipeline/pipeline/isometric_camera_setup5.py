"""
Isometric Camera Setup — 6 views at 60° intervals around the subject.
Provides camera matrices, projection, and triangulation for multi-view reconstruction.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple


class IsometricCameraSetup:
    def __init__(
        self,
        radius: float = 2.0,
        height: float = 1.5,
        focal_length: float = 500,
        image_size: Tuple[int, int] = (400, 400),
        subject_height: float = 0.3,
    ):
        self.radius = radius
        self.height = height
        self.focal_length = focal_length
        self.image_size = image_size
        self.subject_height = subject_height
        self.principal_point = (image_size[0] / 2, image_size[1] / 2)

        self.K = np.array([
            [focal_length, 0, self.principal_point[0]],
            [0, focal_length, self.principal_point[1]],
            [0, 0, 1],
        ])

        self._initialize_cameras()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _initialize_cameras(self):
        self.camera_views: List[Dict] = []
        self.projection_matrices: List[np.ndarray] = []
        for i in range(6):
            cam = self._compute_camera_parameters(i * 60)
            self.camera_views.append(cam)
            self.projection_matrices.append(cam["projection"])

    def _compute_camera_parameters(self, angle: float) -> Dict:
        x = self.radius * np.cos(np.radians(angle))
        y = self.radius * np.sin(np.radians(angle))
        z = self.height
        position = np.array([x, y, z])

        look_at = np.array([0, 0, self.subject_height])

        forward = look_at - position
        forward /= np.linalg.norm(forward)

        right = np.cross(np.array([0, 0, 1]), forward)
        norm = np.linalg.norm(right)
        if norm < 1e-10:
            right = np.array([1, 0, 0])
        else:
            right /= norm

        up = np.cross(forward, right)

        R = np.vstack((right, up, -forward))
        t = -R @ position
        P = self.K @ np.hstack((R, t.reshape(3, 1)))

        return {
            "name": f"ISO-{int(angle // 60) + 1}",
            "angle": angle,
            "position": position,
            "look_at": look_at,
            "rotation": R,
            "translation": t,
            "projection": P,
        }

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_point_to_all_views(self, point: np.ndarray) -> List[Dict]:
        point_h = np.append(point, 1)
        results = []
        for view in self.camera_views:
            p2h = view["projection"] @ point_h
            if abs(p2h[2]) > 1e-10:
                p2 = p2h[:2] / p2h[2]
                valid = 0 <= p2[0] <= self.image_size[0] and 0 <= p2[1] <= self.image_size[1]
            else:
                p2 = np.array([float("nan"), float("nan")])
                valid = False
            results.append({"camera": view["name"], "projection": p2, "valid": valid})
        return results

    def back_project_points(self, points_3d: np.ndarray) -> List[np.ndarray]:
        """Back-project (N,3) 3D points into each of the 6 camera views.

        Returns a list of 6 arrays shaped (N, 3) — [px, py, depth].
        """
        results = []
        for view in self.camera_views:
            P = view["projection"]
            cam_pos = view["position"]
            projected = []
            for pt in points_3d:
                if np.all(pt == 0):
                    projected.append(np.zeros(3))
                    continue
                ph = np.append(pt, 1)
                p2h = P @ ph
                if abs(p2h[2]) > 1e-10:
                    p2 = p2h[:2] / p2h[2]
                    depth = np.linalg.norm(pt - cam_pos) / self.radius
                    projected.append(np.array([p2[0], p2[1], depth]))
                else:
                    projected.append(np.zeros(3))
            results.append(np.array(projected))
        return results

    def get_camera_parameters(self, index: int) -> Dict:
        if not 0 <= index < 6:
            raise ValueError(f"Camera index must be 0-5, got {index}")
        return self.camera_views[index]

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    def triangulate_point(
        self,
        image_points: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Weighted DLT triangulation from named camera 2D observations."""
        A, W = [], []
        for view in self.camera_views:
            name = view["name"]
            if name not in image_points:
                continue
            x, y = image_points[name][:2]
            P = view["projection"]
            w = weights.get(name, 1.0) if weights else 1.0
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
            W.extend([w, w])

        if not A:
            return np.zeros(3)

        A = np.array(A)
        W_mat = np.diag(W)
        _, _, Vh = np.linalg.svd(W_mat @ A)
        X = Vh[-1]
        if abs(X[3]) < 1e-10:
            return np.zeros(3)
        return (X / X[3])[:3]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_configuration(self, path: str):
        config = {
            "parameters": {
                "radius": self.radius,
                "height": self.height,
                "focal_length": self.focal_length,
                "image_size": list(self.image_size),
                "subject_height": self.subject_height,
            },
            "cameras": [
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cam.items()}
                for cam in self.camera_views
            ],
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_configuration(cls, path: str) -> "IsometricCameraSetup":
        with open(path) as f:
            config = json.load(f)
        params = config["parameters"]
        params["image_size"] = tuple(params["image_size"])
        return cls(**params)
