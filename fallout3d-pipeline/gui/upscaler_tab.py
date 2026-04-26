"""
Tab 2 — Upscaler
Applies Real-ESRGAN upscaling to the currently loaded character's frames.
Upscaled result is stored in CharacterData.upscaled_frames for downstream tabs.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QProgressBar, QSpinBox,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState


# -----------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------

MODELS = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "num_block": 6,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus_anime_6B"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _rgb_to_pixmap(rgb: np.ndarray, max_side: int = 320) -> QPixmap:
    """RGB (H, W, 3) uint8 → QPixmap, scaled so the longest side ≤ max_side."""
    h, w = rgb.shape[:2]
    data = rgb.tobytes()
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix  = QPixmap.fromImage(qimg)
    if max(h, w) > max_side:
        pix = pix.scaled(
            max_side, max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


# -----------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------

class UpscaleWorker(QObject):
    progress = pyqtSignal(int, int, str)   # done, total, message
    finished = pyqtSignal(object)          # np.ndarray (6, N, H', W', 3)
    error    = pyqtSignal(str)

    def __init__(self, frames: np.ndarray, model_name: str):
        super().__init__()
        self.frames     = frames
        self.model_name = model_name

    def run(self):
        # Deferred import so a missing package only fails here, not at app start
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as exc:
            self.error.emit(
                f"Real-ESRGAN not installed.\n"
                f"Run:  pip install realesrgan\n\nDetail: {exc}"
            )
            return

        try:
            cfg   = MODELS[self.model_name]
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"],
            )
            upsampler = RealESRGANer(
                scale=cfg["scale"],
                model_path=cfg["url"],
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )

            n_dirs, n_frames = self.frames.shape[:2]
            scale = cfg["scale"]
            total = n_dirs * n_frames
            done  = 0

            # Run the first frame to determine actual output dimensions
            # (RealESRGAN output may differ from h*scale by ±1 in rare cases)
            first_bgr = self.frames[0, 0][..., ::-1].copy()
            first_out, _ = upsampler.enhance(first_bgr, outscale=scale)
            out_h, out_w  = first_out.shape[:2]

            result      = np.zeros((n_dirs, n_frames, out_h, out_w, 3), dtype=np.uint8)
            result[0, 0] = first_out[..., ::-1]          # BGR → RGB
            done += 1
            self.progress.emit(done, total, f"Dir 1/{n_dirs}  Frame 1/{n_frames}")

            for d in range(n_dirs):
                for fi in range(n_frames):
                    if d == 0 and fi == 0:
                        continue
                    bgr     = self.frames[d, fi][..., ::-1].copy()
                    out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
                    out_rgb = out_bgr[..., ::-1]
                    # Guard: resize if output differs (shouldn't happen for uniform input)
                    if out_rgb.shape[:2] != (out_h, out_w):
                        from PIL import Image as _PIL
                        out_rgb = np.array(
                            _PIL.fromarray(out_rgb).resize((out_w, out_h), _PIL.LANCZOS)
                        )
                    result[d, fi] = out_rgb
                    done += 1
                    self.progress.emit(
                        done, total,
                        f"Dir {d + 1}/{n_dirs}  Frame {fi + 1}/{n_frames}",
                    )

            self.finished.emit(result)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class UpscalerTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state     = state
        self._thread: QThread | None = None
        self._upscaled: np.ndarray | None = None
        self._build_ui()
        self.state.selection_changed.connect(self._on_selection_changed)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Controls ─────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Model & Controls")
        ctrl_lay = QVBoxLayout(ctrl_box)

        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))

        self._model_combo = QComboBox()
        for name in MODELS:
            self._model_combo.addItem(name, name)
        self._model_combo.setCurrentText(DEFAULT_MODEL)
        top.addWidget(self._model_combo)

        self._btn_run = QPushButton("Run Upscaler")
        self._btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_run.clicked.connect(self._run)
        top.addWidget(self._btn_run)

        top.addStretch()
        self._status_lbl = QLabel("Load a character in the Asset Loader tab first.")
        top.addWidget(self._status_lbl)
        ctrl_lay.addLayout(top)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        ctrl_lay.addWidget(self._progress)

        root.addWidget(ctrl_box)

        # ── Preview ───────────────────────────────────────────────────────
        prev_box = QGroupBox("Preview")
        prev_lay = QVBoxLayout(prev_box)

        nav = QHBoxLayout()
        nav.addWidget(QLabel("Direction:"))
        self._dir_spin = QSpinBox()
        self._dir_spin.setRange(1, 6)
        self._dir_spin.setFixedWidth(56)
        self._dir_spin.valueChanged.connect(self._update_preview)
        nav.addWidget(self._dir_spin)

        nav.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(1, 1)
        self._frame_spin.setFixedWidth(56)
        self._frame_spin.valueChanged.connect(self._update_preview)
        nav.addWidget(self._frame_spin)

        self._nav_info = QLabel("")
        nav.addWidget(self._nav_info)
        nav.addStretch()
        prev_lay.addLayout(nav)

        images = QHBoxLayout()

        orig_box = QGroupBox("Original")
        orig_lay = QVBoxLayout(orig_box)
        self._orig_lbl = QLabel()
        self._orig_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_lbl.setMinimumSize(200, 200)
        self._orig_lbl.setStyleSheet("background:#111;")
        self._orig_dim = QLabel("")
        self._orig_dim.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_lay.addWidget(self._orig_lbl)
        orig_lay.addWidget(self._orig_dim)
        images.addWidget(orig_box, 1)

        up_box = QGroupBox("Upscaled")
        up_lay = QVBoxLayout(up_box)
        self._up_lbl = QLabel()
        self._up_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._up_lbl.setMinimumSize(200, 200)
        self._up_lbl.setStyleSheet("background:#111;")
        self._up_dim = QLabel("")
        self._up_dim.setAlignment(Qt.AlignmentFlag.AlignCenter)
        up_lay.addWidget(self._up_lbl)
        up_lay.addWidget(self._up_dim)
        images.addWidget(up_box, 1)

        prev_lay.addLayout(images)
        root.addWidget(prev_box, 1)

    # ── Upscaling ─────────────────────────────────────────────────────────

    def _run(self):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return

        model_name = self._model_combo.currentData()
        total      = char.frames.shape[0] * char.frames.shape[1]

        self._btn_run.setEnabled(False)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status_lbl.setText("Loading model…")

        self._worker = UpscaleWorker(char.frames, model_name)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, done: int, total: int, msg: str):
        self._progress.setValue(done)
        self._status_lbl.setText(msg)

    def _on_finished(self, result: np.ndarray):
        self._thread.quit()
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self._upscaled = result

        char = self.state.current_character
        if char is not None:
            char.upscaled_frames = result
            self.state.character_upscaled.emit(self.state.selected_idx)

        n_dirs, n_frames, out_h, out_w = result.shape[:4]
        self._frame_spin.setRange(1, n_frames)
        self._nav_info.setText(f"({n_dirs} dirs × {n_frames} frames)")
        self._status_lbl.setText(
            f"Done — upscaled to {out_w}×{out_h} px"
            f"  ({n_dirs} dirs × {n_frames} frames)"
        )
        self._update_preview()

    def _on_error(self, msg: str):
        self._thread.quit()
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        # Show first 200 chars to avoid flooding the label
        self._status_lbl.setText(f"Error: {msg[:200]}")

    # ── Selection / preview ───────────────────────────────────────────────

    def _on_selection_changed(self, _idx: int):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return

        n_dirs, n_frames = char.frames.shape[0], char.frames.shape[1]
        self._frame_spin.setRange(1, n_frames)
        self._nav_info.setText(f"({n_dirs} dirs × {n_frames} frames)")
        self._status_lbl.setText(
            f"Ready — '{char.name}'  {n_dirs}d × {n_frames}f  "
            f"{char.frames.shape[3]}×{char.frames.shape[2]} px"
        )
        self._upscaled = getattr(char, "upscaled_frames", None)
        self._update_preview()

    def _update_preview(self):
        char = self.state.current_character
        if char is None:
            return

        d  = max(0, min(self._dir_spin.value() - 1,  char.frames.shape[0] - 1))
        fi = max(0, min(self._frame_spin.value() - 1, char.frames.shape[1] - 1))

        orig = char.frames[d, fi]                    # (H, W, 3)
        oh, ow = orig.shape[:2]
        self._orig_lbl.setPixmap(_rgb_to_pixmap(orig))
        self._orig_dim.setText(f"{ow}×{oh} px")

        if self._upscaled is not None:
            up = self._upscaled[d, fi]
            uh, uw = up.shape[:2]
            self._up_lbl.setPixmap(_rgb_to_pixmap(up))
            self._up_dim.setText(f"{uw}×{uh} px")
        else:
            self._up_lbl.clear()
            self._up_dim.setText("(not yet upscaled)")
