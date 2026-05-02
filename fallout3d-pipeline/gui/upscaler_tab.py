"""
Tab 2 — Upscaler
Supports three backends selectable at runtime:
  • EDSR (OpenCV DNN)  — fast, deterministic, requires EDSR_x4.pb
  • Real-ESRGAN        — high-quality neural upscaler
  • Custom PyTorch     — passthrough for SwinIR / SPAN / HAT / etc.

Upscaled result is stored in CharacterData.upscaled_frames for downstream tabs.
Cache: <source_dir>/<char_name>_upscaled_<backend>.npy — loaded automatically
       if present; saved after each successful run.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QProgressBar, QSpinBox, QLineEdit,
    QFileDialog, QStackedWidget,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState

# Pull model registry from the shared module so it stays in one place
from upscaler import REALESRGAN_MODELS, REALESRGAN_DEFAULT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_to_pixmap(rgb: np.ndarray, max_side: int = 320) -> QPixmap:
    """RGB (H, W, 3) uint8 → QPixmap scaled to at most max_side pixels."""
    h, w  = rgb.shape[:2]
    data  = rgb.tobytes()
    qimg  = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix   = QPixmap.fromImage(qimg)
    if max(h, w) > max_side:
        pix = pix.scaled(
            max_side, max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


def _cache_path(char, backend: str) -> str:
    """Derive a .npy cache path next to the source file (or in cwd)."""
    base_dir = (
        os.path.dirname(char.source_path)
        if char.source_path
        else os.getcwd()
    )
    return os.path.join(base_dir, f"{char.name}_upscaled_{backend}.npy")


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class UpscaleWorker(QObject):
    progress = pyqtSignal(int, int, str)   # done, total, message
    finished = pyqtSignal(object)          # np.ndarray (6, N, H', W', 3)
    error    = pyqtSignal(str)

    def __init__(
        self,
        frames: np.ndarray,
        backend: str,
        *,
        model_path: str = "",
        model_name: str = REALESRGAN_DEFAULT,
        cache_path: str = "",
    ):
        super().__init__()
        self.frames     = frames
        self.backend    = backend
        self.model_path = model_path
        self.model_name = model_name
        self.cache_path = cache_path

    def run(self):
        # ── Cache hit ────────────────────────────────────────────────────
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                result = np.load(self.cache_path)
                self.progress.emit(1, 1, f"Loaded from cache: {os.path.basename(self.cache_path)}")
                self.finished.emit(result)
                return
            except Exception as exc:
                self.progress.emit(0, 1, f"Cache load failed ({exc}), re-running…")

        # ── Run upscaler ─────────────────────────────────────────────────
        try:
            from upscaler import upscale_sequence
        except ImportError as exc:
            self.error.emit(f"upscaler module not found: {exc}")
            return

        try:
            result = upscale_sequence(
                self.frames,
                self.backend,
                model_path=self.model_path,
                model_name=self.model_name,
                progress_cb=lambda d, t, m: self.progress.emit(d, t, m),
            )
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
            return

        # ── Save cache ───────────────────────────────────────────────────
        if self.cache_path:
            try:
                np.save(self.cache_path, result)
            except Exception:
                pass  # non-fatal

        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class UpscalerTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state     = state
        self._thread: QThread | None = None
        self._upscaled: np.ndarray | None = None
        self._build_ui()
        self.state.selection_changed.connect(self._on_selection_changed)
        self.state.character_updated.connect(self._on_selection_changed)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Backend & Controls")
        ctrl_lay = QVBoxLayout(ctrl_box)

        # Row 1 — backend + model + run
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItem("EDSR (OpenCV DNN)",  "edsr")
        self._backend_combo.addItem("Real-ESRGAN",         "realesrgan")
        self._backend_combo.addItem("Custom PyTorch",      "torch")
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row1.addWidget(self._backend_combo)

        # Stacked widget: one row per backend's extra options
        self._opt_stack = QStackedWidget()

        # Page 0 — EDSR: model path + browse button
        edsr_page = QWidget()
        edsr_lay  = QHBoxLayout(edsr_page)
        edsr_lay.setContentsMargins(0, 0, 0, 0)
        edsr_lay.addWidget(QLabel("Model (.pb):"))
        self._edsr_path = QLineEdit()
        self._edsr_path.setPlaceholderText("Path to EDSR_x4.pb …")
        edsr_lay.addWidget(self._edsr_path, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(72)
        btn_browse.clicked.connect(self._browse_edsr_model)
        edsr_lay.addWidget(btn_browse)
        self._opt_stack.addWidget(edsr_page)

        # Page 1 — Real-ESRGAN: model variant picker
        esrgan_page = QWidget()
        esrgan_lay  = QHBoxLayout(esrgan_page)
        esrgan_lay.setContentsMargins(0, 0, 0, 0)
        esrgan_lay.addWidget(QLabel("Model:"))
        self._esrgan_combo = QComboBox()
        for name in REALESRGAN_MODELS:
            self._esrgan_combo.addItem(name, name)
        self._esrgan_combo.setCurrentText(REALESRGAN_DEFAULT)
        esrgan_lay.addWidget(self._esrgan_combo, 1)
        self._opt_stack.addWidget(esrgan_page)

        # Page 2 — PyTorch: informational label only
        torch_page = QWidget()
        torch_lay  = QHBoxLayout(torch_page)
        torch_lay.setContentsMargins(0, 0, 0, 0)
        torch_lay.addWidget(QLabel(
            "Set torch_model in UpscaleWorker before running."
        ))
        self._opt_stack.addWidget(torch_page)

        row1.addWidget(self._opt_stack, 2)

        self._btn_run = QPushButton("Run Upscaler")
        self._btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_run.clicked.connect(self._run)
        row1.addWidget(self._btn_run)

        row1.addStretch()
        ctrl_lay.addLayout(row1)

        # Row 2 — status + progress
        self._status_lbl = QLabel("Load a character in the Asset Loader tab first.")
        ctrl_lay.addWidget(self._status_lbl)

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

    # ── Backend switching ─────────────────────────────────────────────────

    def _on_backend_changed(self, idx: int):
        self._opt_stack.setCurrentIndex(idx)

    def _browse_edsr_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select EDSR Model", "",
            "PB Model (*.pb);;All Files (*)",
        )
        if path:
            self._edsr_path.setText(path)

    # ── Upscaling ─────────────────────────────────────────────────────────

    def _run(self):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return

        backend    = self._backend_combo.currentData()
        model_path = self._edsr_path.text().strip()
        model_name = self._esrgan_combo.currentData()
        cache_fp   = _cache_path(char, backend)
        total      = char.frames.shape[0] * char.frames.shape[1]

        self._btn_run.setEnabled(False)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status_lbl.setText("Starting…")

        self._worker = UpscaleWorker(
            char.frames, backend,
            model_path=model_path,
            model_name=model_name,
            cache_path=cache_fp,
        )
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
            f"Done — {out_w}×{out_h} px  ({n_dirs} dirs × {n_frames} frames)"
        )
        self._update_preview()

    def _on_error(self, msg: str):
        self._thread.quit()
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
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

        # Auto-detect an existing cache for whichever backend is selected
        backend  = self._backend_combo.currentData()
        cache_fp = _cache_path(char, backend)
        if char.upscaled_frames is not None:
            status = "upscaled frames already loaded"
        elif os.path.exists(cache_fp):
            status = f"cache found: {os.path.basename(cache_fp)}"
        else:
            status = "no cache — press Run to upscale"

        self._status_lbl.setText(
            f"'{char.name}'  {n_dirs}d × {n_frames}f  "
            f"{char.frames.shape[3]}×{char.frames.shape[2]} px  —  {status}"
        )
        self._upscaled = getattr(char, "upscaled_frames", None)
        self._update_preview()

    def _update_preview(self):
        char = self.state.current_character
        if char is None:
            return

        d  = max(0, min(self._dir_spin.value()   - 1, char.frames.shape[0] - 1))
        fi = max(0, min(self._frame_spin.value() - 1, char.frames.shape[1] - 1))

        orig = char.frames[d, fi]
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
