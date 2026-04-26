"""
FrmLoader — Fallout 2 FRM asset loading with bundled palette.

FRM binary layout (big-endian):
  4  B  version
  2  B  fps
  2  B  action_frame
  2  B  frames_per_direction
  12 B  shift_x[6]  (int16 × 6)
  12 B  shift_y[6]  (int16 × 6)
  24 B  dir_offsets[6]  (uint32 × 6, relative to frame-data block start)
  4  B  data_size
  --- frame data block ---
  Per frame: uint16 w, uint16 h, uint32 pixel_size, int16 ox, int16 oy, w*h bytes
"""

import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


NUM_DIRECTIONS = 6


@dataclass
class FrameSequence:
    frames: list              # list[list[PIL.Image.Image]]  — [direction][frame_index]
    width: int                # max frame width across all directions/frames
    height: int               # max frame height across all directions/frames
    num_directions: int       # always 6 for Fallout 2
    frames_per_direction: int


class FrmLoader:
    def __init__(self, frm_path: str, pal_path: str = "color/color.pal"):
        self.frm_path = frm_path
        self.pal_path = pal_path

    def load(self) -> FrameSequence:
        pal_path = Path(self.pal_path)
        if not pal_path.exists():
            raise FileNotFoundError(
                f"Palette file not found: {pal_path.resolve()}. "
                "Place the Fallout 2 color.pal at 'color/color.pal' relative to the "
                "project root, or pass pal_path= explicitly."
            )
        palette_flat = self._read_pal(pal_path)
        return self._decode_frm(self.frm_path, palette_flat)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pal(path: Path) -> list:
        """Return a flat list of 768 ints [R,G,B, R,G,B, ...] scaled to 8-bit."""
        with open(path, "rb") as f:
            raw = f.read(768)
        # Fallout .pal stores 6-bit values (0-63); multiply by 4 → 0-252
        return [b * 4 for b in raw]

    @staticmethod
    def _decode_frm(frm_path: str, palette_flat: list) -> FrameSequence:
        with open(frm_path, "rb") as f:
            data = f.read()

        pos = 0
        _version        = struct.unpack_from(">I", data, pos)[0]; pos += 4
        _fps            = struct.unpack_from(">H", data, pos)[0]; pos += 2
        _action_frame   = struct.unpack_from(">H", data, pos)[0]; pos += 2
        frames_per_dir  = struct.unpack_from(">H", data, pos)[0]; pos += 2
        pos += 12   # shift_x[6]
        pos += 12   # shift_y[6]
        dir_offsets     = struct.unpack_from(">6I", data, pos); pos += 24
        pos += 4    # data_size

        frame_data_base = pos

        all_dirs: list[list[Image.Image]] = []
        max_w = max_h = 0

        for d in range(NUM_DIRECTIONS):
            p = frame_data_base + dir_offsets[d]
            dir_frames: list[Image.Image] = []
            for _ in range(frames_per_dir):
                w, h        = struct.unpack_from(">HH", data, p); p += 4
                _pixel_size = struct.unpack_from(">I",  data, p)[0]; p += 4
                _ox, _oy    = struct.unpack_from(">hh", data, p); p += 4
                pixels      = data[p : p + w * h]; p += w * h

                # Build an indexed (P-mode) image and convert to RGB via palette
                img = Image.frombytes("P", (w, h), pixels)
                img.putpalette(palette_flat)
                dir_frames.append(img.convert("RGB"))

                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h

            all_dirs.append(dir_frames)

        return FrameSequence(
            frames=all_dirs,
            width=max_w,
            height=max_h,
            num_directions=NUM_DIRECTIONS,
            frames_per_direction=frames_per_dir,
        )
