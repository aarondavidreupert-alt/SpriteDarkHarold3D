import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config() -> dict:
    """Return config dict from config.json, falling back to defaults."""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"input_mode": "npy", "input_path": "", "upscale": False}


def save_config(cfg: dict) -> None:
    """Write config dict to config.json next to main.py."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
