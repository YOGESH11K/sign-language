"""Sign Language Communication - application entry point.

Usage::

    python main.py                      # full app with webcam
    python main.py --no-camera          # UI without camera (reverse mode only)
    python main.py --camera 1           # pick a different webcam device

Other one-off commands live in the ``training`` package:

    python -m training.collect_data --label HELLO --samples 20
    python -m training.preprocess
    python -m training.train --kind both
    python -m training.evaluate --kind both
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign Language Communication")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam device index to use (default: config value, usually 0)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Start the UI without a webcam (reverse communication only)")
    parser.add_argument("--config", default=None,
                        help="Path to a JSON config file (optionally generated "
                             "by --write-config)")
    parser.add_argument("--write-config", action="store_true",
                        help="Write a default config.json and exit")
    args = parser.parse_args()

    from app.config import load_config, ensure_dirs, save_default_config

    if args.write_config:
        path = save_default_config(args.config)
        print(f"Wrote default config to {path}")
        return 0

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.camera is not None:
        cfg.camera_index = args.camera

    if not (cfg.paths["labels"]).exists():
        print("No data/labels.json found. The app works, but no vocabulary is "
              "configured - see README 'Add new signs'.")
        return 1

    from ui.main_window import launch

    return launch(cfg, start_camera=not args.no_camera)


if __name__ == "__main__":
    sys.exit(main())