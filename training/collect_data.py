"""Dataset collection from the webcam.

Usage::

    python -m training.collect_data --label HELLO --user alice --samples 20
    python -m training.collect_data --label WATER --samples 15 --type static
    python -m training.collect_data --list

Controls in the live window:

    SPACE  start / stop recording (manual mode)
    ESC    discard the take currently being recorded
    Q      quit

With ``--auto`` you never need to press anything: perform the sign in front of
the camera (e.g. raise your hand and wave). The window automatically starts
recording when your hand appears, stops when it leaves, and saves the take.
Repeat until the sample counter fills up.

Samples are saved to ``data/raw/<user>/<label>_<n>.json`` as raw landmarks
(normalisation happens later in ``training/preprocess.py``). A reference take is
copied to ``data/reference/<label>.json`` for the reverse-communication display.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RecognitionConfig, load_config, ensure_dirs
from app.hand_detector import HandDetector


def list_labels(cfg: RecognitionConfig) -> None:
    labels_path = cfg.paths["labels"]
    if not labels_path.exists():
        print("No labels file yet.")
        return
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    print(f"Language: {data.get('language_full', data.get('language'))}")
    for name, meta in data.get("signs", {}).items():
        print(f"  {name:12s} {meta.get('type', 'static'):8s} en={meta.get('en', name.lower())}")


def ensure_label(cfg: RecognitionConfig, label: str, sign_type: str, en: str) -> None:
    labels_path = cfg.paths["labels"]
    if not labels_path.exists():
        data = {
            "language": "isl",
            "language_full": "Indian Sign Language (initial vocabulary)",
            "notes": [],
            "signs": {},
        }
    else:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
    signs = data.setdefault("signs", {})
    if label not in signs:
        signs[label] = {"type": sign_type, "en": en if en else label.lower()}
        print(f"  Added '{label}' to {labels_path} (type={sign_type}).")
    labels_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def draw_status(frame: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _write_sample(
    cfg: RecognitionConfig,
    label: str,
    sign_type: str,
    user: str,
    user_dir: Path,
    take: list[dict],
    min_frames: int,
    saved_count: int,
    num_samples: int,
) -> Path | None:
    """Save one take; returns the saved path, or None if it was discarded."""
    if len(take) < min_frames:
        print(f"  discarded take with only {len(take)} frames (< {min_frames}).")
        return None
    duration = max(0.001, take[-1]["t"])
    sample = {
        "label": label,
        "type": sign_type,
        "user": user,
        "collected_at": datetime.now().isoformat(),
        "fps": round(len(take) / duration),
        "frames": take,
        "remote": False,
        "session": user,
    }
    out_path = user_dir / (
        f"{label}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    )
    out_path.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print(f"  saved sample {saved_count}/{num_samples}: {out_path.name}")
    if saved_count == 1:
        ref = cfg.paths["reference_signs"] / f"{label}.json"
        ref.write_text(json.dumps(sample, indent=1), encoding="utf-8")
        print(f"  saved reference take for reverse display: {ref}")
    return out_path


def collect_for_label(
    cfg: RecognitionConfig,
    label: str,
    sign_type: str,
    user: str,
    num_samples: int,
    max_frames: int,
    min_frames: int,
    camera_index: int,
    auto: bool = False,
) -> list[str]:
    model_path = cfg.paths["hand_model"]
    if not model_path.exists():
        raise SystemExit(
            f"Hand model missing at {model_path}. Run the app once or download from:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )

    detector = HandDetector(model_path, num_hands=2)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        detector.close()
        raise SystemExit(f"Could not open camera {camera_index}.")

    user_dir = cfg.paths["raw_data"] / user
    user_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths["reference_signs"].mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    if not auto:
        saved.extend(
            _collect_manual(
                cfg, cap, detector, label, sign_type, user, user_dir,
                num_samples, max_frames, min_frames,
            )
        )
    else:
        saved.extend(
            _collect_auto(
                cfg, cap, detector, label, sign_type, user, user_dir,
                num_samples, max_frames, min_frames,
            )
        )

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    return saved


def _collect_manual(
    cfg: RecognitionConfig,
    cap: cv2.VideoCapture,
    detector: HandDetector,
    label: str,
    sign_type: str,
    user: str,
    user_dir: Path,
    num_samples: int,
    max_frames: int,
    min_frames: int,
) -> list[str]:
    saved: list[str] = []
    recording = False
    frames: list[dict] = []
    pending_save = False
    t0 = 0.0

    print(f"\nRecording '{label}' ({sign_type}) for user '{user}' - {num_samples} samples.")
    print("Show the sign and press SPACE. The take stops when your hand leaves the frame,")
    print("when SPACE is pressed again, or after", max_frames, "frames. ESC discards. Q quits.\n")

    while len(saved) < num_samples:
        ok, frame = cap.read()
        if not ok:
            break
        hands = detector.process(frame)
        now = time.time()

        if recording:
            if hands:
                frames.append({"t": round(now - t0, 4),
                               "hands": [h.to_dict() for h in hands]})
            if frames and not hands:       # hand left the frame -> end take
                recording = False
                pending_save = True
            elif len(frames) >= max_frames:
                recording = False
                pending_save = True

        display = frame.copy()
        if hands:
            display = HandDetector.draw_landmarks(display, hands)
        if recording:
            draw_status(display,
                        f"RECORDING {label}  frames={len(frames)}  "
                        "[SPACE=stop] [ESC=discard]",
                        (0, 255, 0))
        else:
            draw_status(display,
                        f"{label} ({sign_type})  saved={len(saved)}/{num_samples}  "
                        "[SPACE=record] [Q=quit]")
        cv2.imshow("Sign Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if recording:
            if key == 32:               # SPACE stops the take
                recording = False
                pending_save = True
            elif key == 27:             # ESC discards it
                recording = False
                frames = []
                pending_save = False
        else:
            if key == 32:               # SPACE starts a new take
                recording = True
                frames = []
                t0 = now
                pending_save = False
            elif key in (ord("q"), ord("Q")):
                break

        if pending_save and not recording:
            pending_save = False
            take, frames = frames, []
            path = _write_sample(cfg, label, sign_type, user, user_dir,
                                 take, min_frames, len(saved) + 1, num_samples)
            if path is not None:
                saved.append(str(path))

    return saved


def _collect_auto(
    cfg: RecognitionConfig,
    cap: cv2.VideoCapture,
    detector: HandDetector,
    label: str,
    sign_type: str,
    user: str,
    user_dir: Path,
    num_samples: int,
    max_frames: int,
    min_frames: int,
) -> list[str]:
    saved: list[str] = []
    settle_s = 0.35
    cool_s = 0.9
    hand_since = 0.0
    phase = "idle"          # idle -> settle -> record -> (finalize) -> cool
    frames: list[dict] = []
    t0 = 0.0
    phase_since = time.time()

    print(f"\nAUTO: recording '{label}' ({sign_type}) - {num_samples} samples.")
    print("Just perform the sign in front of the camera. No buttons needed.")
    print("It records while your hand is visible, saves when you lower it, repeats.\n")

    while len(saved) < num_samples:
        ok, frame = cap.read()
        if not ok:
            break
        hands = detector.process(frame)
        now = time.time()

        if phase == "idle":
            hand_since = now
            if hands:
                phase, phase_since = "settle", now
        elif phase == "settle":
            if not hands:
                phase = "idle"
            elif now - hand_since >= settle_s:
                phase, t0, frames = "record", now, []
        elif phase == "record":
            if hands:
                frames.append({"t": round(now - t0, 4),
                               "hands": [h.to_dict() for h in hands]})
            if frames and not hands:
                phase = "cool"
                phase_since = now
                take, frames = frames, []
                path = _write_sample(cfg, label, sign_type, user, user_dir,
                                     take, min_frames, len(saved) + 1, num_samples)
                if path is not None:
                    saved.append(str(path))
            elif len(frames) >= max_frames:
                phase = "cool"
                phase_since = now
                take, frames = frames, []
                path = _write_sample(cfg, label, sign_type, user, user_dir,
                                     take, min_frames, len(saved) + 1, num_samples)
                if path is not None:
                    saved.append(str(path))
        elif phase == "cool":
            if now - phase_since >= cool_s:
                phase = "idle"

        display = frame.copy()
        if hands:
            display = HandDetector.draw_landmarks(display, hands)
        status, color = _auto_status(phase, label, len(saved), num_samples, len(frames))
        draw_status(display, status, color)
        cv2.imshow("Sign Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key == 27:
            frames = []
            phase = "idle"

    return saved


def _auto_status(phase: str, label: str, saved: int, num_samples: int,
                 n_frames: int) -> tuple[str, tuple[int, int, int]]:
    if phase == "settle":
        return f"AUTO {label}: hold your hand still, recording in a moment...", (255, 165, 0)
    if phase == "record":
        return f"RECORDING {label}  {n_frames} frames  saved={saved}/{num_samples}", (0, 255, 0)
    if phase == "cool":
        return f"Saved!  {saved}/{num_samples}  -  next one...", (200, 200, 255)
    return f"AUTO {label}: perform the sign NOW  saved={saved}/{num_samples}", (255, 255, 255)


def run_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect sign-language training samples.")
    parser.add_argument("--label", "-l", required=False, help="Sign label, e.g. HELLO")
    parser.add_argument("--user", "-u", default=None, help="Who is signing")
    parser.add_argument("--samples", "-n", type=int, default=15, help="Samples per label")
    parser.add_argument("--type", dest="sign_type", choices=["static", "dynamic"],
                        default=None, help="static or dynamic sign")
    parser.add_argument("--en", default=None,
                        help="Display word used in sentences (default: lowercase label)")
    parser.add_argument("--max-frames", type=int, default=45,
                        help="Max frames recorded per take (dynamic signs)")
    parser.add_argument("--min-frames", type=int, default=8,
                        help="Minimum frames to keep a take")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-record: no key presses - perform the sign, "
                             "it saves when your hand leaves the frame")
    parser.add_argument("--list", action="store_true", help="List configured labels")

    args = parser.parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)

    if args.list:
        list_labels(cfg)
        return 0
    if not args.label:
        parser.print_help()
        return 2

    user = args.user or cfg.user_default
    sign_type = args.sign_type
    if sign_type is None:
        sign_type = "dynamic"
    ensure_label(cfg, args.label, sign_type, args.en or "")
    saved = collect_for_label(
        cfg,
        args.label,
        sign_type,
        user,
        args.samples,
        args.max_frames,
        args.min_frames,
        args.camera,
        auto=args.auto,
    )
    print(f"\nDone. {len(saved)} sample(s) saved for '{args.label}'.")
    print("Next: python -m training.preprocess  then  python -m training.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())