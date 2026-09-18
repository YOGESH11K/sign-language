"""Two-way communication desktop UI (Tkinter).

Tab 1 "Sign -> Text": live webcam + detection + sentence building + TTS.
Tab 2 "Text -> Sign": type (or speak) text, get the sign-gloss sequence and
replay recorded reference visuals.

Design aims (feature 8): large buttons, high contrast, large readable text,
minimal animation, keyboard shortcuts, clear status feedback.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from tkinter import ttk
import tkinter as tk

import cv2
import numpy as np

from app.config import RecognitionConfig
from app.camera import CameraUnavailableError, WebcamCapture
from app.hand_detector import HandDetector
from app.predictor import SignPredictor, Prediction
from app.sentence_builder import SentenceBuilder
from app.speech import SpeechSynthesizer
from app.speech_input import SpeechInput
from app.reverse_communication import (
    SignDictionary,
    SignDictionaryService,
    SignRenderer,
    load_reference_sample,
    reference_frame_count,
)

try:
    from PIL import Image, ImageTk

    _PIL_OK = True
except ImportError:  # pragma: no cover
    _PIL_OK = False


FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_LARGE = ("Segoe UI", 22, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_BIG_WORD = ("Segoe UI", 40, "bold")


class CameraWorker(threading.Thread):
    """Captures, detects and predicts in a loop; pushes UI messages."""

    def __init__(self, cfg: RecognitionConfig, msg_queue: queue.Queue,
                 show_landmarks, auto_add):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.queue = msg_queue
        self.show_landmarks = show_landmarks
        self.auto_add = auto_add
        self._stop = threading.Event()
        self.frame_count = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            cam = WebcamCapture(self.cfg.camera_index,
                                width=self.cfg.camera_width,
                                height=self.cfg.camera_height)
            cam.open()
        except CameraUnavailableError as exc:
            self.queue.put(("error", str(exc)))
            return
        try:
            detector = HandDetector(self.cfg.paths["hand_model"], num_hands=2)
        except FileNotFoundError as exc:
            cam.release()
            self.queue.put(("error", str(exc)))
            return

        predictor = SignPredictor(self.cfg)
        status = predictor.load_models()
        self.queue.put(("model_status", (status.static_message,
                                         status.sequence_message)))

        t0 = time.time()
        try:
            while not self._stop.is_set():
                try:
                    ok, frame = cam.read()
                    if not ok:
                        self.queue.put(("error", "Camera stream ended unexpectedly."))
                        break
                    self.frame_count += 1
                    hands = detector.process(frame)
                    emission = predictor.update(hands)

                    display = frame
                    if self.show_landmarks.get():
                        display = HandDetector.draw_landmarks(frame, hands)
                    if not hands:
                        cv2.rectangle(display, (10, 40), (450, 80), (0, 0, 0), -1)
                        cv2.putText(display, "No hand detected - show a sign",
                                    (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                    self.queue.put(("frame", display, len(hands)))
                    if emission is not None:
                        if self.auto_add.get():
                            self.queue.put(("auto_add", emission.label))
                        self.queue.put(("new_sign", emission.label,
                                        emission.confidence))
                except Exception as exc:
                    self.queue.put(("error",
                                    f"Detection stopped: {type(exc).__name__}: {exc}"))
                    break

                if time.time() - t0 > 6:
                    t0 = time.time()
                    self.queue.put(("hand_report", len(hands)))
        finally:
            cam.release()
            detector.close()


class ReversePlayer(threading.Thread):
    """Animate recorded sign references across the reverse tab canvas."""

    def __init__(self, cfg: RecognitionConfig, glosses: list[str],
                 visual_queue: queue.Queue, fps: int = 12):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.glosses = glosses
        self.visual_queue = visual_queue
        self.fps = fps
        self._stop = threading.Event()
        self.renderer = SignRenderer()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        for gloss in self.glosses:
            if self._stop.is_set():
                break
            sample = load_reference_sample(self.cfg, gloss)
            n = reference_frame_count(sample)
            for i in range(n):
                if self._stop.is_set():
                    return
                img = self.renderer.render_sample(sample, i)
                self.visual_queue.put(("visual", gloss, img))
                time.sleep(1.0 / self.fps)
            if n <= 1:
                time.sleep(0.8)
        self.visual_queue.put(("visual_done",))


class MainWindow(tk.Tk):
    """The whole application window."""

    def __init__(self, cfg: RecognitionConfig, start_camera: bool = True):
        super().__init__()
        self.cfg = cfg
        self._msg_queue: queue.Queue = queue.Queue()
        self._visual_queue: queue.Queue = queue.Queue()
        self._worker: CameraWorker | None = None
        self._player: ReversePlayer | None = None
        self._vtk = None
        self._visual_tk = None
        self._last_hand_count = 0
        self._signs_seen: list[str] = []

        self.title("Sign Language Communication")
        self.geometry("1180x820")
        self.minsize(960, 680)
        self.configure(bg="#101418")

        self.sentence_builder = SentenceBuilder(cfg.paths["labels"])
        self.speech = SpeechSynthesizer(rate=cfg.speech_rate, muted=cfg.tts_muted)
        self.mic = SpeechInput(language="en-IN")
        self.sign_dict = SignDictionaryService(SignDictionary(cfg.paths["labels"]))

        self.show_landmarks = tk.BooleanVar(value=True)
        self.auto_add = tk.BooleanVar(value=True)
        self.muted = tk.BooleanVar(value=cfg.tts_muted)

        self._build_ui()
        self._bind_keys()
        self.speech.on_started = lambda _t: self.after(0, self._speech_started)
        self.speech.on_finished = lambda _t: self.after(0, self._speech_finished)
        self._destroyed = False
        self.after(50, self._poll_visual)

        if start_camera:
            self.start_camera()

    # ------------------------------------------------------------ UI build
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=FONT_TITLE, foreground="#000000")
        style.configure("Hint.TLabel", font=("Segoe UI", 10), foreground="#404040")
        style.configure("Big.TLabel", font=FONT_BODY)

        header = tk.Label(self, text="SIGN LANGUAGE COMMUNICATION",
                          font=FONT_TITLE, bg="#101418", fg="#31d0aa")
        header.pack(fill="x", padx=8, pady=(8, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)
        self.tab_sign = ttk.Frame(self.notebook, padding=6)
        self.tab_reverse = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_sign, text="  Sign -> Text  ")
        self.notebook.add(self.tab_reverse, text="  Text -> Sign  ")

        self._build_sign_tab()
        self._build_reverse_tab()

        self.status_var = tk.StringVar(value="Starting up...")
        status = tk.Label(self, textvariable=self.status_var, anchor="w",
                          bg="#101418", fg="#cfcfcf", font=("Segoe UI", 11))
        status.pack(fill="x", padx=8, pady=(0, 6))

    def _build_sign_tab(self) -> None:
        left = ttk.Frame(self.tab_sign)
        right = ttk.Frame(self.tab_sign)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="both", expand=True)

        # --- video side ---
        video_box = tk.LabelFrame(left, text="Live webcam", relief="groove",
                                  font=("Segoe UI", 11, "bold"))
        video_box.pack(fill="both", expand=True)
        self.video_label = tk.Label(video_box, text="Starting camera...",
                                    bg="#000000", fg="#ffffff",
                                    font=("Segoe UI", 13))
        self.video_label.pack(fill="both", expand=True)

        ctl = ttk.Frame(left)
        ctl.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(ctl, text="Show hand landmarks",
                        variable=self.show_landmarks).pack(side="left")
        self.hand_var = tk.StringVar(value="hands: 0")
        ttk.Label(ctl, textvariable=self.hand_var,
                  font=FONT_BODY).pack(side="left", padx=12)

        # --- detected text side ---
        det = tk.LabelFrame(right, text="Detected", relief="groove",
                            font=("Segoe UI", 11, "bold"))
        det.pack(fill="both", expand=True)
        self.sign_var = tk.StringVar(value="-")
        tk.Label(det, textvariable=self.sign_var, font=FONT_BIG_WORD,
                 height=2, bg="#0b66d9", fg="#ffffff").pack(fill="x")
        self.conf_var = tk.StringVar(value="confidence: -")
        ttk.Label(det, textvariable=self.conf_var,
                  font=("Segoe UI", 12)).pack(anchor="w", padx=6)

        history_frame = ttk.Frame(det)
        history_frame.pack(fill="both", expand=True, padx=6, pady=4)
        ttk.Label(history_frame, text="Recent terms:",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.history_var = tk.StringVar(value="")
        ttk.Label(history_frame, textvariable=self.history_var,
                  font=("Segoe UI", 12), wraplength=420,
                  justify="left").pack(fill="x")

        # --- sentence row ---
        sent = tk.LabelFrame(self.tab_sign, text="Sentence", relief="groove",
                             font=("Segoe UI", 11, "bold"))
        sent.pack(side="bottom", fill="x", padx=6, pady=6)
        self.sentence_var = tk.StringVar(value="")
        tk.Label(sent, textvariable=self.sentence_var, font=FONT_LARGE,
                 bg="#ffffff", fg="#000000", height=2, anchor="w",
                 highlightthickness=1, highlightbackground="#888888",
                 padx=8).pack(fill="x", padx=6, pady=(6, 2))

        btns = ttk.Frame(sent)
        btns.pack(fill="x", padx=6, pady=(0, 8))
        self.btn_add = ttk.Button(btns, text="Add Word (Alt+A)",
                                  command=self.on_add, style="Big.TButton")
        self.btn_del = ttk.Button(btns, text="Delete Last (Alt+B)",
                                  command=self.on_delete)
        self.btn_clear = ttk.Button(btns, text="Clear (Alt+C)",
                                    command=self.on_clear)
        self.btn_speak = ttk.Button(btns, text="Speak (Alt+S)",
                                    command=self.on_speak)
        self.btn_stop = ttk.Button(btns, text="Stop",
                                   command=self.on_stop_speech)
        for b in (self.btn_add, self.btn_del, self.btn_clear,
                  self.btn_speak, self.btn_stop):
            b.configure(style="Big.TButton", padding=(12, 8))
            b.pack(side="left", padx=4)
        ttk.Checkbutton(btns, text="Auto-add detected sign",
                        variable=self.auto_add).pack(side="left", padx=10)
        ttk.Checkbutton(btns, text="Mute TTS",
                        variable=self.muted,
                        command=self._apply_mute).pack(side="left", padx=4)

        rate_row = ttk.Frame(sent)
        rate_row.pack(fill="x", padx=6)
        ttk.Label(rate_row, text="Speech rate:").pack(side="left")
        self.rate_var = tk.IntVar(value=self.cfg.speech_rate)
        ttk.Scale(rate_row, from_=-10, to=10, variable=self.rate_var,
                  command=self._rate_changed).pack(side="left", padx=8, fill="x",
                                                    expand=True)
        ttk.Label(rate_row, text="(slow  -10 ... +10  fast)").pack(side="left")

    def _build_reverse_tab(self) -> None:
        top = ttk.Frame(self.tab_reverse)
        top.pack(fill="x")
        ttk.Label(top, text="Type a sentence to show as sign language:",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.entry_text = tk.StringVar()
        self.entry = ttk.Entry(top, textvariable=self.entry_text, font=("Segoe UI", 13))
        self.entry.pack(fill="x", pady=(4, 0))
        self.entry.bind("<Return>", lambda _e: self.on_show_signs())

        row = ttk.Frame(top)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="Show Signs (Enter)",
                   command=self.on_show_signs, style="Big.TButton",
                   padding=(12, 8)).pack(side="left")
        ttk.Button(row, text="Use Microphone...", command=self.on_mic,
                   style="Big.TButton", padding=(12, 8)).pack(side="left", padx=6)
        ttk.Button(row, text="Play References",
                   command=self.on_play_reference, style="Big.TButton",
                   padding=(12, 8)).pack(side="left", padx=6)
        ttk.Button(row, text="Speak Text", command=self.on_speak_reverse,
                   style="Big.TButton", padding=(12, 8)).pack(side="left", padx=6)
        self.mic_status = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.mic_status,
                  font=("Segoe UI", 10)).pack(side="left", padx=8)

        mid = ttk.Frame(self.tab_reverse)
        mid.pack(fill="both", expand=True, pady=6)
        # gloss cards
        self.cards_frame = ttk.Frame(mid)
        self.cards_frame.pack(fill="x")
        self.cards_var = tk.StringVar(value="")
        self.cards_label = ttk.Label(self.cards_frame,
                                     textvariable=self.cards_var,
                                     font=("Segoe UI", 16, "bold"), anchor="w",
                                     wraplength=900)
        self.cards_label.pack(fill="x")

        visual_box = tk.LabelFrame(mid, text="Sign visual (recorded references)",
                                   relief="groove")
        visual_box.pack(fill="both", expand=True, pady=(6, 0))
        self.visual_label = tk.Label(visual_box, text="Press 'Show Signs' to begin.",
                                     bg="#000000", fg="#ffffff",
                                     font=("Segoe UI", 12))
        self.visual_label.pack(fill="both", expand=True)

        note = tk.Label(self.tab_reverse,
                        text="Note: visuals replay genuine recorded landmark takes. "
                             "Word->sign mapping is limited to the trained vocabulary; "
                             "unknown words are fingerspelled if letter signs exist.",
                        font=("Segoe UI", 9), fg="#555555")
        note.pack(anchor="w")

    def _run_and_break(self, cmd):
        cmd()
        return "break"

    def _bind_keys(self) -> None:
        if hasattr(self, "_bound"):
            return
        self._bound = True
        self.bind("<Alt-a>", lambda e: self._run_and_break(self.on_add))
        self.bind("<Alt-b>", lambda e: self._run_and_break(self.on_delete))
        self.bind("<Alt-c>", lambda e: self._run_and_break(self.on_clear))
        self.bind("<Alt-s>", lambda e: self._run_and_break(self.on_speak))
        self.bind("<Control-Return>",
                  lambda e: self._run_and_break(self.on_speak_reverse))

    # ----------------------------------------------------------- commands
    def _apply_mute(self) -> None:
        self.speech.set_muted(self.muted.get())

    def _rate_changed(self, _v) -> None:
        try:
            self.speech.set_rate(self.rate_var.get())
        except Exception:
            pass

    def on_add(self) -> None:
        pred = self._current_prediction()
        label = pred.label if pred else None
        if label:
            self._add_label_to_sentence(label)
            self._refresh_sentence()

    def on_delete(self) -> None:
        self.sentence_builder.undo()
        self._refresh_sentence()

    def on_clear(self) -> None:
        self.sentence_builder.clear()
        self._refresh_sentence()

    def _current_prediction(self) -> Prediction | None:
        return getattr(self, "_last_pred", None)

    def _add_label_to_sentence(self, label: str) -> None:
        added = self.sentence_builder.add_word(label, suppress_duplicates=True)
        if added:
            self._status(f"Added '{self.sentence_builder.display_word(label)}'.")
        else:
            self._status(f"Ignored duplicate '{label}'.")

    def on_speak(self) -> None:
        text = self.sentence_builder.confirm()
        if not text:
            self._status("Nothing to speak yet.")
            return
        ok = self.speech.speak(text)
        if ok:
            self._status(f"Speaking: {text}")
        else:
            self._status("Text-to-speech unavailable on this system.")

    def on_stop_speech(self) -> None:
        self.speech.stop()

    # ------------------------------------------------------------ reverse
    def on_show_signs(self) -> None:
        text = self.entry_text.get().strip()
        if not text:
            self._status("Type some text first.")
            return
        tokens = self.sign_dict.to_sign_sequence(text)
        self._tokens = tokens
        if self._player is not None:
            self._player.stop()
        glosses = [t.gloss for t in tokens]
        has_ref = sum(1 for g in glosses
                      if load_reference_sample(self.cfg, g) is not None)
        label = "  |  ".join(t.display() for t in tokens) if tokens else "(empty)"
        self.cards_var.set(label)
        self._status(f"Sign sequence: {len(tokens)} sign(s) "
                     f"({has_ref} with recorded visual).")
        self._start_player(glosses)

    def _start_player(self, glosses: list[str]) -> None:
        if self._player is not None:
            self._player.stop()
        if not glosses:
            self.visual_label.configure(image="")
            self.visual_label.configure(text="(no signs generated)")
            return
        self._player = ReversePlayer(self.cfg, glosses, self._visual_queue)
        self._player.start()

    def on_play_reference(self) -> None:
        glosses = [t.gloss for t in getattr(self, "_tokens", [])]
        if not glosses:
            self._status("Generate a sign sequence first.")
            return
        self._start_player(glosses)

    def on_speak_reverse(self) -> None:
        text = self.entry_text.get().strip()
        if text:
            ok = self.speech.speak(text)
            self._status(f"Speaking: {text}" if ok else "TTS unavailable on this system.")

    def on_mic(self) -> None:
        if not self.mic.microphone_available:
            self.mic_status.set("Microphone unavailable.")
            return
        if not self.mic.recognizer_available:
            self.mic_status.set("Install SpeechRecognition package.")
            return

        def work():
            self.mic_status.set("Listening (5s)... speak now.")
            result = self.mic.listen_and_transcribe(duration=5.0)
            self.mic_status.set(self.mic.error or "Done.")
            if result:
                self.after(0, lambda: (self.entry_text.set(result),
                                       self.on_show_signs()))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------- camera
    def start_camera(self) -> None:
        self._ensure_hand_model()
        self._worker = CameraWorker(self.cfg, self._msg_queue,
                                    self.show_landmarks, self.auto_add)
        self._worker.start()
        self.after(20, self._poll_messages)
        self.after(20, self._poll_visual)

    def _ensure_hand_model(self) -> None:
        path = self.cfg.paths["hand_model"]
        if path.exists():
            return
        self._status("Downloading hand landmark model (one-time)...")
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(self.cfg.hand_model_url, path)
            self._status("Hand landmark model downloaded.")
        except Exception as exc:
            self._status(f"Could not download hand model: {exc}")

    def _poll_messages(self) -> None:
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                try:
                    self._handle_message(msg)
                except Exception as exc:  # never let one bad message kill the poller
                    self._status(f"UI error: {exc}")
        except queue.Empty:
            pass
        if self._worker is not None and self._worker.is_alive():
            self.after(16, self._poll_messages)

    def _handle_message(self, msg) -> None:
        kind = msg[0]
        if kind == "frame":
            _, bgr, hands = msg
            self._last_hand_count = hands
            self.hand_var.set(f"hands: {hands}")
            if _PIL_OK:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._vtk = ImageTk.PhotoImage(img)
                self.video_label.configure(image=self._vtk)
        elif kind == "new_sign":
            _, label, conf = msg
            self.sign_var.set(label.replace("_", " "))
            self.conf_var.set(f"confidence: {conf*100:.0f}%")
            self._last_pred = Prediction(label=label, confidence=conf,
                                         source=getattr(self, "_last_src", "static"))
            if label not in self._signs_seen:
                self._signs_seen.append(label)
                self.history_var.set(", ".join(self._signs_seen[-8:]))
            if self.auto_add.get():
                self._add_label_to_sentence(label)
                self._refresh_sentence()
        elif kind == "auto_add":
            _, label = msg
            self._add_label_to_sentence(label)
            self._refresh_sentence()
        elif kind == "model_status":
            _, (static_msg, seq_msg) = msg
            self._status(f"{static_msg} | {seq_msg}")
        elif kind == "hand_report":
            self._last_hand_count = msg[1]
        elif kind == "error":
            self.video_label.configure(text="Camera unavailable.\n"
                                            "Check that a webcam is connected and not "
                                            "blocked by another app. Restart to retry.")
            self._status(msg[1])

    def _poll_visual(self) -> None:
        try:
            while True:
                msg = self._visual_queue.get_nowait()
                if msg[0] == "visual":
                    _, gloss, bgr = msg
                    if _PIL_OK:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        self._visual_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
                        self.visual_label.configure(image=self._visual_tk,
                                                    text="")
                elif msg[0] == "visual_done":
                    pass
        except queue.Empty:
            pass
        self.after(20, self._poll_visual)

    def _speech_started(self) -> None:
        self._status("Speaking...")

    def _speech_finished(self) -> None:
        self._status("Ready.")

    # ------------------------------------------------------------- misc
    def _refresh_sentence(self) -> None:
        self.sentence_var.set(self.sentence_builder.to_sentence())

    def _status(self, text: str) -> None:
        self.status_var.set(text)

    def close(self) -> None:
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        if self._worker is not None:
            self._worker.stop()
        if self._player is not None:
            self._player.stop()
        self.speech.stop()
        self.destroy()


def launch(cfg: RecognitionConfig, start_camera: bool = True) -> int:
    if not _PIL_OK:
        print("Pillow is required for the UI: pip install Pillow")
        return 1
    app = MainWindow(cfg, start_camera=start_camera)
    app.mainloop()
    app.close()
    return 0