"""Text-to-speech using a local backend.

Backends tried in order:
  1. Windows SAPI via PowerShell (built-in, offline, no extra dependency)
  2. pyttsx3 (cross-platform) - installed optionally

Includes speak/stop, a speech-rate control and a mute switch.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
from typing import Callable, Optional


class SpeechSynthesizer:
    def __init__(self, rate: int = 0, muted: bool = False, language: str = "en-IN"):
        self.rate = max(-10, min(10, int(rate)))
        self.muted = muted
        self.language = language
        self.backend: Optional[str] = None
        self.backend_detail: str = "no TTS backend found"
        self.on_started: Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[str], None]] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._current_text = ""
        self._detect_backend()

    def _detect_backend(self) -> None:
        if shutil.which("powershell"):
            self.backend = "powershell-sapi"
            self.backend_detail = "Windows SAPI (System.Speech, offline)"
            return
        try:
            import pyttsx3  # noqa: F401

            self.backend = "pyttsx3"
            self.backend_detail = "pyttsx3"
            return
        except ImportError:
            pass
        self.backend = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def set_rate(self, value: int) -> None:
        self.rate = max(-10, min(10, int(value)))

    def set_muted(self, value: bool) -> None:
        self.muted = bool(value)

    def _powershell_command(self, text: str) -> str:
        safe = json.dumps(text)  # valid as a PowerShell string literal
        script = (
            "Add-Type -AssemblyName System.Speech;"
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$s.Rate = {self.rate};"
            f"$s.Speak({safe});"
        )
        return base64.b64encode(script.encode("utf-16-le")).decode("ascii")

    def _backend_speak(self, text: str) -> None:
        if self.backend == "powershell-sapi":
            cmd = ["powershell", "-NoProfile", "-NonInteractive",
                   "-EncodedCommand", self._powershell_command(text)]
            flags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                flags |= getattr(subprocess, "CREATE_NO_WINDOW")
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            self._proc.wait()
            self._proc = None
        elif self.backend == "pyttsx3":
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 150 + self.rate * 15)
            self._proc = None
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def speak(self, text: str) -> bool:
        """Speak ``text`` in the background. Returns True if it started."""
        text = (text or "").strip()
        if not text or not self.available:
            return False
        self.stop()
        self._current_text = text
        if self.on_started:
            self.on_started(text)
        thread = threading.Thread(target=self._speak_worker, name="tts", daemon=True)
        thread.start()
        return True

    def _speak_worker(self) -> None:
        text = self._current_text
        try:
            if not self.muted:
                with self._lock:
                    self._backend_speak(text)
        except Exception:
            pass
        finally:
            self._proc = None
            self._current_text = ""
            if self.on_finished:
                self.on_finished(text)

    def stop(self) -> None:
        """Cancel any active speech immediately."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
        self._current_text = ""
        if self.on_finished:
            self.on_finished("")

    def is_speaking(self) -> bool:
        with self._lock:
            return self._proc is not None or self._current_text != ""