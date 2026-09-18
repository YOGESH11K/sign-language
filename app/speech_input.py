"""Microphone speech input for the reverse-communication mode.

Captures audio with ``sounddevice`` (no PyAudio needed) and transcribes it with
the free Google Web Speech Recognizer via the ``SpeechRecognition`` package.

Note: the Google recognizer requires an internet connection. If neither is
available the app degrades gracefully and shows a friendly message.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_RECOGNIZER_AVAILABLE = False
try:
    import speech_recognition as sr  # type: ignore

    _RECOGNIZER_AVAILABLE = True
except Exception:
    sr = None  # type: ignore


class MicrophoneUnavailableError(Exception):
    pass


class SpeechInput:
    def __init__(self, language: str = "en-IN", sample_rate: int = 16000):
        self.language = language
        self.sample_rate = sample_rate

    @property
    def microphone_available(self) -> bool:
        try:
            import sounddevice  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def recognizer_available(self) -> bool:
        return _RECOGNIZER_AVAILABLE

    @property
    def status(self) -> str:
        if not self.microphone_available:
            return "Microphone library (sounddevice) unavailable."
        if not self.recognizer_available:
            return ("Speech recognition package unavailable. Install with "
                    "`pip install SpeechRecognition`.")
        return "Microphone recognizer ready (uses Google Web API, needs internet)."

    def record_audio(self, duration: float) -> Optional[np.ndarray]:
        """Record ``duration`` seconds of mono int16 audio at sample_rate."""
        if not self.microphone_available:
            raise MicrophoneUnavailableError("sounddevice is not installed.")
        import sounddevice as sd

        frames = int(self.sample_rate * max(0.5, duration))
        try:
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1,
                           dtype="int16")
            sd.wait()
        except Exception as exc:
            raise MicrophoneUnavailableError(f"Recording failed: {exc}") from exc
        audio = np.asarray(audio).reshape(-1)
        if audio.size == 0:
            return None
        return audio

    def _audio_data(self, audio: np.ndarray):
        raw = audio.astype("<i2").tobytes()
        return sr.AudioData(raw, self.sample_rate, 2)

    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """Record then transcribe. Returns text, or '' with a message set in ``error``."""
        self.error = ""
        if not _RECOGNIZER_AVAILABLE:
            self.error = "SpeechRecognition package is not installed."
            return ""
        audio = self.record_audio(duration)
        if audio is None:
            self.error = "No audio captured - is the microphone working?"
            return ""
        recognizer = sr.Recognizer()
        try:
            return recognizer.recognize_google(
                self._audio_data(audio), language=self.language
            )
        except sr.UnknownValueError:
            self.error = "Could not understand what was said."
            return ""
        except sr.RequestError as exc:
            self.error = f"Speech recognition service offline: {exc}"
            return ""
        except Exception as exc:  # pragma: no cover
            self.error = f"Speech recognition failed: {exc}"
            return ""