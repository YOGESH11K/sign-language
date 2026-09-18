# Real-Time Sign Language Communication System

> **Live in the browser:** https://stem-signlanguage.vercel.app — real-time sign
> recognition runs entirely in the tab (MediaPipe hand tracking + the trained GRU
> exported to ONNX via onnxruntime-web/WebAssembly). The same page also hosts the
> desktop instructions. `python -m training.export_onnx` ships a freshly trained
> model to `web/` for redeploying.

A desktop application that turns a webcam into a two-way communication aid for
deaf / hard-of-hearing / non-speaking users and the people they talk with.

* **Sign -> Text / Speech** — the user signs in front of the webcam; the system
  recognises the sign in real time, builds a sentence, and optionally speaks it.
* **Text / Speech -> Sign** — the other person types (or speaks) a sentence; the
  system shows it as an ordered sequence of sign-language glosses, replaying
  recorded reference visuals of signs that exist in the local dataset.

Everything runs locally. No camera frames are sent anywhere; training samples
are only saved when you explicitly run the collection tool.

> **Status**: this is a working end-to-end ML pipeline with a small initial
> vocabulary. IE the vocab is limited and any accuracy figures come from *your*
> collected data (plus a synthetic CI self-test). See [Limitations](#limitations).

---

## 1. Why ISL (Indian Sign Language)? And what is *actually* supported

This project targets **ISL** first, not ASL. ASL and ISL are different
languages — the two-handed alphabet, word order and many handshapes differ — so
nothing here assumes they are interchangeable.

The vocabulary is defined in [`data/labels.json`](data/labels.json) and is fully
configurable. Current starter vocabulary:

| Sign      | Type    | Display word |
|-----------|---------|--------------|
| A, B, C, D, E … | static (handshape) | a, b, c, … |
| HELLO     | dynamic | hello |
| THANK_YOU | dynamic | thank you |
| YES / NO  | dynamic | yes / no |
| HELP      | dynamic | help |
| HOSPITAL  | dynamic | hospital |
| HOW       | dynamic | how |
| PLEASE / SORRY | static | please / sorry |
| I, YOU, MY, NAME, IS, ARE | static | I, you, my, name, is, are |
| FOOD, WATER, HOME | static | food, water, home |
| GOOD, BAD, STOP | static | good, bad, stop |
| EMERGENCY | dynamic | emergency |
| RAHUL     | static | Rahul (example proper noun) |

* **static** signs are recognised from a single hand posture by the Random
  Forest / MLP classifier.
* **dynamic** signs use the movement over a short sequence and are recognised by
  the GRU sequence model.

Not supported (yet): full grammatical ISL sentences, classifier/role-shift
grammar, finger-spelling of arbitrary words beyond the configured letters, and
any sign not in `labels.json`.

### Dataset source and limitations

* The dataset is collected **by the users of this app** — there is no bundled
  ISL corpus. Use `training/collect_data.py` to record your own samples.
* Raw samples are stored as *landmark sequences* (21 MediaPipe hand landmarks
  per hand, per frame), never as JPEG/PNG frames, which keeps the data small and
  privacy-friendly.
* Limitations: recognition quality depends on how many samples/signers you
  collect, on lighting and camera placement, and on the similarity between
  signers' hand shapes. The letters A–E etc. are fingerspelling shapes in the
  dataset; whether a given regional ISL variant performs them identically is up
  to the person recording the data.

### Model accuracy

No number is hard-coded. `training/evaluate.py` computes everything from the
trained models. On the **synthetic CI fixture** (a deliberately separable test
set used only to prove the pipeline, see `training/synthetic_data.py`) the
reported results were:

```
STATIC  model:   accuracy 1.000  precision 1.000  recall 1.000  f1 1.000 (5 classes, 90 samples)
SEQUENCE model:  accuracy 1.000  precision 1.000  recall 1.000  f1 1.000 (6 classes, 108 samples)
```

Real-world accuracy will be lower and varies per dataset; run
`python -m training.evaluate` after training to see your own numbers.

---

## 2. Features

1. **Live camera** with hand-landmark overlay, "no hand detected" guidance.
2. **Recognition** based on MediaPipe hand landmarks (21 points × both hands),
   with position/scale-invariant feature normalisation.
3. **Static + dynamic** sign recognition (RandomForest/MLP + GRU).
4. **Dataset pipeline** — webcam collection, per-user sessions, raw landmark
   storage, label config, reference-take export.
5. **ML training & evaluation** — scripts that report accuracy / precision /
   recall / F1 / confusion-matrix, plus leave-one-user-out checks.
6. **Sentence formation** — glosses accumulate into a sentence
   (`HELLO MY NAME IS RAHUL` → *"Hello, my name is Rahul."*) with duplicate
   filtering, undo, clear.
7. **Text-to-speech** — local Windows SAPI (offline) or pyttsx3, with
   speak/stop, rate control and mute.
8. **Reverse communication** — type or speak text; it becomes a sign-gloss
   sequence with recorded-reference visuals; unknown words are fingerspelled if
   letter signs exist.
9. **Recognition stability** — confidence threshold, stable-frame counting,
   prediction cooldown (all configurable).
10. **Robustness** — one/two hands, distance/orientation changes, typical
    lighting variations (via normalised features); friendly errors instead of
    crashes for missing camera/model/mic.
11. **Privacy** — local processing, no upload of camera frames.
12. **Accessible UI** — large buttons and text, high contrast, keyboard
    shortcuts, minimal animation.

---

## 3. Architecture

```
webcam → app/camera.py ──► app/hand_detector.py  (MediaPipe Tasks HandLandmarker)
                                │  21 landmarks × 2 hands
                                ▼
                    app/feature_extractor.py   (normalise: wrist-centred,
                                                hand-size scaled, ordered hands)
                                │
                                ▼
                    app/predictor.py  ── static classifier (models/static.joblib)
                              │        ── GRU sequence model (models/sequence.pt)
                              │        smoothing: STABLE_FRAMES + CONFIDENCE_THRESHOLD
                              │        + PREDICTION_COOLDOWN
                              ▼
                    app/sentence_builder.py  → app/speech.py (TTS)
                    app/reverse_communication.py ← text/speech input

ui/main_window.py  — two-tab Tkinter UI (Sign→Text | Text→Sign)
training/ — collect_data.py, preprocess.py, train.py, evaluate.py,
            synthetic_data.py (CI/test fixture only)
```

Directory layout:

```
sign/
├── main.py                  # entry point: python main.py
├── config.json              # generated by --write-config; tune thresholds here
├── app/                     # runtime modules (camera, detector, features,
│                            #  predictor, sentence builder, TTS, mic, reverse)
├── data/
│   ├── labels.json          # vocabulary config (the ONLY place to add signs)
│   ├── raw/<user>/…         # collected landmark samples
│   ├── reference/<sign>.json# reference takes used by reverse tab visuals
│   └── processed/dataset.npz# preprocessed feature dataset
├── models/sign_model/       # hand_landmarker.task (auto-downloaded),
│                            #  static.joblib, sequence.pt, sign_index.json
├── training/                # one-off CLI scripts
├── tests/                   # pytest suite (31 tests)
└── ui/main_window.py        # Tkinter application
```

---

## 4. Installation

**Python**: tested on **3.10 – 3.14** (developed on 3.14). Use 3.11/3.12 for
maximum dependency compatibility.

```powershell
git clone <this repo>         # or copy the folder
cd sign
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
```

Dependencies:

| Package | Used for |
|---|---|
| `mediapipe` | hand landmark detection (Tasks API) |
| `opencv-python` | webcam + image drawing |
| `numpy` | features/tensors |
| `scikit-learn` | static classifier (RandomForest/MLP) + metrics |
| `torch` | GRU sequence model for dynamic signs |
| `Pillow` | Tkinter image rendering |
| `sounddevice` + `SpeechRecognition` | microphone speech input |
| `matplotlib` | confusion-matrix plots in evaluation |
| `pytest` | test suite |

TTS on Windows uses the built-in **System.Speech (SAPI)** — no extra package.
On macOS/Linux install `pip install pyttsx3` for a cross-platform backend.

The hand-landmark model (`hand_landmarker.task`, ~8 MB) is downloaded
automatically the first time the app starts; you can also fetch it manually with
the URL printed in error messages.

---

## 5. Get started

```powershell
python main.py                    # full app with webcam
python main.py --no-camera        # UI without camera (reverse mode only)
python main.py --camera 1         # different webcam
python main.py --write-config     # create config.json with all tunables
```

On first launch the app downloads `hand_landmarker.task`. Without a trained
sign model the camera screen works and detection runs, but no words are added —
you must collect data and train first (below).

Keyboard shortcuts (Sign→Text tab): `Alt+A` add detected word, `Alt+B` delete
last, `Alt+C` clear, `Alt+S` speak. In the Text→Sign tab, `Enter` shows the sign
sequence, `Ctrl+Enter` speaks the typed text.

---

## 6. Collect a dataset (webcam)

For each sign you want: show the sign, press SPACE. The take stops when your
hand leaves the frame, when you press SPACE again, or after `max-frames`.
ESC discards a take; Q quits. `--type static|dynamic` sets the sign type (if
the label is new it is added to `data/labels.json`).

Add `--auto` and you never touch the keyboard: perform the sign in front of
the camera; it records while your hand is visible and saves the take when you
lower your hand, repeating until the sample counter fills (Q quits).

```powershell
# 20 dynamic samples of HELLO, as user alice
python -m training.collect_data --label HELLO --user alice --samples 20 --type dynamic

# same, hands-free
python -m training.collect_data --label HELLO --user alice --samples 20 --type dynamic --auto

# 15 static samples of WATER
python -m training.collect_data --label WATER --samples 15 --type static

# list configured vocabulary
python -m training.collect_data --list
```

Guidelines for quality:

* Record both hands when a two-handed sign needs it.
* Vary your position, distance and slight rotation between samples.
* Collect from **several people** (pass `--user`) so the evaluation can test
  unseen users — this is the most important thing for real-world accuracy.
* Keep the sign slow and steady; dynamic signs should start and end in a
  distinct posture.

Sample format: `data/raw/<user>/<LABEL>_<timestamp>.json` — raw per-frame
landmarks only (no pixels), annotated with label, type, user and fps. The first
take of each sign is copied to `data/reference/<LABEL>.json` for the reverse
tab's visuals.

---

## 7. Preprocess, train, evaluate

```powershell
python -m training.preprocess          # raw → data/processed/dataset.npz
python -m training.train --kind both   # static (RF) + sequence (GRU)
python -m training.train --kind static --model mlp   # or MLP static model
python -m training.evaluate --kind both
```

`evaluate.py` writes `models/eval/{static,sequence}_metrics.json`,
`…_confusion_matrix.csv/.txt` and (if matplotlib is available) a `.png` plot.
When more than one user exists it also reports **leave-one-user-out**
accuracy — performance on people never seen in training.

### Add a new sign (the only place matters)

1. Append an entry to `data/labels.json`:

   ```json
   "COFFEE": { "type": "static", "en": "coffee" }
   ```

   (`collect_data --label COFFEE` can create it for you.)

2. Collect samples: `python -m training.collect_data --label COFFEE --samples 20`
3. Re-run preprocess, train, evaluate. The model index
   (`models/sign_model/sign_index.json`) is regenerated automatically.

---

## 8. How real-time recognition works

1. Each frame is scaled to your configured resolution and handed to MediaPipe.
2. Up to two hands are tracked; landmarks are normalised (wrist-centred,
   hand-size scaled, deterministic hand ordering, zero block for a missing
   hand) so the model is not tied to your position in the frame (feature 10).
3. The **static model** scores the current frame's posture; the **sequence
   model** scores the last `sequence_window` frames for dynamic signs.
4. A frame prediction is kept only if its confidence ≥ `CONFIDENCE_THRESHOLD`;
   `STABLE_FRAMES` consecutive matching frames produce a *stable sign*; the
   same sign is then ignored for `PREDICTION_COOLDOWN` frames (feature 9).
5. A sign is emitted once → added to the sentence (duplicate filtering) →
   sentence formatting → optional TTS.

Knobs live in `config.json` (create with `--write-config`):

```jsonc
{
  "confidence_threshold": 0.6,
  "stable_frames": 4,
  "prediction_cooldown": 30,
  "sequence_window": 30,
  "speech_rate": 0,
  "tts_muted": false
}
```

---

## 9. Tests

```powershell
python -m pytest tests -q
```

* feature extraction & normalisation invariants
* sentence builder (duplicates, grammar, proper nouns)
* predictor streaming logic (stability, threshold, cooldown, label changes)
* reverse-communication tokenisation (phrases, fingerspelling)
* full synthetic pipeline: synthetic landmarks → preprocess → train →
  evaluate → live predictor emits predictions

To regenerate the synthetic fixture yourself:

```powershell
python -m training.synthetic_data --outdir <tmpdir> --samples 25 --user synthetic
```

> The synthetic generator exists **only** for tests/CI to prove the plumbing;
> it is never used by the real application.

---

## 10. Limitations

* Vocabulary is small and configured manually — see the table in §1.
* ISL grammar (word order, classifiers, spatial grammar) is not modelled; the
  sentence assembler only applies light English-style formatting.
* Text→sign is a **dictionary lookup**, not automatic translation. The reverse
  tab says exactly this and shows recorded reference takes; glosses without a
  reference show a labelled placeholder.
* Accuracy depends on your data. Evaluation numbers are yours to measure.
* The letter handshapes in the starter `labels.json` follow common
  fingerspelling conventions; confirm against the ISL practice of your region.
* Speech input uses the free Google Web recognition API (requires internet).

## 11. Future improvements

* Larger crowdsourced ISL dataset + per-signer evaluation report.
* Transformer sequence model, sign spotting in continuous signing (segmentation
  of a flowing signing stream rather than one-sign-at-a-time).
* Animated 3D hand avatar (e.g. via Blender/Unity or browser WebXR) driven by
  recorded skeleton sequences in the reverse tab.
* Neural translation (text→ISL) once a proper parallel gloss corpus exists.
* Persisted per-user calibration, automatic signer detection.

---

## 12. Repro: quick full pipeline (no webcam)

```powershell
# 1) synthetic fixture into a scratch dir
python -m training.synthetic_data --outdir .scratch/raw --samples 20

# 2) point config.json at the scratch dirs, or run:
python - <<'PY'   # (or a small script)
from app.config import RecognitionConfig, ensure_dirs
from pathlib import Path
import json
p = Path("config.json")
cfg = RecognitionConfig()
cfg.raw_data_dir = ".scratch/raw"
cfg.processed_data_dir = ".scratch/processed"
cfg.static_model_path = ".scratch/static.joblib"
cfg.sequence_model_path = ".scratch/sequence.pt"
cfg.sign_index_path = ".scratch/sign_index.json"
cfg.eval_output_dir = ".scratch/eval"
p.write_text(json.dumps({"raw_data_dir": cfg.raw_data_dir,
   "processed_data_dir": cfg.processed_data_dir,
   "static_model_path": cfg.static_model_path,
   "sequence_model_path": cfg.sequence_model_path,
   "sign_index_path": cfg.sign_index_path,
   "eval_output_dir": cfg.eval_output_dir}, indent=2))
PY
python -m training.preprocess
python -m training.train --kind both
python -m training.evaluate --kind both
```

Then delete `.scratch` and restore/remove `config.json`.

---

*Privacy note: raw landmark samples are never uploaded. If you later
`git push`, `data/raw/**`, `data/reference/**`, `models/**` outputs and
`config.json` are git-ignored.*