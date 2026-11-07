# ISL Sign-to-Speech: Alphanumeric Recognition (1-9, A-Z)

Real-time **Indian Sign Language (ISL)** static sign recognition from a webcam. The app detects hand landmarks with **MediaPipe**, classifies the handshape with a small **PyTorch MLP**, and builds text that can be read aloud with **text-to-speech**.

- Recognizes **35 classes**: digits `1-9` and letters `A-Z`
- Supports **one-hand and two-hand** signs (many ISL letters use both hands)
- Lightweight model (~0.44M parameters, ~1.8 MB), suitable for real-time CPU use
- Temporal smoothing so one held sign types one character
- Keyboard controls for space, delete, clear and speak

> **Note on zero:** in ISL the sign for `0` is the same as the letter `O`, so there is no separate `0` class.

---

## How it works

```
Webcam / image
     |
MediaPipe Hand Landmarker  ->  up to 2 hands x 21 landmarks (x, y, z)
     |
features.py                ->  131 numbers (normalized, mirror-consistent)
     |
Residual MLP (model.py)    ->  35 class probabilities
     |
SignStabilizer             ->  smoothed, stable character
     |
On-screen text + text-to-speech
```

**Feature vector (131 values):**

| Part | Size | Description |
|---|---|---|
| Hand 0 | 63 | 21 landmarks x (x, y, z), wrist-centred, scale-normalized |
| Hand 1 | 63 | Same, for the second hand (zeros if absent) |
| Hand offset | 3 | Wrist-to-wrist vector between the two hands |
| Presence flags | 2 | Which hand slots are filled |

Hands are ordered left-to-right by wrist position, so mirrored input stays consistent. Training augmentation includes mirroring, small 3D rotations, axis stretching and landmark noise.

**Model:** linear stem -> 3 residual blocks (Linear, BatchNorm, GELU, Dropout) -> linear head with 35 outputs.

---

## Project structure

```
isl_sign_to_speech/
├── config.py              # classes, paths, hyperparameters
├── features.py            # landmarks -> features, augmentation (shared by all scripts)
├── model.py               # residual MLP
├── prepare_dataset.py     # images/videos -> landmark .npz files
├── train.py               # training, validation, test reports
├── main.py                # live webcam recognition + TTS
├── postprocess.py         # temporal smoothing / stable character emission
├── demo_simulation.py     # smoke test (no camera, model or dataset needed)
├── download_hand_model.py # downloads MediaPipe hand_landmarker.task
├── requirements.txt
└── models/
    ├── hand_landmarker.task        # MediaPipe hand model
    └── isl_alphanumeric_mlp.pth    # trained checkpoint (created by train.py)
```

---

## Requirements

- Python **3.10, 3.11 or 3.12** (MediaPipe wheels may not exist for the newest Python releases)
- A webcam (for live recognition)
- Windows, macOS or Linux

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the CPU build of PyTorch. For an NVIDIA GPU, install the matching build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run the command above.

### 4. Download the MediaPipe hand model (only if missing)

```bash
python download_hand_model.py
```

### 5. Verify the setup (optional)

```bash
python demo_simulation.py
```

Expected output: `All smoke tests passed; classes=35, feature size=131`

---

## Dataset

The dataset is not included in this repository. Use an ISL alphabet/number image dataset arranged in one of these layouts. Class folders must be named `1-9` and `A-Z` (case-insensitive).

**Pre-split:**

```
dataset/
├── Training (A-Z)/   A/ B/ ... Z/ 1/ ... 9/
├── Validation (A-Z)/ A/ B/ ...
└── Testing (A-Z)/    A/ B/ ...
```

(`train/`, `val/`, `test/` folder names also work.)

**Flat (automatic 70/15/15 split):**

```
dataset/
├── A/  B/  ...  Z/
└── 1/  2/  ...  9/
```

If your dataset has a `0` folder, add `--zero-as-o` in the next step to merge it into class `O`.

---

## Usage

### 1. Prepare: extract landmarks

```bash
python prepare_dataset.py "path/to/dataset"
```

With a `0` folder:

```bash
python prepare_dataset.py "path/to/dataset" --zero-as-o
```

This writes `data/processed/train.npz`, `val.npz`, `test.npz` and `summary.json`. Images where no hand is detected are skipped and counted. Check the printed counts, and look at `failed_per_class` in `summary.json` if a class loses many images.

### 2. Train

```bash
python train.py
```

Options:

```bash
python train.py --epochs 150 --batch-size 64 --lr 0.002
python train.py --resume            # continue from the saved checkpoint weights
```

Outputs in `models/`:

- `isl_alphanumeric_mlp.pth` best checkpoint (selected by validation macro-F1)
- `classification_report_val.json`, `classification_report_test.json`
- `confusion_matrix_val.csv`, `confusion_matrix_test.csv`
- `labels.json`

The terminal also prints the most confused class pairs.

### 3. Run live recognition

```bash
python main.py
```

Options:

```bash
python main.py --camera 1          # choose another camera
python main.py --threshold 0.80    # stricter confidence
python main.py --no-tts            # disable text-to-speech
```

**Keyboard controls**

| Key | Action |
|---|---|
| `Q` | Quit |
| `SPACE` | Insert a space |
| `BACKSPACE` | Delete the last character |
| `C` | Clear all text |
| `S` | Speak the text |

**Tips:** use good lighting, keep the full hand in view, hold each sign steady for about half a second, and lower your hand between two identical characters (for example `LL`).

---

## Results

Fill this in with your own numbers from `train.py`:

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation | _xx.x%_ | _0.xxx_ |
| Test | _xx.x%_ | _0.xxx_ |

---

## Configuration

All main settings are in `config.py`: class list, batch size, learning rate, epochs, dropout, MediaPipe confidence thresholds, and smoothing parameters (`CONFIDENCE_THRESHOLD`, `SMOOTHING_WINDOW`, `MIN_STABLE_VOTES`, `RELEASE_FRAMES`).

---

## Limitations

- This is a **static handshape classifier**. It recognizes letters and digits, not signs that depend on movement.
- **Word and sentence signs** need a sequence model (for example GRU or Transformer) trained on video, usually with body pose in addition to hands.
- Visually similar signs (for example `2`/`V`, `6`/`W`) may be confused; check the confusion matrix.
- Accuracy depends on dataset variety (backgrounds, lighting, signers, camera angles).

---

## Roadmap

- [ ] Export to ONNX for Android (MediaPipe Android Hand Landmarker + ONNX Runtime Mobile)
- [ ] Fingerspelled word suggestions with dictionary autocorrect
- [ ] Dynamic word-sign recognition with a sequence model

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `mediapipe` fails to install | Use Python 3.10-3.12 |
| `Checkpoint missing` | Run `python train.py` first |
| `Hand model missing` | Run `python download_hand_model.py` |
| `Training split has no samples for: ...` | Some classes had no usable images; check `data/processed/summary.json` |
| `Cannot open webcam` | Try `--camera 1`, and close other apps using the camera |
| Many images skipped in prepare | Images may be too small, blurry or lack a visible hand |
| `Checkpoint labels/feature size differs` | Retrain after changing `config.py` |
| No speech output | Run with `--no-tts`, or install a system TTS voice |

---

## Tech stack

Python, PyTorch, MediaPipe Tasks, OpenCV, NumPy, scikit-learn, pyttsx3

## License

Add your license here (for example MIT) and include a `LICENSE` file.

## Acknowledgements

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google
- The ISL dataset you used for training (add its name and link here)
