# ISL Sign-to-Speech — Retraining Version

This project replaces the previous ASL-only checkpoint with a **trainable Indian Sign Language (ISL) A–Z classifier**.

The ZIP does **not** contain the ISL dataset or trained weights. Those files are intentionally not bundled. The training script creates `models/isl_resnet18.pth` after the dataset is prepared.

## 1. Recommended Python

On Windows, use **Python 3.11** for the smoothest compatibility with PyTorch/MediaPipe.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get an ISL dataset

A good starting point is the **RealSign Indian Sign Language Alphabet Dataset**. It contains 26 A–Z classes and separate training/testing/validation folders. See:
https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

Download/extract `Dataset.zip` from that repository. Do **not** put the whole dataset into this project ZIP.

Expected source structure:

```text
Dataset/
├── Training (A-Z)/
│   ├── A/
│   ├── B/
│   └── ... Z/
├── Testing (A-Z)/
│   ├── A/
│   └── ... Z/
└── Validation (A-Z)/
    ├── A/
    └── ... Z/
```

The RealSign dataset documentation reports about 700 training, 200 testing and 100 validation images per class. It is specifically an ISL fingerspelled-alphabet dataset.

## 3. Prepare the dataset

From this project folder:

```powershell
python prepare_dataset.py "C:\path\to\Dataset"
```

This creates:

```text
data/
├── train/A ... Z
├── val/A ... Z
└── test/A ... Z
```

## 4. Train the ISL model

```powershell
python train.py --epochs 15
```

For a fresh run:

```powershell
python train.py --fresh --epochs 15
```

The script uses ResNet18 transfer learning, augmentation, label smoothing, validation tracking and a saved best checkpoint.

After training:

```text
models/
├── isl_resnet18.pth
├── labels.txt
└── checkpoint.pth
```

The best model is selected by validation accuracy. **Do not judge the model only from training accuracy**; use the held-out test accuracy and real webcam testing.

## 5. Download MediaPipe hand model

```powershell
python download_hand_model.py
```

This creates:

```text
models/hand_landmarker.task
```

## 6. Run real-time ISL recognition

```powershell
python main.py
```

The program:

- Opens the webcam.
- Detects one hand with MediaPipe.
- Makes a square crop instead of stretching a rectangular crop.
- Runs the trained ISL A–Z classifier.
- Requires a confidence threshold before accepting predictions.
- Smooths several consecutive predictions before adding a letter.
- Builds text from recognized letters.

Controls:

- `SPACE` — add a space
- `BACKSPACE` — delete the last character
- `C` — clear text
- `Q` — quit

## Important limitation

This version recognizes **ISL fingerspelled alphabet signs**, not complete natural ISL words or sentences. For a true sign-to-speech system, the next model should be trained on **dynamic/isolated ISL word videos** rather than only A–Z images.

Also, not every ISL sign has the same semantics as ASL. The model therefore must be trained on genuine ISL examples; an ASL checkpoint should not be relabeled as ISL.

## If accuracy is still poor

1. Make sure the dataset actually contains ISL signs, not ASL.
2. Train for more epochs if validation accuracy is still improving.
3. Test with lighting and backgrounds different from the training set.
4. Keep the whole hand visible in the camera frame.
5. For a production system, add more signers and real webcam samples and fine-tune the model with them.


## Improved real-time recognition

The improved `main.py` includes:
- Tighter hand cropping (`CROP_SCALE = 1.18`) so a clenched fist appears larger to ResNet18.
- A minimum crop size to prevent very small fist crops.
- Faster temporal stabilization (3 stable frames instead of the previous 6-frame/5-vote delay).
- Confidence + top-two prediction margin filtering to reject ambiguous frames.
- Faster letter insertion with a 0.30 second cooldown.
- Non-blocking Windows text-to-speech using `pyttsx3`.
- Press `S` to speak the currently generated text.
- Press `SPACE`, `BACKSPACE`, `C`, and `Q` as before.

### Important accuracy note

The RealSign dataset used here is an **ISL A-Z fingerspelling/alphabet dataset**. The model therefore predicts letters, then the application builds words from those letters. It is not a natural-language ISL word recognition model.

For true direct word recognition, a separate model trained on isolated/dynamic ISL word samples or videos is required.
