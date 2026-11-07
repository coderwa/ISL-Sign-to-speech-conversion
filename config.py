"""Central configuration for ISL static alphanumeric sign recognition (1-9, A-Z).

NOTE: In Indian Sign Language the sign for "0" is the same handshape as the
letter "O", so there are 35 distinct classes: 1-9 and A-Z.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"      # train.npz / val.npz / test.npz live here
MODEL_DIR = BASE_DIR / "models"
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
CHECKPOINT_PATH = MODEL_DIR / "isl_alphanumeric_mlp.pth"
LABELS_PATH = MODEL_DIR / "labels.json"

# ---- classes ---------------------------------------------------------------
DIGITS = [str(i) for i in range(1, 10)]                       # 1..9 (0 == O in ISL)
LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
CLASS_NAMES = DIGITS + LETTERS
NUM_CLASSES = len(CLASS_NAMES)                                 # 35

# ---- features (see features.py) -------------------------------------------
NUM_LANDMARKS = 21
MAX_HANDS = 2                                                  # many ISL letters are two-handed
HAND_FEATURES = NUM_LANDMARKS * 3                              # 63 per hand
# [hand0 (63) | hand1 (63) | wrist1-wrist0 offset (3) | presence flags (2)]
INPUT_SIZE = MAX_HANDS * HAND_FEATURES + 3 + MAX_HANDS         # 131

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---- training --------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0            # reliable on Windows
PATIENCE = 25
LABEL_SMOOTHING = 0.10
DROPOUT = 0.25
HIDDEN_SIZE = 256
NUM_BLOCKS = 3

# ---- MediaPipe -------------------------------------------------------------
STATIC_DETECTION_CONFIDENCE = 0.30   # offline dataset images
MIN_DETECTION_CONFIDENCE = 0.50      # live webcam
MIN_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

# ---- live inference --------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 9
MIN_STABLE_VOTES = 5
RELEASE_FRAMES = 4         # frames without a hand before the same char may repeat


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
