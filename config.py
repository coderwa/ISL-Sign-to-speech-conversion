from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "isl_resnet18.pth"
LABELS_PATH = MODEL_DIR / "labels.txt"

IMAGE_SIZE = 224
NUM_CLASSES = 26
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Training defaults. Increase epochs if validation accuracy has not stabilized.
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Windows-safe default
SEED = 42

# Minimum confidence shown/accepted by the webcam application.
CONFIDENCE_THRESHOLD = 0.70
