"""Download the MediaPipe hand landmarker model."""
from pathlib import Path
from urllib.request import urlretrieve

out = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
out.parent.mkdir(exist_ok=True)
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
print("Downloading MediaPipe hand landmarker...")
urlretrieve(url, out)
print(f"Saved: {out}")
