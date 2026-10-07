import cv2
import time
import threading
from collections import deque

import torch
from PIL import Image
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from model import build_model
from config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
)

# ============================================================
# SETTINGS
# ============================================================

# Smaller padding makes the hand/fist occupy MORE of the
# classifier image. Increase to 1.25 if the hand is sometimes
# cut at the edges.
CROP_SCALE = 2.28 

# Faster recognition while retaining temporal stability.
HISTORY_SIZE = 5
MIN_STABLE_FRAMES = 3
MIN_STABLE_RATIO = 0.60
LETTER_COOLDOWN = 0.30

# A prediction must beat the second-best class by this margin.
# This helps reject uncertain frames.
MIN_MARGIN = 0.08

# How often inference is performed. 1 = every webcam frame.
INFERENCE_EVERY_N_FRAMES = 1

# ============================================================
# TEXT-TO-SPEECH
# ============================================================

tts_lock = threading.Lock()

def speak_text(text):
    """Speak text without blocking the camera/recognition loop."""
    text = text.strip()
    if not text:
        return

    if pyttsx3 is None:
        print("Text-to-speech unavailable. Install it with:")
        print("python -m pip install pyttsx3")
        return

    def worker():
        try:
            with tts_lock:
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as exc:
            print(f"TTS error: {exc}")

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise SystemExit(
        "ISL model not found. Prepare the dataset and run: "
        "python train.py"
    )

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# LOAD TRAINED ISL MODEL
# ============================================================

ckpt = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = build_model(26, pretrained=False)
model.load_state_dict(
    ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
)
model.to(device)
model.eval()

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

HAND_MODEL = "models/hand_landmarker.task"

base = python.BaseOptions(model_asset_path=HAND_MODEL)

opts = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55,
)

hands = vision.HandLandmarker.create_from_options(opts)

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(
        "Could not open webcam. Check Windows camera permissions."
    )

# Request a useful camera resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ============================================================
# RECOGNITION STATE
# ============================================================

history = deque(maxlen=HISTORY_SIZE)
last_display = "--"
last_conf = 0.0
last_margin = 0.0

sentence = ""
last_added = None
last_add_time = 0.0
last_hand_time = time.monotonic()

timestamp_ms = 0
frame_count = 0

# Used to avoid repeatedly speaking the same unchanged sentence.
last_spoken_text = ""

# ============================================================
# HELPER: STABLE PREDICTION
# ============================================================

def stable_prediction(history_values):
    if len(history_values) < MIN_STABLE_FRAMES:
        return None, 0.0

    counts = {}
    for value in history_values:
        counts[value] = counts.get(value, 0) + 1

    label, count = max(counts.items(), key=lambda item: item[1])
    ratio = count / len(history_values)

    if count >= MIN_STABLE_FRAMES and ratio >= MIN_STABLE_RATIO:
        return label, ratio

    return None, ratio


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]

        # Mirror camera.
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = hands.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33
        frame_count += 1

        if result.hand_landmarks:

            lm = result.hand_landmarks[0]

            xs = [p.x * width for p in lm]
            ys = [p.y * height for p in lm]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0

            hand_w = max_x - min_x
            hand_h = max_y - min_y

            # Tighter crop => larger hand/fist in the 224x224
            # classifier input.
            side = max(hand_w, hand_h) * CROP_SCALE

            # Prevent an extremely small crop for a clenched fist.
            min_side = min(width, height) * 0.16
            side = max(side, min_side)

            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(width, int(cx + side / 2))
            y2 = min(height, int(cy + side / 2))

            if x2 > x1 and y2 > y1:

                crop = frame[y1:y2, x1:x2]

                # Add a small border when the crop touches an edge.
                # This avoids cutting off a hand near the camera edge.
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)

                tensor = preprocess(pil_image).unsqueeze(0).to(device)

                # Run the classifier.
                with torch.inference_mode():
                    output = model(tensor)
                    prob = torch.softmax(output, dim=1)[0]

                    top_values, top_indices = torch.topk(
                        prob, k=min(2, len(CLASS_NAMES))
                    )

                conf = float(top_values[0].item())
                idx = int(top_indices[0].item())

                second_conf = (
                    float(top_values[1].item())
                    if len(top_values) > 1 else 0.0
                )

                margin = conf - second_conf

                pred = CLASS_NAMES[idx]

                last_conf = conf
                last_margin = margin
                last_display = pred
                last_hand_time = time.monotonic()

                # Only add confident + sufficiently separated predictions.
                if conf >= CONFIDENCE_THRESHOLD and margin >= MIN_MARGIN:
                    history.append(pred)

                    stable, ratio = stable_prediction(history)

                    now = time.monotonic()

                    if (
                        stable is not None
                        and stable != last_added
                        and now - last_add_time >= LETTER_COOLDOWN
                    ):
                        sentence += stable

                        last_added = stable
                        last_add_time = now

                        # Clear old votes so the next letter can be
                        # recognized immediately.
                        history.clear()

                # Draw bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw center point.
                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    5,
                    (0, 255, 255),
                    -1
                )

        else:
            # Clear temporal history when the hand disappears.
            history.clear()
            last_added = None
            last_display = "--"
            last_conf = 0.0
            last_margin = 0.0

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"ISL: {last_display}  {last_conf * 100:.1f}%",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Margin: {last_margin * 100:.1f}%",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Text: {sentence}",
            (10, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE=space  BACKSPACE=delete  C=clear  S=speak  Q=quit",
            (10, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.imshow("ISL Sign-to-Speech", frame)

        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            sentence = ""
            history.clear()
            last_added = None

        elif key == 32:
            # Space
            if sentence and not sentence.endswith(" "):
                sentence += " "
            history.clear()
            last_added = None

        elif key == 8:
            # Backspace
            sentence = sentence[:-1]
            history.clear()
            last_added = None

        elif key == ord("s"):
            # Speak button equivalent: press S.
            text_to_speak = sentence.strip()

            if text_to_speak:
                print(f"Speaking: {text_to_speak}")
                last_spoken_text = text_to_speak
                speak_text(text_to_speak)
            else:
                print("Nothing to speak.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
