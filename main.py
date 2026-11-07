"""Live webcam ISL alphanumeric recognition: MediaPipe (up to 2 hands) + MLP + TTS."""
import argparse, threading, time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import build_model
from postprocess import SignStabilizer
from features import hands_to_features, pixel_hands_from_mediapipe
from config import (HAND_LANDMARKER_PATH, CHECKPOINT_PATH, CLASS_NAMES, INPUT_SIZE, MAX_HANDS,
                    MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
                    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW, MIN_STABLE_VOTES, RELEASE_FRAMES)


def load_model(device, checkpoint_path):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    names = ck.get("class_names", CLASS_NAMES)
    if list(names) != CLASS_NAMES: raise RuntimeError("Checkpoint labels differ from config.py; retrain the model.")
    if ck.get("input_size") != INPUT_SIZE: raise RuntimeError("Checkpoint feature size differs from config.py; retrain the model.")
    model = build_model(num_classes=len(names), **ck["model_config"])
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()
    return model, names


def speak(text):
    """Speak in a background thread so the video never freezes."""
    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init(); engine.setProperty("rate", 155); engine.say(text); engine.runAndWait()
        except Exception as e:
            print(f"TTS unavailable: {e}")
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=str(CHECKPOINT_PATH)); ap.add_argument("--hand-model", default=str(HAND_LANDMARKER_PATH))
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD); ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()
    if not Path(args.model).exists(): raise SystemExit(f"Checkpoint missing: {args.model}. Train the model first.")
    if not Path(args.hand_model).exists(): raise SystemExit(f"Hand model missing: {args.hand_model}. Run python download_hand_model.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, names = load_model(device, args.model)
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=args.hand_model),
        running_mode=vision.RunningMode.VIDEO, num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): landmarker.close(); raise SystemExit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    stabilizer = SignStabilizer(names, SMOOTHING_WINDOW, MIN_STABLE_VOTES, args.threshold, release_frames=RELEASE_FRAMES)
    sentence = ""; raw_label = "--"; confidence = 0.0; last_ts = 0
    print("Live ISL recognition started. Q=quit, SPACE=space, BACKSPACE=delete, C=clear, S=speak")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1); h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), last_ts + 1); last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            probabilities = None
            if result.hand_landmarks:
                hands, mask = pixel_hands_from_mediapipe(result.hand_landmarks, w, h)
                x = torch.from_numpy(hands_to_features(hands, mask)).unsqueeze(0).to(device)
                with torch.inference_mode(): probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                i = int(np.argmax(probabilities)); raw_label = names[i]; confidence = float(probabilities[i])
                for lm in result.hand_landmarks:
                    xs = [int(p.x * w) for p in lm]; ys = [int(p.y * h) for p in lm]
                    cv2.rectangle(frame, (max(0, min(xs) - 20), max(0, min(ys) - 20)), (min(w - 1, max(xs) + 20), min(h - 1, max(ys) + 20)), (0, 220, 80), 2)
            else:
                raw_label = "--"; confidence = 0.0
            stable = stabilizer.update(probabilities)
            if stable.emitted:
                sentence += stable.emitted; print(f"Recognized: {stable.emitted} ({stable.confidence:.2f})")
            cv2.rectangle(frame, (0, 0), (w, 115), (10, 10, 10), -1)
            cv2.putText(frame, f"Raw: {raw_label}  {confidence:.1%}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 240, 120), 2)
            cv2.putText(frame, f"Stable: {stable.dominant or '--'}  {stable.confidence:.1%}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
            cv2.putText(frame, f"Text: {sentence[-55:]}", (12, 100), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(frame, "Q quit | SPACE space | BACKSPACE delete | C clear | S speak", (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            cv2.imshow("ISL 1-9 / A-Z Recognition", frame); key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): sentence = ""; stabilizer.reset()
            elif key == 32: sentence += " "; stabilizer.reset()
            elif key == 8: sentence = sentence[:-1]; stabilizer.reset()
            elif key == ord('s') and sentence.strip() and not args.no_tts: speak(sentence)
    finally:
        cap.release(); cv2.destroyAllWindows(); landmarker.close()


if __name__ == "__main__":
    main()
