"""Capture normalized two-hand webcam examples for local calibration."""
from pathlib import Path
import cv2
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
from feature_extraction import FEATURE_NAMES, assign_handedness, extract_two_hand_features

ROOT = Path("webcam_dataset")
MODEL_PATH = Path("hand_landmarker.task")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TARGET = 120

def main():
    opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)), num_hands=2)
    detector = vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Camera unavailable. Check camera permissions or camera index.")
    ix, captured, notice = 0, 0, "Ready"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read from webcam.")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
            left, right = assign_handedness(result)
            features = extract_two_hand_features(left, right)
            letter = LETTERS[ix]
            folder = ROOT / letter
            folder.mkdir(parents=True, exist_ok=True)
            csv_path = folder / "samples.csv"
            count = max(0, sum(1 for _ in csv_path.open(encoding="utf-8")) - 1) if csv_path.exists() else 0
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 155), (23, 27, 38), -1)
            cv2.putText(frame, "ISL DATA COLLECTION", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
            cv2.putText(frame, f"Current letter: {letter}     Samples: {count} / {TARGET}", (22, 75), cv2.FONT_HERSHEY_SIMPLEX, .7, (80,220,180), 2)
            cv2.putText(frame, "Hold sign; move slightly between captures. SPACE capture | N next | B previous | Q quit", (22, 120), cv2.FONT_HERSHEY_SIMPLEX, .48, (230,230,230), 1)
            cv2.putText(frame, "BOTH HANDS READY" if features is not None else "SHOW BOTH HANDS", (22, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, .7, (80,220,180) if features is not None else (0,170,255), 2)
            if notice:
                cv2.putText(frame, notice, (frame.shape[1]-190, 38), cv2.FONT_HERSHEY_SIMPLEX, .65, (80,220,180), 2)
            cv2.imshow("ISL Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            notice = ""
            if key in (ord('q'), ord('Q')): break
            if key in (ord('n'), ord('N')): ix = (ix+1) % 26
            elif key in (ord('b'), ord('B')): ix = (ix-1) % 26
            elif key == 32:
                if features is None:
                    notice = "Need both hands"
                else:
                    exists = csv_path.exists()
                    pd.DataFrame([list(features)+[letter]], columns=FEATURE_NAMES+["label"]).to_csv(csv_path, mode="a", header=not exists, index=False)
                    notice = "CAPTURED"
    finally:
        cap.release(); cv2.destroyAllWindows(); detector.close()

if __name__ == "__main__": main()
