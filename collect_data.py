import cv2
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


MODEL_PATH = "hand_landmarker.task"
DATASET_DIR = "dataset"

SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "PLEASE",
    "STOP",
    "HELP",
    "GOOD",
    "BAD",
    "I_LOVE_YOU"
]

os.makedirs(DATASET_DIR, exist_ok=True)


# -----------------------------
# MEDIAPIPE
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# SELECT SIGN
# -----------------------------

print("\n==============================")
print("       ISL DATA COLLECTOR")
print("==============================\n")

for i, sign in enumerate(SIGNS, 1):
    print(f"{i}. {sign}")

while True:

    try:
        choice = int(input("\nEnter sign number: "))

        if 1 <= choice <= len(SIGNS):
            selected_sign = SIGNS[choice - 1]
            break

        print("Enter a number from 1 to 10.")

    except ValueError:
        print("Please enter a number.")


print(f"\nSelected sign: {selected_sign}")


# -----------------------------
# CSV
# -----------------------------

csv_path = os.path.join(
    DATASET_DIR,
    f"{selected_sign}.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline=""
)

writer = csv.writer(csv_file)


# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    csv_file.close()
    detector.close()
    exit()


print("\nCamera started.")
print("Show your hand clearly.")
print("SPACE = capture")
print("Q = quit\n")


sample_count = 0
flash_counter = 0


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    # -------------------------
    # DRAW LANDMARKS
    # -------------------------

    hand_detected = len(result.hand_landmarks) > 0

    if hand_detected:

        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

    # -------------------------
    # UI
    # -------------------------

    cv2.putText(
        frame,
        f"SIGN: {selected_sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"SAMPLES: {sample_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    if hand_detected:

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "SPACE = SAVE SAMPLE",
        (20, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = QUIT",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # -------------------------
    # FLASH AFTER CAPTURE
    # -------------------------

    if flash_counter > 0:

        cv2.putText(
            frame,
            "SAVED!",
            (frame.shape[1] // 2 - 100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        flash_counter -= 1


    cv2.imshow(
        "ISL Data Collector",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # -------------------------
    # SPACE = SAVE
    # -------------------------

    if key == 32:

        if not hand_detected:

            print("NO HAND DETECTED - sample not saved.")

        else:

            # First detected hand
            hand = result.hand_landmarks[0]

            wrist = hand[0]

            features = []

            # Normalize relative to wrist
            for landmark in hand:

                x = landmark.x - wrist.x
                y = landmark.y - wrist.y
                z = landmark.z - wrist.z

                features.extend([
                    x,
                    y,
                    z
                ])

            writer.writerow(
                features + [selected_sign]
            )

            csv_file.flush()

            sample_count += 1
            flash_counter = 10

            print(
                f"Saved {selected_sign} sample #{sample_count}"
            )


    # -------------------------
    # Q = QUIT
    # -------------------------

    if key == ord("q"):

        break


# -----------------------------
# CLEANUP
# -----------------------------

csv_file.close()

cap.release()

cv2.destroyAllWindows()

detector.close()


print("\n==============================")
print("DATA COLLECTION COMPLETE")
print("==============================")
print(f"Sign: {selected_sign}")
print(f"Samples: {sample_count}")
print(f"File: {csv_path}")
print("==============================")