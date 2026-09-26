# hand-sign-by-one-hand
Built something I’ve been experimenting with lately — an Indian Sign Language (ISL)
# Indian Sign Language (ISL) Hand Sign Recognition

A Python-based computer vision project that uses a webcam to recognize **Indian Sign Language (ISL) alphabet hand signs** and convert them into letters in real time.

This project is an experimental prototype built using **Python, OpenCV, MediaPipe and Machine Learning**.

## About

I wanted to experiment with something beyond a basic ML tutorial, so I used an existing ISL dataset and built a real-time webcam recognition system around it.

The current version mainly focuses on **one-hand alphabet signs (A–Z)**.

The project is still under development. Real-time performance can be limited depending on the laptop, camera and lighting conditions.

## Demo

The system uses a webcam to:

1. Detect the hand using MediaPipe
2. Extract hand landmarks
3. Convert the landmarks into numerical features
4. Use a trained Machine Learning model to predict the sign
5. Display the predicted letter on the screen
6. Build the detected letters into text

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Joblib

## Dataset

The initial training data comes from the **RealSign Indian Sign Language Dataset**.

Dataset:
https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

The dataset contains ISL alphabet images that were processed into hand-landmark features for training.

## Machine Learning

The project currently uses a **Random Forest Classifier** for hand-sign classification.

The original dataset test split achieved around **99% accuracy**, but this should not be interpreted as real-world webcam accuracy.

Webcam performance can be affected by:

* Lighting
* Camera quality
* Hand position
* Background
* Hand orientation
* Distance from camera
* Differences between dataset images and real webcam input

Improving real-world accuracy is one of the next goals of the project.

## Project Structure

```text
hand sign/
│
├── collect_data.py
├── prepare_dataset.py
├── prepare_dataset_2hand.py
├── train_model.py
├── train_model_2hand.py
│
├── live_recognition.py
├── live_recognition_2hand.py
├── isl_translator.py
├── sign_language.py
│
├── test_hand.py
│
├── hand_landmarker.task
├── training_data.csv
├── training_data_2hand.csv
├── isl_model.pkl
├── isl_model_2hand.pkl
│
└── dataset/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd "hand sign"
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure the MediaPipe hand landmark model is present:

```text
hand_landmarker.task
```

## Run the Project

For the one-hand recognition system:

```bash
python live_recognition.py
```

For the two-hand experiment:

```bash
python live_recognition_2hand.py
```

A webcam window should open and display the detected sign.

## Controls

| Key         | Action                    |
| ----------- | ------------------------- |
| `SPACE`     | Add a space               |
| `BACKSPACE` | Delete the last character |
| `C`         | Clear the sentence        |
| `Q`         | Quit                      |

## Current Status

**Prototype / Work in Progress**

The current system can recognize static alphabet signs, but it is not yet a complete ISL translation system.

The main focus right now is improving:

* Real-time FPS
* Webcam accuracy
* Prediction stability
* Hand landmark normalization
* Training data quality
* One-hand recognition
* Duplicate-letter prevention

## Future Plans

* Collect more webcam data
* Combine multiple ISL datasets
* Improve landmark normalization
* Compare different ML models
* Improve real-time performance
* Add confidence-based prediction
* Add prediction smoothing
* Add word and sentence formation
* Add text-to-speech
* Explore dynamic ISL signs using video/temporal models
* Build a more polished user interface

## Important Note

This project is an **experimental learning project**, not a production-ready ISL translator.

Static alphabet recognition is only one part of sign language. A complete sign-language translation system would need to handle dynamic signs, movement, both hands, facial expressions, body pose, context and continuous sentences.

## Credits

Dataset:

**RealSign — Indian Sign Language Dataset**
https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

Built as a personal AI/ML and computer vision experiment.

---

**Built with Python, OpenCV, MediaPipe and Machine Learning.**
