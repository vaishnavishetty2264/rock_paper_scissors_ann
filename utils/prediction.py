# import joblib
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# from utils.landmark_extraction import extract_landmarks
# from utils.preprocessing import normalize_landmarks

# # Load saved files
# MODEL_PATH = "models/emotion_model.keras"
# SCALER_PATH = "models/scaler.pkl"

# model = load_model(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# CLASS_NAMES = [
#     "paper",
#     "rock",
#     "scissors"
# ]


# def predict_image(image_path):
#     """
#     Predict gesture from an image.
#     """

#     image = cv2.imread(image_path)

#     landmarks = extract_landmarks(image)

#     if landmarks is None:
#         return "No Hand Detected"

#     landmarks = normalize_landmarks(landmarks)

#     landmarks = scaler.transform([landmarks])

#     prediction = model.predict(landmarks, verbose=0)

#     predicted_class = np.argmax(prediction)

#     confidence = np.max(prediction)

#     return CLASS_NAMES[predicted_class], confidence


import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.landmark_extraction import (
    extract_landmarks,
    extract_two_hands
)

from utils.preprocessing import normalize_landmarks


# ==========================================================
# Load Saved Files
# ==========================================================

MODEL_PATH = "model/final_model.keras"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

CLASS_NAMES = [
    "paper",
    "rock",
    "scissors"
]


# ==========================================================
# Predict from Landmark Vector
# ==========================================================

def predict_gesture(landmarks):
    """
    Predict gesture from extracted landmarks.

    Parameters
    ----------
    landmarks : ndarray (63,)

    Returns
    -------
    gesture, confidence
    """

    landmarks = normalize_landmarks(landmarks)

    landmarks = scaler.transform([landmarks])

    prediction = model.predict(
        landmarks,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return CLASS_NAMES[predicted_class], float(confidence)


# ==========================================================
# Predict from Image Path
# ==========================================================

import os

def predict_image(image_path):
    print("Image path:", image_path)
    print("File exists:", os.path.exists(image_path))

    image = cv2.imread(image_path)

    print("Image:", image)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    landmarks = extract_landmarks(image)

    if landmarks is None:
        return None, None

    return predict_gesture(landmarks)
# ==========================================================
# Predict from Webcam Frame
# ==========================================================

def predict_frame(frame):
    """
    Predict gesture from webcam frame.
    """

    if landmarks is None:
        return None, None

    return predict_gesture(landmarks)


# ==========================================================
# Predict Two Hands (Game Mode)
# ==========================================================

def predict_two_hands(frame):
    """
    Predict both hands.

    Returns
    -------
    List of dictionaries
    """

    detected_hands = extract_two_hands(frame)

    predictions = []

    for hand in detected_hands:

        gesture, confidence = predict_gesture(
            hand["landmarks"]
        )

        predictions.append({

            "label": hand["label"],

            "gesture": gesture,

            "confidence": confidence,

            "points": hand["points"]

        })

    return predictions


# ==========================================================
# Decide Winner
# ==========================================================

def decide_winner(left_gesture, right_gesture):
    """
    Decide winner of Rock-Paper-Scissors.
    """

    if left_gesture == right_gesture:
        return "Draw"

    rules = {

        "rock": "scissors",

        "paper": "rock",

        "scissors": "paper"

    }

    if rules[left_gesture] == right_gesture:
        return "Left"

    return "Right"