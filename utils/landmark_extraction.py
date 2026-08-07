import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================================
# Load MediaPipe Tasks Hand Landmarker
# ==========================================================

MODEL_PATH = "model/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    #running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    # min_hand_detection_confidence=0.5,
    # min_hand_presence_confidence=0.5,
    # min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ==========================================================
# Single Hand Extraction
# ==========================================================

def extract_landmarks(image):
    """
    Returns
    -------
    np.ndarray or None
        Shape = (63,)
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if len(result.hand_landmarks) == 0:
        return None

    hand = result.hand_landmarks[0]

    landmarks = []

    for lm in hand:
        landmarks.extend([
            lm.x,
            lm.y,
            lm.z
        ])

    return np.array(landmarks, dtype=np.float32)


# ==========================================================
# Two Hand Extraction
# ==========================================================

def extract_two_hands(image):
    """
    Returns
    -------
    list

    [
        {
            "label": "Left",
            "landmarks": ndarray(63,),
            "points": hand_landmarks
        },
        ...
    ]
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    detected_hands = []

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):

        landmarks = []

        for lm in hand_landmarks:
            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        detected_hands.append({

            "label": handedness[0].category_name,

            "landmarks": np.array(
                landmarks,
                dtype=np.float32
            ),

            "points": hand_landmarks

        })

    return detected_hands