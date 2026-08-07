import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks by:
    1. Translating the wrist to the origin.
    2. Scaling using the maximum distance from the wrist.

    Parameters
    ----------
    landmarks : np.ndarray
        Shape: (63,) or (21, 3)

    Returns
    -------
    np.ndarray
        Normalized landmarks of shape (63,)
    """

    landmarks = np.array(landmarks)

    # Convert (63,) -> (21,3)
    if landmarks.shape == (63,):
        landmarks = landmarks.reshape(21, 3)

    # Wrist is landmark 0
    wrist = landmarks[0]

    # Translate wrist to origin
    landmarks = landmarks - wrist

    # Scale
    max_distance = np.max(np.linalg.norm(landmarks, axis=1))

    if max_distance > 0:
        landmarks = landmarks / max_distance

    return landmarks.flatten()