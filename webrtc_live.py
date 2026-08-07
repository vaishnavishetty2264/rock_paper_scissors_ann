import av
import cv2
import numpy as np

from streamlit_webrtc import VideoProcessorBase

from utils.landmark_extraction import extract_landmarks
from utils.prediction import predict_frame


class LiveVideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        image = frame.to_ndarray(format="bgr24")

        output = image.copy()

        landmarks = extract_landmarks(image)

        if landmarks is not None:

            gesture, confidence = predict_frame(landmarks)

            points = landmarks.reshape(21, 3)

            h, w = image.shape[:2]

            xs = (points[:, 0] * w).astype(np.int32)
            ys = (points[:, 1] * h).astype(np.int32)

            x1 = max(0, np.min(xs) - 20)
            y1 = max(0, np.min(ys) - 20)
            x2 = min(w, np.max(xs) + 20)
            y2 = min(h, np.max(ys) + 20)

            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label background
            cv2.rectangle(
                output,
                (x1, y1 - 60),
                (x2, y1),
                color,
                -1
            )

            # Gesture
            cv2.putText(
                output,
                gesture.upper(),
                (x1 + 10, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Confidence
            cv2.putText(
                output,
                f"{confidence * 100:.1f}%",
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )