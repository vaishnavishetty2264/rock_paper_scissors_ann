import av
import cv2
import numpy as np

from streamlit_webrtc import VideoProcessorBase

from utils.prediction import (
    predict_two_hands,
    decide_winner
)


class GameVideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # Flip like webcam mirror
        image = cv2.flip(image, 1)

        output = image.copy()

        predictions = predict_two_hands(image)

        left_gesture = None
        right_gesture = None

        left_box = None
        right_box = None

        h, w = image.shape[:2]

        for hand in predictions:

            gesture = hand["gesture"]
            confidence = hand["confidence"]
            label = hand["label"]
            points = hand["points"]

            xs = np.array(
                [lm.x for lm in points]
            ) * w

            ys = np.array(
                [lm.y for lm in points]
            ) * h

            x1 = int(max(0, np.min(xs) - 20))
            y1 = int(max(0, np.min(ys) - 20))
            x2 = int(min(w, np.max(xs) + 20))
            y2 = int(min(h, np.max(ys) + 20))

            if label == "Left":
                left_gesture = gesture
                left_box = (x1, y1, x2, y2)
            else:
                right_gesture = gesture
                right_box = (x1, y1, x2, y2)

            # Bounding Box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label Background
            cv2.rectangle(
                output,
                (x1, y1 - 65),
                (x2, y1),
                (255, 0, 0),
                -1
            )

            # Gesture
            cv2.putText(
                output,
                gesture.upper(),
                (x1 + 5, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Confidence
            cv2.putText(
                output,
                f"{confidence*100:.1f}%",
                (x1 + 5, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Winner
        if left_gesture is not None and right_gesture is not None:

            winner = decide_winner(
                left_gesture,
                right_gesture
            )

            if winner == "Draw":

                text = "DRAW"

                (tw, th), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    3
                )

                x = (w - tw) // 2

                cv2.putText(
                    output,
                    text,
                    (x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

            elif winner == "Left":

                if left_box is not None:

                    x1, y1, x2, y2 = left_box

                    cv2.putText(
                        output,
                        "WINNER",
                        (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        3
                    )

            elif winner == "Right":

                if right_box is not None:

                    x1, y1, x2, y2 = right_box

                    cv2.putText(
                        output,
                        "WINNER",
                        (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        3
                    )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )