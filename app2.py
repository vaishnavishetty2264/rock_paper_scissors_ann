import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, webrtc_streamer
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import av
from PIL import Image

import sys
import tensorflow as tf
from tensorflow import keras
from webrtc_live import LiveVideoProcessor
from webrtc_game import GameVideoProcessor
from utils.prediction import predict_image



# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Rock Paper Scissors",
    page_icon="✋",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#1565C0;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    "<div class='title'>✋ Rock Paper Scissors Gesture Recognition</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>MediaPipe Hands + Artificial Neural Network</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# ICONS
# ==========================================================

gesture_icons = {

    "rock":"✊",

    "paper":"✋",

    "scissors":"✌️"

}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Project")

st.sidebar.success("Artificial Neural Network")

st.sidebar.info("MediaPipe Hands")

st.sidebar.markdown("---")

st.sidebar.write("### Gesture Classes")

for g in gesture_icons:

    st.sidebar.write(
        gesture_icons[g],
        g.capitalize()
    )

st.sidebar.markdown("---")

mode = st.sidebar.radio(

    "Select Mode",

    [

        "Upload Image",

        "Live Webcam",

        "Game Mode"

    ]

)

# ==========================================================
# IMAGE MODE
# ==========================================================

if mode == "Upload Image":

    st.subheader("Upload an Image")

    uploaded = st.file_uploader(

        "Choose an Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )

    if uploaded is not None:

        image = Image.open(uploaded)

        image.save("temp_image.png")

        gesture, confidence = predict_image("temp_image.png")

        if gesture is None:

            st.error("No Hand Detected.")

        else:

            col1, col2 = st.columns([2,1])

            with col1:

                st.image(image)

            with col2:

                st.success("Prediction")

                st.metric(
                    "Gesture",
                    f"{gesture_icons[gesture]} {gesture.capitalize()}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                st.progress(float(confidence))

                st.write("### Prediction Status")

                if confidence > 0.90:

                    st.success("Very High Confidence")

                elif confidence > 0.75:

                    st.info("Good Confidence")

                elif confidence > 0.50:

                    st.warning("Moderate Confidence")

                else:

                    st.error("Low Confidence")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": []
    }
)
# ==========================================================
# LIVE WEBCAM MODE
# ==========================================================

from streamlit_webrtc import webrtc_streamer
from webrtc_live import LiveVideoProcessor

if mode == "Live Webcam":

    st.subheader("Live Gesture Recognition")

    st.info(
        "Allow camera access and show one hand to the camera."
    )

    webrtc_streamer(
        key="rps",
        video_processor_factory=LiveVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("✊", "Rock")

    with c2:
        st.metric("✋", "Paper")

    with c3:
        st.metric("✌️", "Scissors")
                   


# ==========================================================
# GAME MODE
# ==========================================================

from webrtc_game import GameVideoProcessor

if mode == "Game Mode":

    st.subheader("Rock Paper Scissors Game")

    st.info(
        "Allow camera access and show two hands to the camera."
    )

    webrtc_streamer(
        key="game",
        video_processor_factory=GameVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("✊", "Rock")

    with c2:
        st.metric("✋", "Paper")

    with c3:
        st.metric("✌️", "Scissors")
# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

<b>Rock Paper Scissors Gesture Recognition</b><br>

Developed using <b>MediaPipe Hands</b> +
<b>Artificial Neural Network (ANN)</b> +
<b>Streamlit</b>

</div>
""",
unsafe_allow_html=True
)