# ✋ Rock Paper Scissors Hand Gesture Recognition

A real-time Rock-Paper-Scissors hand gesture recognition application built using **MediaPipe Hand Landmarker**, **Artificial Neural Network (ANN)**, **OpenCV**, **Streamlit**, and **WebRTC**.

The system detects hand landmarks from webcam input, preprocesses the landmark coordinates, and classifies hand gestures into Rock, Paper, or Scissors using a trained ANN model.

---

## 📌 Features

- Real-time hand gesture recognition using webcam
- MediaPipe Hand Landmarker based hand detection
- 21 hand landmark extraction (63 numerical features)
- ANN-based gesture classification
- Real-time prediction using Streamlit WebRTC
- OpenCV-based gameplay mode
- Confidence score display
- One-hand gesture prediction
- Two-hand detection support for game mode

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MediaPipe Tasks API
- OpenCV
- Streamlit
- Streamlit-WebRTC
- NumPy
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Rock_Paper_Scissor/
│
├── app2.py                  # Streamlit application
├── webrtc_live.py           # Live webcam gesture prediction
├── webrtc_game.py           # Rock Paper Scissors game mode
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── final_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── hand_landmarker.task
│
├── notebooks/
│   └── *.ipynb
│
└── utils/
    ├── landmark_extraction.py
    ├── prediction.py
    └── preprocessing.py
```

---

## 🔄 Workflow

1. Collect Rock-Paper-Scissors hand gesture images.
2. Extract hand landmarks using MediaPipe Hand Landmarker.
3. Convert landmarks into numerical features:
   - 21 landmarks
   - x, y, z coordinates
   - Total 63 features
4. Normalize and scale landmark features.
5. Train an Artificial Neural Network classifier.
6. Save:
   - ANN model
   - StandardScaler
   - Label Encoder
7. Use webcam input for real-time gesture prediction.

---

## 📁 Dataset

The dataset is not included in this repository because of its size.

Dataset source:

Rock Paper Scissors Dataset (Kaggle)

```
https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset
```

Dataset structure:

```
dataset/
│
├── train/
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
├── validation/
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
└── test/
    ├── rock/
    ├── paper/
    └── scissors/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/vaishnavishetty2264/Rock_Paper_Scissors.git
```

Navigate into the project:

```bash
cd Rock_Paper_Scissors
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app2.py
```

---

## 🧠 Model Details

The model is an **Artificial Neural Network (ANN)** trained on MediaPipe hand landmark features.

### Input:

- 21 hand landmarks
- 63 numerical features:
  - x-coordinate
  - y-coordinate
  - z-coordinate

### Output Classes:

| Gesture | Class |
|---|---|
| ✋ Paper | Paper |
| ✊ Rock | Rock |
| ✌️ Scissors | Scissors |

---

## 🎮 Application Modes

### 1. Live Gesture Recognition

File:

```
webrtc_live.py
```

Features:

- Detects one hand
- Extracts landmarks
- Predicts gesture
- Displays confidence score

---

### 2. Rock Paper Scissors Game

File:

```
webrtc_game.py
```

Features:

- Supports two-hand detection
- Uses webcam interaction
- Provides game-based gesture recognition

---

## 📦 Saved Model Files

The following files are required for prediction:

```
model/
│
├── final_model.keras
├── scaler.pkl
├── label_encoder.pkl
└── hand_landmarker.task
```

---

## 🎯 Future Improvements

- Improve model accuracy
- Add more hand gesture classes
- Improve deployment optimization
- Add mobile support
- Enhance real-time FPS performance

---

## 👩‍💻 Author

**Shetty Vaishnavi**

GitHub:  
https://github.com/vaishnavishetty2264

LinkedIn:  
https://www.linkedin.com/in/shetty-vaishnavi-338508259
