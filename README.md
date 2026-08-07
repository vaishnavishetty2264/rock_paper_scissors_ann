# ✋ Rock Paper Scissors Hand Gesture Recognition

A real-time Rock-Paper-Scissors hand gesture recognition system built using **MediaPipe**, **Artificial Neural Networks (ANN)**, **OpenCV**, and **Streamlit**. The application detects hand landmarks from webcam input, preprocesses them, and predicts the corresponding gesture.

---

## 📌 Features

- Real-time hand gesture recognition using webcam
- Hand landmark extraction using MediaPipe
- ANN-based gesture classification
- Streamlit web application
- OpenCV-based live prediction
- Model trained on normalized hand landmark features

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- Streamlit
- NumPy
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Rock_Paper_Scissor/
│
├── app.py
├── opencv_game.py
├── opencv_live.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── final_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── hand_landmarker.task
│
├── notebooks/
│   └── *.ipynb
│
├── utils/
│   └── preprocessing.py
│   ├── prediction.py
│   ├── preprocessing.py
```

---

## 📊 Workflow

1. Collect hand gesture images.
2. Extract 21 hand landmarks using MediaPipe.
3. Normalize landmark coordinates.
4. Scale the features using StandardScaler.
5. Train an Artificial Neural Network.
6. Save the trained model and preprocessing objects.
7. Perform real-time predictions using webcam input.

---

## 📁 Dataset

The dataset is **not included** in this repository due to its size.

Download the dataset from Kaggle:

**Rock Paper Scissors Dataset**

> https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset

After downloading, place it inside the project directory:

```
dataset/
├── train/
├── validation/
└── test/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/vaishnavishetty2264/Rock_Paper_Scissors.git
```

Move into the project

```bash
cd <Rock_Paper_Scissors>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📈 Model

The model is an Artificial Neural Network trained on hand landmark coordinates extracted using MediaPipe.

Input Features:
- 21 Hand Landmarks
- 63 Numerical Features (x, y, z)

Output Classes:
- ✋ Paper
- ✊ Rock
- ✌️ Scissors

---

## 🎯 Future Improvements

- Improve prediction accuracy
- Support two-hand detection
- Deploy the application online
- Extend to additional hand gestures

---

## 👩‍💻 Author

**Shetty Vaishnavi**

GitHub: https://github.com/vaishnavishetty2264

LinkedIn: https://www.linkedin.com/in/shetty-vaishnavi-338508259