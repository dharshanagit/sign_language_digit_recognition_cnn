# Sign_Language_Digit_Recognition_CNN
American Sign Language Digit Recognition using CNN

## 📌 Project Overview
This project is a real-time American Sign Language (ASL) Digit Recognition system developed using Convolutional Neural Networks (CNN) and OpenCV.

The system captures hand gestures through a webcam, processes the hand region using image preprocessing techniques, and predicts ASL digits (0–9) in real time.

---

## 🚀 Features
- Real-time webcam prediction
- ASL digit recognition (0–9)
- CNN-based deep learning model
- Image preprocessing using OpenCV
- Thresholding for hand segmentation
- Live prediction display with confidence score

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Workflow

```text
Dataset Collection
        ↓
Image Preprocessing
(Grayscale, Blur, Threshold)
        ↓
Data Augmentation
        ↓
CNN Model Training
        ↓
Model Saving
        ↓
Real-Time Webcam Detection
        ↓
ASL Digit Prediction
```

---

## 🧠 Model Architecture
The CNN model contains:
- Convolution Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layer
- Softmax Output Layer

---

## 📸 Image Preprocessing Steps
1. Capture webcam frame
2. Extract Region of Interest (ROI)
3. Convert image to grayscale
4. Apply Gaussian Blur
5. Apply Thresholding
6. Resize image to 64×64
7. Normalize pixel values
8. Predict using CNN model

---

## 📊 Dataset
The dataset contains hand gesture images for digits:
```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Folder structure:
```text
dataset/
   0/
   1/
   2/
   ...
   9/
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2️⃣ Train the Model
```bash
python ASL_Digit.py
```

### 3️⃣ Run Real-Time Prediction
```bash
python realtime.py
```

---

## 🎯 Output
- Detects ASL hand gesture digits
- Displays predicted digit on webcam screen
- Shows threshold processed image

---

## 📌 Applications
- Sign language assistance
- Human-computer interaction
- Gesture-based systems
- Accessibility technology
- Educational tools

---

## 🔮 Future Enhancements
- Alphabet recognition (A–Z)
- Dynamic gesture recognition
- Improved accuracy with larger dataset
- Mobile deployment
- Speech output integration

---

## 📈 Conclusion
This project demonstrates how Deep Learning and Computer Vision can be used together to build a real-time ASL digit recognition system. The CNN model successfully learns hand gesture patterns and predicts ASL digits accurately through live webcam input.

---
