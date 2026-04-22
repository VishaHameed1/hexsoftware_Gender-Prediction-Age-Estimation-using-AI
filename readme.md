# 📊 Gender Prediction & Age Estimation using AI

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

Gender Prediction & Age Estimation using AI is a computer vision system that analyzes facial features to predict **gender**, **age range**, and optionally **emotion** in real-time using OpenCV and deep learning models.

It works on both **webcam live feed** and **static images**, making it suitable for AI demos and portfolio projects.

---

## ✨ Features

### 🧠 Core Features

* 🚻 Gender Detection (Male / Female + confidence score)
* 📅 Age Estimation (e.g., 0-2, 4-6, 25-32, 60+)
* 😊 Emotion Recognition (optional module)
* 🎥 Real-time webcam processing (15–20 FPS)
* 🖼️ Image-based analysis

### ⚙️ Technical Features

* 👥 Multi-face detection support
* 📊 Prediction confidence display
* 🎨 Colored bounding boxes with labels
* 📸 Auto-save annotated results
* 🔍 High accuracy pre-trained CNN models

---

## 🧠 How It Works

1. Detect face using OpenCV (Haar Cascade or DNN)
2. Extract face region (ROI)
3. Preprocess image (resize, normalize)
4. Pass through pre-trained age/gender model
5. Get predictions:

   * Gender: Male / Female
   * Age: Age bucket (0–2, 4–6, etc.)
6. Display results on frame

---

## 📦 Installation

### 1️⃣ Check Python version

```bash
python --version
```

---

### 2️⃣ Install dependencies

```bash id="x3m1qz"
pip install opencv-python numpy
```

---

### 3️⃣ (Optional) Install deep learning support

```bash id="v9k2sa"
pip install tensorflow keras
```

---

## 📁 Project Structure

```id="p8k1ld"
Gender-Age-Prediction-AI/
│── models/
│     ├── age_net.caffemodel
│     ├── gender_net.caffemodel
│     ├── deploy_age.prototxt
│     ├── deploy_gender.prototxt
│
│── images/
│── outputs/
│── main.py
│── utils.py
│── README.md
```

---

## ▶️ Usage

### 🔹 Run Webcam Detection

```bash id="2m8qwe"
python main.py
```

---

### 🔹 Run Image Prediction

```bash id="k1l9rt"
python main.py --image path/to/image.jpg
```

---

## 💻 Example Output

```text id="o3p8zx"
Face Detected
Gender: Male (92.4%)
Age: 25-32
Emotion: Happy 😊
```

---

## 📊 Applications

* 🎯 Smart surveillance systems
* 🏬 Retail customer analytics
* 📱 Social media filters
* 🧠 AI demo projects
* 🏫 Academic final year projects

---

## 🔧 Model Details

* **Age Model**: CNN trained on Adience dataset
* **Gender Model**: CNN classifier
* **Framework**: OpenCV DNN module
* **Input Size**: 227×227 RGB images

---

## 🚀 Future Improvements

* 🔥 Upgrade to YOLOv8 face detection
* 🎭 Improve emotion detection accuracy
* 🌐 Deploy as Flask web app
* 📱 Mobile integration (Android/iOS)
* ☁️ Cloud API deployment

---

## 🤝 Contribution

1. Fork repository
2. Create feature branch
3. Commit changes
4. Submit pull request

---

## 📜 License

This project is licensed under the MIT License.

---
