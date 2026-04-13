# 🌿 Medicinal Leaf Identification System

### Using Fine-Tuned Transfer Learning (ResNet-18)

---

## 📌 Project Overview

The **Medicinal Leaf Identification System** is a deep learning-based application designed to classify plant leaves and identify whether they belong to medicinal species.

This project leverages **transfer learning** by fine-tuning a pre-trained **ResNet-18 model** to achieve high accuracy even with a limited dataset.

---

## 🎯 Objectives

* Identify plant leaves from images
* Provide descriptive information about medicinal plants
* Build a user-friendly GUI for real-time prediction

---

## 🧠 Model Used

* **Architecture:** ResNet-18 (Residual Neural Network)
* **Technique:** Transfer Learning + Fine-tuning
* **Framework:** TensorFlow / Keras

### 🔹 Why ResNet-18?

* Solves vanishing gradient problem using skip connections
* Lightweight and faster than deeper models
* Suitable for embedded / real-time applications
* Good accuracy with smaller datasets

---

## ⚙️ Methodology

### 1. Data Collection

* Dataset collected from Kaggle / custom sources
* Includes multiple medicinal plant leaf images

### 2. Data Preprocessing

* Image resizing (160×160)
* Normalization: pixel values scaled to [-1, 1]
* Data augmentation:

  * Rotation
  * Flipping
  * Zoom

### 3. Model Training

* Pre-trained ResNet-18 used as base model
* Top layers modified for classification
* Fine-tuned on medicinal leaf dataset

### 4. Evaluation

* Accuracy and loss metrics monitored
* Validation dataset used to avoid overfitting

---

## 🖥️ GUI Application

* Built using **Tkinter**
* Features:

  * Upload leaf image
  * Display image preview
  * Show prediction with confidence
  * Display plant description

---

## 📂 Project Structure

```
LeafAppp/
│── leaf_guii.py              # GUI application
│── model.keras              # Trained model
│── descriptions.py          # Plant descriptions
│── dataset/                 # Training dataset
│── README.md                # Project documentation
```

---

## 🚀 How to Run


### 1. Install Dependencies

```
pip install tensorflow pillow numpy
```

### 2. Run Application

```
python leaf_guii.py
```

---

## 📊 Output

* If confidence < 60% → Not a leaf
* 60–80% → Non-medicinal leaf
* > 80% → Medicinal leaf + description

---

## 📈 Applications

* Agriculture and farming
* Herbal medicine identification
* Educational tools
* Smart plant recognition systems

---

## 🔮 Future Enhancements

* Mobile app integration
* Real-time camera detection
* Larger dataset for improved accuracy
* Deployment on embedded systems

---

## 🧾 Conclusion

This project demonstrates the effectiveness of **transfer learning using ResNet-18** in solving real-world image classification problems.
It provides a simple yet powerful solution for identifying medicinal plants using leaf images.

---

## 👨‍💻 Author

**Sindhu D M ,Spoorthi H S ,Spoorthi V N**
Electronics & Communication Engineering Students

---

## ⭐ Acknowledgment

* Kaggle datasets
* TensorFlow & Keras libraries
* Open-source community

---
