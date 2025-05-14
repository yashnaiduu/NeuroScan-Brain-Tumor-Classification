# 🧠 BrainScan: Brain Tumor Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="screenshots/hero.png" alt="BrainScan Hero" width="800">
</p>

## 📋 Overview

BrainScan is an advanced deep learning system for accurate classification of brain tumors from MRI scans using Convolutional Neural Networks (CNNs). The system can classify brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary with high accuracy.

This project combines state-of-the-art deep learning techniques with a user-friendly web interface, making it accessible for medical professionals to use without extensive technical knowledge.

## ✨ Features

- **Multi-Class Classification**: Accurately identifies four different types of brain conditions
- **High Accuracy**: Trained on extensive datasets to achieve high precision and recall
- **Real-time Analysis**: Upload MRI scans and get instant classification results
- **Interactive Web Interface**: User-friendly interface with visualization capabilities
- **Performance Metrics**: Comprehensive model evaluation with confusion matrix and training curves
- **Sample Images**: Test the system with pre-loaded sample images

## 🛠️ Technologies Used

- **Backend**: Python, Flask, TensorFlow, Keras, OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Pandas, Matplotlib, Seaborn
- **Model Evaluation**: Scikit-learn

## 📊 Model Architecture

The CNN architecture consists of:
- Multiple convolutional layers with increasing filter sizes (32, 64, 128)
- Batch normalization for training stability
- MaxPooling layers for spatial dimension reduction
- Dropout (0.4) for regularization
- Dense layers for classification

<p align="center">
  <img src="screenshots/model_architecture.png" alt="Model Architecture" width="600">
</p>

## 📈 Performance Metrics

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 96.8%  |
| Precision | 95.7%  |
| Recall    | 95.2%  |
| F1 Score  | 95.4%  |

<p align="center">
  <img src="screenshots/confusion_matrix.png" alt="Confusion Matrix" width="400">
  <img src="screenshots/training_curves.png" alt="Training Curves" width="600">
</p>

## 🗂️ Dataset

The model was trained on a dataset of brain MRI scans with four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

The dataset was split into training and testing sets to evaluate the model's performance.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
