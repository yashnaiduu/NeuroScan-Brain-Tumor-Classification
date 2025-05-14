<div align="center">
  <img src="assets/brainscan-logo.png" alt="BrainScan Logo" width="200"/>
  <h1>🧠 BrainScan: Brain Tumor Classification</h1>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.x-green.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
  [![Stars](https://img.shields.io/github/stars/yourusername/brain-tumor-classification?style=for-the-badge&logo=github)](https://github.com/yourusername/brain-tumor-classification/stargazers)
  
  <p>Advanced deep learning system for accurate classification of brain tumors from MRI scans</p>
  
  <a href="#demo">View Demo</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#model">Model</a> •
  <a href="#results">Results</a> •
  <a href="#team">Team</a>
  
  <img src="assets/demo.gif" alt="BrainScan Demo" width="700"/>
</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Live Demo](#-live-demo)
- [Screenshots](#-screenshots)
- [Technologies](#-technologies)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

<img align="right" src="assets/brain-scan.png" width="300"/>

BrainScan is an advanced deep learning system that accurately classifies brain tumors from MRI scans using Convolutional Neural Networks (CNNs). The system can identify four categories of brain conditions with high precision:

- Glioma
- Meningioma
- No Tumor
- Pituitary

This project combines cutting-edge deep learning techniques with an intuitive web interface, making it accessible for medical professionals without requiring extensive technical knowledge.

<details>
<summary>📊 Click to view project statistics</summary>
<br>

| Metric | Value |
|--------|-------|
| Classification Accuracy | 96.8% |
| Training Time | ~45 minutes |
| Model Size | 24MB |
| Dataset Size | 3,000 images |
| Development Time | 3 months |

</details>

---

## ✨ Features

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/multi-class.png" width="100"/><br>
        <b>Multi-Class Classification</b><br>
        Identifies four different types of brain conditions
      </td>
      <td align="center">
        <img src="assets/accuracy.png" width="100"/><br>
        <b>High Accuracy</b><br>
        96.8% accuracy on test dataset
      </td>
      <td align="center">
        <img src="assets/realtime.png" width="100"/><br>
        <b>Real-time Analysis</b><br>
        Instant classification results
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="assets/interface.png" width="100"/><br>
        <b>Interactive Interface</b><br>
        User-friendly web application
      </td>
      <td align="center">
        <img src="assets/metrics.png" width="100"/><br>
        <b>Performance Metrics</b><br>
        Comprehensive model evaluation
      </td>
      <td align="center">
        <img src="assets/samples.png" width="100"/><br>
        <b>Sample Images</b><br>
        Test with pre-loaded samples
      </td>
    </tr>
  </table>
</div>

---

## 🌐 Live Demo

<div align="center">
  <a href="https://yourusername.github.io/brain-tumor-classification">
    <img src="assets/live-demo-button.png" alt="Try Live Demo" width="300"/>
  </a>
  
  <p>Scan this QR code to try the demo on your mobile device:</p>
  <img src="assets/demo-qr-code.png" alt="Demo QR Code" width="200"/>
</div>

---

## 📱 Screenshots

<div align="center">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
    <img src="assets/screenshot1.png" width="400" style="border-radius: 10px;"/>
    <img src="assets/screenshot2.png" width="400" style="border-radius: 10px;"/>
    <img src="assets/screenshot3.png" width="400" style="border-radius: 10px;"/>
    <img src="assets/screenshot4.png" width="400" style="border-radius: 10px;"/>
  </div>
  
  <p>
    <a href="assets/screenshots.md">View More Screenshots</a>
  </p>
</div>

---

## 🛠️ Technologies

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" width="100"/></td>
    </tr>
    <tr>
      <td align="center"><img src="https://img.shields.io/badge/-HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" width="100"/></td>
      <td align="center"><img src="https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" width="100"/></td>
    </tr>
  </table>
</div>

---

## 📊 Model Architecture

<div align="center">
  <img src="assets/model-architecture.png" alt="Model Architecture" width="700"/>
  
  <details>
  <summary>View Model Code</summary>
  
  ```python
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
      BatchNormalization(),
      MaxPooling2D(2, 2),
      Conv2D(64, (3, 3), activation='relu'),
      BatchNormalization(),
      MaxPooling2D(2, 2),
      Conv2D(128, (3, 3), activation='relu'),
      BatchNormalization(),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.4),
      Dense(4, activation='softmax')
  ])
