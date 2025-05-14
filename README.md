# NeuroScan - Empowering Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-%E2%98%AB%EF%B8%8F-brightgreen.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellowgreen.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-NeuroScan-brightgreen)](https://neuroscan.mystichqra.me/)
![Project Demo](link_to_your_demo_gif_or_screenshot_here)

## Overview

NeuroScan is a sophisticated web application engineered to assist in the detection and classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. Built with the robust and flexible Flask microframework in Python, NeuroScan leverages the power of deep learning, utilizing a TensorFlow/Keras-trained model to provide accurate and insightful predictions. The application offers users a seamless experience in uploading MRI images and receiving comprehensive diagnostic support, including tumor classification (glioma, meningioma, pituitary, or no tumor), prediction confidence levels, and a detailed breakdown of the model's assessment across all tumor types.

Furthermore, NeuroScan incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) technology, generating intuitive heatmaps that visually highlight the specific regions within the MRI scan that significantly influenced the model's classification decision. This interpretability feature enhances trust and understanding in the AI-driven diagnostic process. The application also includes a functionality to explore random images from a curated dataset, facilitating continuous evaluation and demonstration of the model's capabilities.

**Access the live application:** [NeuroScan](https://neuroscan.mystichqra.me/)

## Key Capabilities

* **Intelligent Image Analysis:** Enables users to upload MRI scans in various standard image formats (PNG, JPG, JPEG, BMP) for automated analysis.
* **High-Accuracy Prediction:** Delivers precise classifications of brain tumor types with associated confidence scores, aiding in preliminary diagnosis.
* **Granular Diagnostic Insights:** Presents a detailed analysis of the model's probability assessment for each potential tumor class, offering a comprehensive view.
* **Visualized Decision Support:** Generates Grad-CAM heatmaps, providing a visual explanation of the model's focus areas within the MRI image.
* **Dataset Exploration:** Features an option to analyze random MRI images from an integrated dataset, showcasing the model's generalization ability.
* **Intuitive User Interface:** Designed with a clean and user-friendly web interface to ensure ease of interaction for medical professionals and researchers.

## Core Technologies

* **Python:** The foundational programming language, ensuring scalability and maintainability.
* **Flask:** A lightweight and adaptable web framework, providing a robust backend for the application.
* **TensorFlow/Keras:** A leading deep learning library, empowering the development and deployment of the sophisticated classification model.
* **OpenCV (cv2):** Essential for efficient image loading, preprocessing, and manipulation.
* **NumPy:** The cornerstone for numerical computations and efficient array handling.
* **Werkzeug:** Provides critical utilities for secure file uploads and HTTP request/response handling.
* **Pillow (PIL):** Used for robust image format validation and handling.
* **Base64:** Facilitates the seamless embedding of images within the web interface for visualization.

## Getting Started

Follow these steps to set up and run NeuroScan locally:

1.  **Clone the Repository:**
    ```bash
    git clone <your_github_repository_link>
    cd NeuroScan
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` file accurately lists all project dependencies.)*

3.  **Obtain the Trained Model:**
    * Place your trained brain tumor classification model file (e.g., `brain_tumor_classifier.h5`) in the root directory of the project. Alternatively, update the `MODEL_PATH` configuration within the `app.py` file.

4.  **Organize the Dataset (for random image functionality):**
    * Create a directory named `Dataset` at the project's root.
    * Inside `Dataset`, create two primary subdirectories: `Training` and `Testing`.
    * Within both `Training` and `Testing`, establish four class-specific subdirectories: `glioma`, `meningioma`, `notumor`, and `pituitary`.
    * Populate these class directories with the corresponding MRI scan images.

5.  **Configure Environment Variables (Optional):**
    * For enhanced flexibility, you can set environment variables such as `UPLOAD_FOLDER`, `DATASET_PATH`, and `MODEL_PATH` instead of hardcoding them in `app.py`.

6.  **Launch the Application:**
    ```bash
    python app.py
    ```
    * The NeuroScan web application will typically be accessible at `http://127.0.0.1:5050/`.

## Application Structure
