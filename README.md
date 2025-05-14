# NeuroScan - Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-%E2%98%AB%EF%B8%8F-brightgreen.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellowgreen.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Project Demo](link_to_your_demo_gif_or_screenshot_here)

## Overview

NeuroScan is a web application built with Flask that utilizes a deep learning model (trained with TensorFlow/Keras) to classify brain tumors from MRI scans. It allows users to upload an image, receive a prediction of the tumor type (glioma, meningioma, pituitary, or no tumor), along with a confidence score and a detailed analysis of the probabilities for each class. Additionally, it features the generation of Grad-CAM heatmaps to visualize the regions of the input image that were most important for the model's prediction.

## Key Features

* **Image Upload and Prediction:** Users can upload MRI scans in common image formats (PNG, JPG, JPEG, BMP) for classification.
* **Real-time Results:** The application provides immediate predictions with confidence scores.
* **Detailed Analysis:** Displays the probability for each of the four brain tumor classes.
* **Grad-CAM Heatmap Visualization:** Generates and displays heatmaps highlighting relevant regions in the MRI scan.
* **Random Image Exploration:** Allows users to fetch and predict on random images from a predefined dataset.
* **Clear and User-Friendly Interface:** A simple web interface built with HTML and potentially CSS.

## Technologies Used

* **Python:** The primary programming language.
* **Flask:** A micro web framework for building the web application.
* **TensorFlow/Keras:** A powerful library for building and training the deep learning model.
* **OpenCV (cv2):** Used for image processing tasks (resizing, loading).
* **NumPy:** For numerical operations and array manipulation.
* **Werkzeug:** Utilities for handling file uploads securely.
* **Pillow (PIL):** For image format validation.
* **Base64:** For encoding images for display in the web interface.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your_github_repository_link>
    cd NeuroScan
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you create a `requirements.txt` file - see section below)*

3.  **Download the trained model:**
    * Place your trained brain tumor classification model file (e.g., `brain_tumor_classifier.h5`) in the project's root directory, or update the `MODEL_PATH` in the Flask application (`app.config['MODEL_PATH']`).

4.  **Organize the dataset (if you want to use the random image feature):**
    * Create a directory named `Dataset` in the project's root.
    * Inside `Dataset`, create two subdirectories: `Training` and `Testing`.
    * Within `Training` and `Testing`, create four subdirectories corresponding to the class names: `glioma`, `meningioma`, `notumor`, and `pituitary`.
    * Place your MRI scan images into the respective class directories.

5.  **Set environment variables (optional):**
    * You can optionally set environment variables for `UPLOAD_FOLDER`, `DATASET_PATH`, and `MODEL_PATH` if you prefer not to hardcode them in the Flask app.

6.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    * The application should now be accessible at `http://127.0.0.1:5050/` (or the port you configured).

## Usage

1.  Open your web browser and navigate to the application's URL.
2.  **Upload Image:** Click on the "Choose File" button and select an MRI scan image.
3.  **Predict:** Click the "Predict" button to get the classification results, including the predicted tumor type, confidence score, and detailed analysis.
4.  **Heatmap:** If the prediction is successful, a Grad-CAM heatmap will be displayed below the results, highlighting the important regions.
5.  **Random Image:** Click the "Fetch Random Image" button to load and predict on a random image from the `Dataset`.

## Project Structure

NeuroScan/
├── app.py             # The main Flask application file
├── templates/
│   └── NeuroScan.html # The HTML template for the web interface
├── static/
│   └── style.css      # (Optional) CSS file for styling
├── uploads/           # Directory to store uploaded images temporarily
├── Dataset/           # (Optional) Directory containing the image dataset
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── brain_tumor_classifier.h5 # The trained deep learning model
├── requirements.txt   # List of Python dependencies
├── README.md          # This file
└── LICENSE            

