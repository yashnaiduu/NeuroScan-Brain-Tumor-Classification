from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from werkzeug.utils import secure_filename
import random
import base64
import os
import logging
from PIL import Image
import io
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'Uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['DATASET_PATH'] = os.getenv('DATASET_PATH', './Dataset')
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'mobilenet_brain_tumor_classifier.h5')

# --- Original classes from the model ---
MODEL_CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
# --- Extended classes for reporting (includes non-MRI) ---
REPORTING_CLASS_NAMES = MODEL_CLASS_NAMES + ['not_mri']

DATASET_SUBFOLDERS = ['Training', 'Testing']

# Configure Gemini API
# The API key is read from the GOOGLE_API_KEY environment variable
try:
    genai.configure(api_key='AIzaSyD9Pr7kT2QQPMb3WxfD8U46juvn3fGsg80')
    gemini_vision_model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    gemini_vision_model = None # Indicate that Gemini is not available


# Load model
try:
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    logger.info("Brain tumor classification model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load classification model: {str(e)}")
    exit(1) # Exit if the core model fails to load

# --- Gemini Validation Function ---
def check_if_mri_with_gemini(image_bytes: bytes) -> bool:
    """Uses Gemini Vision API to check if the image is a brain MRI."""
    if not gemini_vision_model:
        logger.warning("Gemini API not available. Skipping MRI validation.")
        return True # Default to True if Gemini is not configured


# Load model
try:
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    logger.info("Brain tumor classification model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load classification model: {str(e)}")
    exit(1)

# --- Gemini Validation Function ---
def check_if_mri_with_gemini(image_bytes: bytes) -> bool:
    """
    Uses Gemini Vision API to check if the image is a brain MRI.
    If Gemini API is not available or encounters an error, it defaults to True
    to allow the image to proceed to the classification model.
    """
    if not gemini_vision_model:
        logger.warning("Gemini API not available. Skipping MRI validation and proceeding to model.")
        return True # Default to True if Gemini is not configured

    try:
        # Open image using PIL from bytes
        image_pil = Image.open(io.BytesIO(image_bytes))

        # Define the prompt for Gemini
        prompt = "Analyze this image. Is it a medical image, specifically a brain MRI scan of a human? Respond ONLY with 'YES_MRI' if it is clearly a human brain MRI scan, and ONLY with 'NO_MRI' otherwise. Do not include any other text, explanations, or punctuation."

        # Generate content from the model
        response = gemini_vision_model.generate_content([prompt, image_pil])

        # Log the raw response text for debugging
        logger.info(f"Gemini raw response text: '{response.text}'")

        # Parse the response
        text_response = response.text.strip().upper()

        # Return True only if Gemini explicitly says it's an MRI
        return text_response == 'YES_MRI'

    except Exception as e:
        logger.error(f"Error calling Gemini API for validation: {str(e)}. Proceeding to model.")
        # If Gemini API fails for any reason during the call, default to True
        # to allow the image to proceed to the classification model.
        return True


def preprocess_image(image_path):
    """Loads, resizes, and normalizes the image for the classification model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure correct color channel order
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def allowed_file(filename: str) -> bool:
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def cleanup_file(filepath: str) -> None:
    """Safely removes a file."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def get_last_conv_layer(model):
    """Finds the last convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        # Check for Conv2D and ignore depthwise/separable if needed, but Conv2D is typical
        if 'conv' in layer.name.lower() and isinstance(layer, tf.keras.layers.Conv2D):
             # Exclude activation layers sometimes named like conv layers
            if not isinstance(layer, tf.keras.layers.Activation):
                 return layer.name
    # Fallback if a typical Conv2D isn't found - try finding any conv-like layer
    for layer in reversed(model.layers):
         if 'conv' in layer.name.lower():
              return layer.name
    raise ValueError("No convolutional layer found in model for Grad-CAM")

grad_model = None
def initialize_grad_model():
    """Initializes the model needed for Grad-CAM."""
    global grad_model
    if grad_model is not None:
        return
    try:
        last_conv_layer = get_last_conv_layer(model)
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.output, model.get_layer(last_conv_layer).output]
        )
        logger.info(f"Grad-CAM model initialized using layer: {last_conv_layer}")
    except ValueError as e:
        logger.error(f"Could not initialize Grad-CAM model: {str(e)}")
        grad_model = None

def generate_gradcam(model, img_array, class_index):
    """Generates a Grad-CAM heatmap for a specific class."""
    if grad_model is None:
         initialize_grad_model()
         if grad_model is None:
              raise RuntimeError("Grad-CAM model is not initialized.")

    # Ensure the class_index is valid for the model's output shape (4 classes)
    if not (0 <= class_index < len(MODEL_CLASS_NAMES)):
         raise ValueError(f"Invalid class index {class_index} for Grad-CAM. Must be between 0 and {len(MODEL_CLASS_NAMES)-1}.")

    with tf.GradientTape() as tape:
        # Ensure we are watching the input tensor
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        # Use the prediction of the target class
        loss = predictions[:, class_index]

    # Compute gradients of the loss with respect to the convolutional outputs
    grads = tape.gradient(loss, conv_outputs)

    # Mean intensity of the gradient over the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channel activation maps by the average gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap
    max_heatmap = np.max(heatmap)
    if max_heatmap == 0:
        logger.warning("Max heatmap value is 0, cannot normalize.")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    heatmap /= max_heatmap

    # Resize heatmap to match original image size (224x224)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    # Convert heatmap to a 0-255 integer scale
    heatmap = np.uint8(255 * heatmap)

    # Apply a colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def is_valid_image(filepath):
    """Checks if a file is a valid image by attempting to open it with PIL."""
    try:
        Image.open(filepath).verify()
        return True
    except Exception:
        return False

def fetch_random_image_path():
    """Fetches a random image path from the dataset (excluding 'not_mri' implicitly)."""
    dataset_path = app.config['DATASET_PATH']
    dataset_subfolders = [
        os.path.join(dataset_path, sub) for sub in DATASET_SUBFOLDERS if os.path.isdir(os.path.join(dataset_path, sub))
    ]
    if not dataset_subfolders:
        raise FileNotFoundError(f"No '{DATASET_SUBFOLDERS}' subfolders found in: {dataset_path}")

    available_classes_paths = []
    # Iterate only through the original model classes for fetching random images from the dataset
    for subfolder in dataset_subfolders:
        for class_name in MODEL_CLASS_NAMES:
            class_path = os.path.join(subfolder, class_name)
            if os.path.isdir(class_path):
                 # Check if the directory is not empty
                 if any(os.path.isfile(os.path.join(class_path, f)) for f in os.listdir(class_path)):
                    available_classes_paths.append(class_path)


    if not available_classes_paths:
        raise FileNotFoundError(f"No image directories with content found within {DATASET_SUBFOLDERS} and classes {MODEL_CLASS_NAMES} in: {dataset_path}")

    random_class_path = random.choice(available_classes_paths)
    image_files = [f for f in os.listdir(random_class_path) if os.path.isfile(os.path.join(random_class_path, f))]

    if not image_files:
         raise FileNotFoundError(f"No image files found in: {random_class_path}")


    random_image_name = random.choice(image_files)
    random_image_path = os.path.join(random_class_path, random_image_name)
    return random_image_path


def format_classification_results(predictions, class_names):
    """Formats raw prediction probabilities into a list of dictionaries."""
    # Ensure predictions match the number of class names for reporting
    if len(predictions) != len(class_names):
        logger.error(f"Prediction length ({len(predictions)}) does not match reporting class names length ({len(class_names)})")
        # Attempt to proceed, but results might be misaligned
        min_len = min(len(predictions), len(class_names))
        classes = [
            {
                'label': class_names[i].replace('_', ' ').capitalize(),
                'percent': round(float(predictions[i]) * 100, 2)
            }
            for i in range(min_len)
        ]
    else:
        classes = [
            {
                'label': class_names[i].replace('_', ' ').capitalize(),
                'percent': round(float(predictions[i]) * 100, 2)
            }
            for i in range(len(class_names))
        ]
    return sorted(classes, key=lambda x: x['percent'], reverse=True)


@app.route('/')
def home():
    return render_template('NeuroScan.html') # Assuming your HTML file is named NeuroScan.html


@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = None
    try:
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filepath}")

        # --- Step 1: Basic file validity check ---
        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400

        # --- Step 2: Attempt Gemini API check if it's an MRI ---
        is_mri_scan = True # Assume it's an MRI by default
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            # Only call Gemini if the model was initialized successfully
            if gemini_vision_model:
                 is_mri_scan = check_if_mri_with_gemini(image_bytes)
            else:
                 logger.warning("Gemini API model not initialized. Skipping MRI check.")

        except Exception as e:
             logger.error(f"Error during Gemini check preparation: {str(e)}. Assuming it's an MRI.")
             is_mri_scan = True # Assume True if reading file or other prep fails


        if not is_mri_scan:
            # --- Handle Non-MRI image (Gemini explicitly said NO_MRI) ---
            logger.info("Gemini classified image as NOT a Brain MRI.")
            # Create a prediction-like structure indicating it's not an MRI
            # Assume 'not_mri' is the last class in REPORTING_CLASS_NAMES
            not_mri_preds = np.zeros(len(REPORTING_CLASS_NAMES))
            # Find the index of 'not_mri' in REPORTING_CLASS_NAMES
            try:
                 not_mri_index = REPORTING_CLASS_NAMES.index('not_mri')
                 not_mri_preds[not_mri_index] = 1.0
            except ValueError:
                 logger.error("'not_mri' not found in REPORTING_CLASS_NAMES. Cannot report non-MRI correctly.")
                 return jsonify({'error': 'Configuration error: Could not find "not_mri" class.'}), 500


            classes = format_classification_results(not_mri_preds, REPORTING_CLASS_NAMES)

            return jsonify({
                'class': 'Likely Not a Brain MRI Scan',
                'confidence': 1.0,
                'classes': classes
            })

        # --- Step 3: If it is an MRI (either confirmed by Gemini or check skipped/failed), proceed with model prediction ---
        logger.info("Proceeding with tumor classification using the local model.")
        processed_image = preprocess_image(filepath)

        # Get predictions from the classification model
        predictions = model.predict(processed_image, verbose=0)[0]

        # Prepare predictions for reporting (add a small value for 'not_mri')
        # This ensures 'not_mri' is included in the formatted results, but with very low probability
        full_predictions_for_reporting = np.append(predictions, 1e-6)

        # Determine the predicted class based on the model's 4 outputs
        model_predicted_index = np.argmax(predictions)
        predicted_class_name = MODEL_CLASS_NAMES[model_predicted_index]
        confidence_in_model_class = float(predictions[model_predicted_index])

        # Format results using the extended list for the frontend bars
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)

        return jsonify({
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence_in_model_class,
            'classes': classes
        })

    except cv2.error:
        logger.error(f"OpenCV error processing image: {filepath}")
        return jsonify({'error': 'Image processing error (OpenCV)'}), 400
    except tf.errors.OpError:
        logger.error("TensorFlow OpError during model prediction.")
        return jsonify({'error': 'Model prediction failed'}), 500
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

    finally:
        if filepath:
            cleanup_file(filepath)


@app.route('/random', methods=['GET'])
def random_prediction():
    """Fetches a random dataset image and predicts."""
    try:
        # Fetching random image assumes it's an MRI from the dataset
        random_image_path = fetch_random_image_path()
        logger.info(f"Fetching random image: {random_image_path}")

        processed_image = preprocess_image(random_image_path)

        # Get predictions from the classification model
        predictions = model.predict(processed_image, verbose=0)[0]

        # Prepare predictions for reporting (add a small value for 'not_mri')
        full_predictions_for_reporting = np.append(predictions, 1e-6)

        # Determine the predicted class based on the model's 4 outputs
        model_predicted_index = np.argmax(predictions)
        predicted_class_name = MODEL_CLASS_NAMES[model_predicted_index]
        confidence_in_model_class = float(predictions[model_predicted_index])

        # Format results using the extended list for the frontend bars
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)

        # Encode the random image to base64 to send to the frontend
        base64_image = encode_image_to_base64(random_image_path)
        if not base64_image:
            return jsonify({'error': 'Failed to encode random image'}), 500

        return jsonify({
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence_in_model_class,
            'classes': classes,
            'image': base64_image
        })

    except FileNotFoundError as e:
        logger.error(f"Dataset or random image not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Unexpected error during random prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing random image: {str(e)}'}), 500


@app.route('/heatmap', methods=['POST'])
def get_heatmap():
    """Generates and returns a Grad-CAM heatmap."""
    if 'file' not in request.files:
        logger.warning("No file provided in request for heatmap.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = None
    try:
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file for heatmap: {filepath}")

        # --- Step 1: Basic file validity check ---
        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file for heatmap is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400

        # --- Step 2: Attempt Gemini API check if it's an MRI ---
        is_mri_scan = True # Assume it's an MRI by default for heatmap
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            # Only call Gemini if the model was initialized successfully
            if gemini_vision_model:
                 is_mri_scan = check_if_mri_with_gemini(image_bytes)
            else:
                 logger.warning("Gemini API model not initialized. Skipping MRI check for heatmap.")
        except Exception as e:
             logger.error(f"Error during Gemini check preparation for heatmap: {str(e)}. Assuming it's an MRI.")
             is_mri_scan = True # Assume True if reading file or other prep fails


        if not is_mri_scan:
            # --- Handle Non-MRI image (Gemini explicitly said NO_MRI) ---
            logger.warning("Gemini classified image as NOT a Brain MRI. Cannot generate heatmap.")
            return jsonify({'error': 'Cannot generate heatmap for non-MRI images'}), 400

        # --- Step 3: If it is an MRI (either confirmed by Gemini or check skipped/failed), proceed with heatmap generation ---
        logger.info("Proceeding with heatmap generation using the local model.")
        processed_image = preprocess_image(filepath)

        # Predict to get the class index for heatmap generation
        predictions = model.predict(processed_image, verbose=0)[0]
        class_index_for_heatmap = np.argmax(predictions)

        # Generate the heatmap
        heatmap = generate_gradcam(model, processed_image, class_index_for_heatmap)

        # Encode the heatmap image to base64
        _, buffer = cv2.imencode('.png', heatmap)
        encoded_heatmap = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'heatmap': encoded_heatmap})

    except cv2.error:
        logger.error(f"OpenCV error processing image for heatmap: {filepath}")
        return jsonify({'error': 'Image processing error (OpenCV)'}), 400
    except tf.errors.OpError as e:
        logger.error(f"TensorFlow OpError during heatmap generation prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Model prediction failed during heatmap generation'}), 500
    except RuntimeError as e:
        logger.error(f"Runtime error during heatmap generation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error during heatmap generation: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during heatmap generation'}), 500

    finally:
        if filepath:
            cleanup_file(filepath)


if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Initialize the Grad-CAM model when the app starts
    initialize_grad_model()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5050)
