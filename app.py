import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
#from tensorflow import keras
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
MODEL_PATH = "lung_cancer_model3.h5"  # Update with your actual path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names based on model
CLASS_NAMES = ['benign', 'malignant', 'normal']

# Ensure upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess and predict the class of an image
def classify_image(img_path):
    try:
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = img.resize((254, 254))  # Resize to match model input size
        img_array = np.asarray(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (H, W, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (B, H, W, C)

        # Predict the probabilities
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index])

        return predicted_class, confidence
    except Exception as e:
        return str(e), 0.0

# Route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return "Lung Cancer Detection API is running!"

# Route for image classification
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Securely save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Perform classification
    predicted_class, confidence = classify_image(file_path)

    # Remove the file after processing
    os.remove(file_path)

    return jsonify({"prediction": predicted_class, "confidence": confidence})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
