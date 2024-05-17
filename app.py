from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import cloudinary
from cloudinary.uploader import upload
import tempfile
import os

# Initialize Flask app
app = Flask(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Load the trained model
model = load_model('Modeltrained.h5')

# Define classes
classes = ['paper', 'rock', 'scissors', 'other']

# Function to predict hand sign from image
def predict_sign(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)  # Convert to float32
    image = tf.reshape(image, (-1, 224, 224, 3))
    image = image / 255.0  # Normalize

    # Predict the hand sign
    predictions = model.predict(image)
    sign_index = np.argmax(predictions)
    return classes[sign_index]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Read the image file
        image = Image.open(file.stream)
        image = np.array(image)

        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_pil = Image.fromarray(image)
            image_pil.save(temp_file, format='JPEG')
            temp_file_path = temp_file.name

        try:
            # Upload image to Cloudinary
            upload_result = upload(temp_file_path)

            # Get the URL of the uploaded image
            image_url = upload_result.get("url")

            # Predict the hand sign
            result = predict_sign(image)

        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

        return render_template('result.html', prediction=result, image_url=image_url)

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
