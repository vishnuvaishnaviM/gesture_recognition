from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

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

        # Predict the hand sign
        result = predict_sign(image)

        return render_template('result.html', prediction=result)

# Run the app
#if __name__ == '__main__':
#    app.run(debug=False,host='0.0.0.0')
