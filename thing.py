from flask import Flask, request, redirect, url_for
import os

app = Flask(__name__)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import regularizers
from PIL import Image
import numpy as np
import urllib.request

coder = ['coat', 'toy', 'toilet paper', 'tissue paper', 'cooker', 'utensil', 'soap', 'blankets']

model = Sequential([
    Dense(units=512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=8, activation='softmax')
])

model.build((None, 784))  # Replace 784 with the correct input shape if different
model.load_weights('model_weights.h5')

def resize_image_to_grayscale(image_path, output_size=(28, 28)):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        # Resize the image
        resized_img = img.resize(output_size, Image.ANTIALIAS)
        return resized_img

# Example usage


# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images', methods=['POST'])
def images():
    if request.method == 'POST':
        # Check if file part is present
        print(request.body)
        image_path =   # Replace with your image path
        resized_image = resize_image_to_grayscale(image_path)

        # Convert to numpy array
        resized_image_array = np.array(resized_image)

        input = resized_image_array.flatten().reshape((1, 784))

        output = model.predict(input)
        return coder[int(np.argmax(output, axis = 1))]

if __name__ == '__main__':
    app.run(port=5000)