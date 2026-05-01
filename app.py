from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model once
model = load_model("model.h5")

def preprocess_image(file):
    # Read image from request
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Resize
    img = cv2.resize(img, (28,28))
    
    # Invert
    img = 255 - img
    
    # Normalize
    img = img / 255.0
    
    # Reshape
    img = img.reshape(1,28,28,1)
    
    return img

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    
    img = preprocess_image(file)
    
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    
    return jsonify({"prediction": digit})

if __name__ == "__main__":
    app.run(debug=True)