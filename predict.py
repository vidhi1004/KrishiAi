from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load trained models
crop_model = joblib.load("crop_recommendation.pkl")
disease_model = tf.keras.models.load_model("plant_disease_model.h5")

# Crop Recommendation Endpoint
@app.route("/recommend", methods=["POST"])
def recommend_crop():
    data = request.json
    input_features = np.array([[data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]])
    prediction = crop_model.predict(input_features)
    return jsonify({"Recommended Crop": prediction[0]})

# Plant Disease Detection Endpoint
@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    file = request.files['file']
    image = Image.open(file).resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = np.argmax(disease_model.predict(image_array), axis=1)[0]
    diseases = ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Healthy", "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy", "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy", "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy"]
    return jsonify({"Detected Disease": diseases[prediction]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
