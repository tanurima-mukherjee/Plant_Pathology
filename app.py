import os
import joblib
import numpy as np
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained models
models = {
    "Gradient Boosting": joblib.load("models/gradient_boosting_model.pkl"),
    "Random Forest": joblib.load("models/tuned_random_forest_model.pkl"),
    "SVM": joblib.load("models/tuned_svm_model.pkl")
}

# Load scaler and label encoder
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# Function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Feature extraction from image
def extract_features(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (64, 64))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return scaler.transform([hist])


# === ROUTES === #

@app.route('/')
def landing():  # 👈 sets landing.html as default homepage
    return render_template('landing.html')


@app.route('/predict-page')
def index():  # 👈 old index.html upload form
    return render_template('index.html', model_names=models.keys())


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        model_name = request.form['model']
        model = models[model_name]

        features = extract_features(filepath)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template('result.html',
                               prediction=predicted_label,
                               model_name=model_name,
                               image_path=f"uploads/{filename}")

    return "Invalid file format"


if __name__ == '__main__':
    app.run(debug=True)
