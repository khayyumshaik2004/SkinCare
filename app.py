import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from recommendations_data import RECOMMENDATIONS  # Your recommendation dictionary
import json
from flask import session
from foods_data import FOODS


# ------------------------
# Flask Configuration
# ------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_skin_key_123'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Load Models Once
# ------------------------
MODEL_MAIN_PATH = 'skin_classification_model.keras'
MODEL_SECONDARY_PATH = 'Skin-Type-Recognition'

# Load models safely
try:
    model_main = tf.keras.models.load_model(MODEL_MAIN_PATH)
    model_secondary = tf.keras.models.load_model(MODEL_SECONDARY_PATH)
    print("✅ Models loaded successfully.")
except Exception as e:
    print("❌ Error loading models:", e)
    model_main, model_secondary = None, None

# ------------------------
# Utility Functions
# ------------------------
def allowed_file(filename):
    """Check allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for MobileNetV2-based models."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# ------------------------
# Prediction Logic
# ------------------------
def predict_skin_type(image_path):
    """
    Predicts skin type using two models:
    1. Main model: acne/dry/oil
    2. Secondary model: differentiates dry vs oily if not acne
    """
    if model_main is None or model_secondary is None:
        return "Model not loaded", []

    # --- First Model: Acne/Dry/Oil ---
    img_array = preprocess_image(image_path)
    preds = model_main.predict(img_array)
    class_names = ['acne', 'dry', 'oil']
    predicted_class = class_names[np.argmax(preds)]

    # --- If Acne ---
    if predicted_class == 'acne':
        recs = RECOMMENDATIONS.get('acne', [])
        return "Acne", recs

    # --- Else: use secondary model (Dry vs Oily) ---
    # Load and process image for secondary model
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)

    label = model_secondary.predict(img)
    secondary_class = ['Dry', 'Oily'][int(tf.argmax(tf.squeeze(label).numpy()))]

    recs = RECOMMENDATIONS.get(secondary_class.lower(), [])
    return secondary_class, recs

# ------------------------
# Flask Routes
# ------------------------
LANGUAGES = ['en', 'te', 'hi']

def load_translation(lang):
    """Load JSON translation file"""
    path = os.path.join('translations', f'{lang}.json')
    if not os.path.exists(path):
        path = os.path.join('translations', 'en.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/')
def home():
    lang = session.get('lang', 'en')
    text = load_translation(lang)
    return render_template('index.html', text=text, lang=lang)

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in LANGUAGES:
        session['lang'] = lang
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # --- Predict Skin Type ---
    skin_type, recommendations = predict_skin_type(file_path)

    lang = session.get('lang', 'en')
    text = load_translation(lang)
    # Determine audio file path based on skin type and language
    audio_filename = f"{skin_type.lower()}.mp3"
    audio_url = url_for('static', filename=f'audios/{lang}/{audio_filename}')
    foods = FOODS.get(lang, FOODS[lang])
    
    return render_template(
        'results.html',
        skin_type=skin_type,
        image_url=url_for('static', filename=f'uploads/{filename}'),
        recommendations=recommendations,
        text=text,
        lang=lang,
        audio_url=audio_url,
        foods=foods
    )

if __name__ == '__main__':
    app.run(debug=True)