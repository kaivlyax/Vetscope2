from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///database/vetscope.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the model if available
MODEL_PATH = os.getenv('MODEL_PATH', '/opt/render/project/src/models/dog_disease_model_96.h5')
model = None

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def load_model_safely():
    global model
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Convert relative path to absolute path
            abs_model_path = os.path.abspath(MODEL_PATH)
            print(f"Attempt {attempt + 1}/{max_retries} to load model")
            print(f"Environment MODEL_PATH: {MODEL_PATH}")
            print(f"Absolute model path: {abs_model_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Check if directory exists
            model_dir = os.path.dirname(abs_model_path)
            if not os.path.exists(model_dir):
                print(f"Creating directory: {model_dir}")
                os.makedirs(model_dir, exist_ok=True)
            
            print(f"Directory contents: {os.listdir(model_dir)}")
            
            if os.path.exists(abs_model_path):
                file_size = os.path.getsize(abs_model_path)
                print(f"Model file exists at {abs_model_path}")
                print(f"File size: {file_size} bytes")
                
                # Basic integrity check
                if file_size < 1000:  # Assuming model should be at least 1KB
                    print(f"Warning: Model file size ({file_size} bytes) seems too small")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return False
                
                model = tf.keras.models.load_model(abs_model_path)
                print(f"Model loaded successfully from {abs_model_path}")
                return True
            else:
                print(f"Model file not found at {abs_model_path}")
                print(f"Directory contents: {os.listdir(model_dir)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return False
        except Exception as e:
            print(f"Error loading model (attempt {attempt + 1}): {str(e)}")
            print(f"Model path: {MODEL_PATH}")
            print(f"Absolute path: {abs_model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir(os.path.dirname(abs_model_path))}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False
    
    return False

# Try to load the model
model_loaded = load_model_safely()

# Define disease information
DISEASE_INFO = {
    'healthy': {
        'symptoms': 'No visible symptoms of illness',
        'tips': ['Regular vet check-ups', 'Maintain balanced diet', 'Regular exercise']
    },
    'skin_infection': {
        'symptoms': 'Redness, itching, hair loss, scabs',
        'tips': ['Keep affected area clean', 'Use prescribed medications', 'Prevent scratching']
    },
    # Add more diseases as per your model's classes
}

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(120))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
            
        user = User(email=email, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', model_available=model is not None)

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((300, 300))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not available. Please contact support.',
            'status': 'maintenance'
        }), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        predicted_class = tf.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease = model.class_names[predicted_class]
        disease_info = DISEASE_INFO.get(disease, {
            'symptoms': 'Information not available',
            'tips': ['Consult a veterinarian']
        })
        
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'symptoms': disease_info['symptoms'],
            'tips': disease_info['tips']
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 