# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Flask Web Application
# ============================================================================
# This is the main Flask web application that provides a user-friendly
# interface for skin lesion classification. Users can upload dermoscopic
# images and receive:
#   - Predicted lesion class
#   - Confidence score
#   - Probability distribution across all 7 classes
#   - Grad-CAM heatmap visualization
#
# As described in Chapter 5.1 (System Architecture):
# "The user interface is developed as a web application using the Flask
#  framework, enabling clinicians to upload dermoscopic images and receive
#  classification results through a browser-accessible interface."
# ============================================================================

import os
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from src import config
from src.predict import predict_image
from src.model import load_trained_model

# ============================================================================
# Flask Application Setup
# ============================================================================

app = Flask(__name__)
app.secret_key = 'skin-lesion-classifier-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOADS_DIR

# Ensure required directories exist
config.ensure_directories()

# ============================================================================
# Load the trained model at startup
# ============================================================================

model = None


def get_model():
    """
    Lazy-load the trained model. The model is loaded once and cached
    in the global variable for all subsequent requests.
    
    Returns:
        tf.keras.Model: Loaded model, or None if not available
    """
    global model
    if model is None:
        model = load_trained_model(config.MODEL_SAVE_PATH)
        if model is not None:
            print("[INFO] Model loaded successfully for inference")
        else:
            print("[WARNING] No trained model found. Please train the model first.")
            print(f"  Expected model path: {config.MODEL_SAVE_PATH}")
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    Allowed: .png, .jpg, .jpeg
    
    Args:
        filename (str): Name of the uploaded file
    
    Returns:
        bool: True if the file extension is allowed
    """
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS)


def generate_unique_filename(filename):
    """
    Generate a unique filename to prevent overwriting existing uploads.
    
    Args:
        filename (str): Original filename
    
    Returns:
        str: Unique filename with UUID prefix
    """
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(filename)}"
    return unique_name


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """
    Landing page with image upload form.
    
    Displays:
    - Project title and description
    - Drag-and-drop image upload area
    - Information about the 7 skin lesion types
    """
    return render_template('index.html',
                           class_info=config.CLASS_FULL_NAMES,
                           risk_info=config.CLASS_RISK_LEVELS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and run prediction.
    
    Process:
    1. Validate uploaded file
    2. Save to uploads directory
    3. Run prediction pipeline
    4. Return results page with classification and Grad-CAM
    """
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file selected. Please upload an image.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if the filename is empty (no file selected)
    if file.filename == '':
        flash('No file selected. Please choose an image to upload.', 'error')
        return redirect(url_for('index'))
    
    # Validate file extension
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.',
              'error')
        return redirect(url_for('index'))
    
    # Check if model is loaded
    loaded_model = get_model()
    if loaded_model is None:
        flash('Model not available. Please train the model first '
              '(see README.md for instructions).', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate unique filename and save the uploaded image
        filename = generate_unique_filename(file.filename)
        filepath = os.path.join(config.UPLOADS_DIR, filename)
        file.save(filepath)
        
        print(f"[INFO] Image uploaded: {filename}")
        
        # Run prediction pipeline
        results = predict_image(loaded_model, filepath, generate_heatmap=True)
        
        if results is None:
            flash('Error processing the image. Please try a different image.',
                  'error')
            return redirect(url_for('index'))
        
        # Render results page
        return render_template('result.html', results=results)
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        flash(f'An error occurred during prediction. Please try again.',
              'error')
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded images and generated Grad-CAM heatmaps.
    
    Args:
        filename (str): Name of the file to serve
    """
    return send_from_directory(config.UPLOADS_DIR, filename)


@app.route('/about')
def about():
    """
    About page with project information and methodology description.
    """
    return render_template('index.html',
                           class_info=config.CLASS_FULL_NAMES,
                           risk_info=config.CLASS_RISK_LEVELS,
                           show_about=True)


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error (>16 MB)."""
    flash('File too large. Maximum file size is 16 MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 Not Found errors."""
    return render_template('index.html',
                           class_info=config.CLASS_FULL_NAMES,
                           risk_info=config.CLASS_RISK_LEVELS), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 Internal Server errors."""
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SKIN LESION CLASSIFICATION — WEB APPLICATION")
    print("=" * 60)
    print(f"  Server:  http://localhost:{config.FLASK_PORT}")
    print(f"  Model:   {config.MODEL_SAVE_PATH}")
    print(f"  Uploads: {config.UPLOADS_DIR}")
    print("=" * 60 + "\n")
    
    # Attempt to load model at startup
    get_model()
    
    # Start Flask development server
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
