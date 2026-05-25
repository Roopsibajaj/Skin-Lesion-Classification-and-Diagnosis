# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Prediction Pipeline Module
# ============================================================================
# This module provides the single-image prediction pipeline used by the
# Flask web application. It chains preprocessing, model inference,
# probability extraction, and Grad-CAM generation into a single call.
# ============================================================================

import os
import numpy as np
import cv2
from . import config
from .preprocessing import preprocess_for_prediction
from .gradcam import generate_gradcam


def predict_image(model, image_path, generate_heatmap=True):
    """
    Run the complete prediction pipeline on a single uploaded image.
    
    Pipeline steps:
    1. Preprocess image (resize, hair removal, CLAHE, normalize)
    2. Run model inference to get class probabilities
    3. Extract predicted class and confidence score
    4. Generate Grad-CAM heatmap for visual explanation
    5. Return all results for display in the web application
    
    Args:
        model (tf.keras.Model): Trained classification model
        image_path (str): Path to the uploaded image file
        generate_heatmap (bool): Whether to generate Grad-CAM heatmap
    
    Returns:
        dict: Prediction results containing:
            - predicted_class: Class abbreviation (e.g., 'mel')
            - predicted_class_full: Full class name (e.g., 'Melanoma')
            - confidence: Confidence score (0-1) for predicted class
            - risk_level: Clinical risk level string
            - probabilities: Dict of all class probabilities
            - gradcam_path: Path to saved Grad-CAM image (or None)
            - original_path: Path to the uploaded image
        None: If image preprocessing fails
    """
    print(f"[INFO] Processing image: {image_path}")
    
    # ------------------------------------------------------------------
    # Step 1: Preprocess the image
    # ------------------------------------------------------------------
    preprocessed, original_rgb = preprocess_for_prediction(image_path)
    
    if preprocessed is None:
        print("[ERROR] Image preprocessing failed")
        return None
    
    # ------------------------------------------------------------------
    # Step 2: Run model inference
    # ------------------------------------------------------------------
    print("  Running model inference...")
    predictions = model.predict(preprocessed, verbose=0)
    
    # Get predicted class index and confidence
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    
    # Get class names
    predicted_class = config.CLASS_NAMES[predicted_index]
    predicted_class_full = config.CLASS_FULL_NAMES[predicted_class]
    
    # Get risk level
    risk_level = config.CLASS_RISK_LEVELS[predicted_class]
    
    # ------------------------------------------------------------------
    # Step 3: Extract all class probabilities
    # ------------------------------------------------------------------
    probabilities = {}
    for idx, prob in enumerate(predictions[0]):
        class_abbr = config.CLASS_NAMES[idx]
        class_full = config.CLASS_FULL_NAMES[class_abbr]
        probabilities[class_abbr] = {
            'name': class_full,
            'probability': float(prob),
            'percentage': f"{float(prob) * 100:.2f}%"
        }
    
    # Sort probabilities by value (descending)
    probabilities = dict(
        sorted(probabilities.items(),
               key=lambda x: x[1]['probability'],
               reverse=True)
    )
    
    # ------------------------------------------------------------------
    # Step 4: Generate Grad-CAM heatmap
    # ------------------------------------------------------------------
    gradcam_path = None
    
    if generate_heatmap:
        print("  Generating Grad-CAM heatmap...")
        
        # Save heatmap to uploads directory
        gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
        gradcam_path = os.path.join(config.UPLOADS_DIR, gradcam_filename)
        
        overlay, heatmap = generate_gradcam(
            model=model,
            image_preprocessed=preprocessed,
            original_image=original_rgb,
            pred_index=predicted_index,
            save_path=gradcam_path
        )
        
        if overlay is None:
            gradcam_path = None
            print("  [WARNING] Grad-CAM generation failed")
    
    # ------------------------------------------------------------------
    # Step 5: Compile results
    # ------------------------------------------------------------------
    results = {
        'predicted_class': predicted_class,
        'predicted_class_full': predicted_class_full,
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.2f}%",
        'risk_level': risk_level,
        'risk_color': config.RISK_COLORS.get(risk_level, '#ffffff'),
        'probabilities': probabilities,
        'gradcam_path': gradcam_path,
        'gradcam_filename': os.path.basename(gradcam_path) if gradcam_path else None,
        'original_path': image_path,
        'original_filename': os.path.basename(image_path)
    }
    
    print(f"  Prediction: {predicted_class_full} ({predicted_class})")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print(f"  Risk Level: {risk_level}")
    
    return results


def predict_batch(model, image_paths):
    """
    Run predictions on a batch of images.
    Useful for batch evaluation or testing.
    
    Args:
        model (tf.keras.Model): Trained classification model
        image_paths (list): List of image file paths
    
    Returns:
        list: List of prediction result dictionaries
    """
    results = []
    total = len(image_paths)
    
    for idx, path in enumerate(image_paths):
        print(f"\n[{idx+1}/{total}] Processing: {os.path.basename(path)}")
        result = predict_image(model, path, generate_heatmap=False)
        if result is not None:
            results.append(result)
    
    return results
