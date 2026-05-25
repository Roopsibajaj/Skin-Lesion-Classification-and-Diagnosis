# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Dummy Model Generator (For UI/System testing without full dataset training)
# ============================================================================
# This script creates a dummy trained model and generates simulated training
# history and plots. This allows testing the Flask application, Grad-CAM,
# and UI styling immediately without downloading the full 2.4 GB HAM10000 dataset.
# ============================================================================

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config

def generate_simulated_plots():
    """
    Generate simulated evaluation plots matching the figures in Chapter 6:
    - training_validation_accuracy.png (Figure 6.2)
    - training_validation_loss.png (Figure 6.3)
    - confusion_matrix.png (Figure 6.5)
    - performance_metrics.png (Figure 6.4)
    - class_distribution.png
    """
    print("[INFO] Generating simulated evaluation plots...")
    
    # Set matplotlib backend to Agg for headless environments
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src import evaluate

    # Ensure static images directory exists
    os.makedirs(config.STATIC_IMAGES_DIR, exist_ok=True)
    
    # 1. Generate Simulated History
    epochs = 20
    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    # Simulate values
    np.random.seed(42)
    for epoch in range(1, epochs + 1):
        # Phase 1: Feature Extraction (Epochs 1-5)
        if epoch <= 5:
            acc = 0.40 + 0.08 * epoch + np.random.normal(0, 0.01)
            val_acc = 0.38 + 0.07 * epoch + np.random.normal(0, 0.015)
            loss = 1.6 - 0.20 * epoch + np.random.normal(0, 0.02)
            val_loss = 1.7 - 0.18 * epoch + np.random.normal(0, 0.03)
        # Phase 2: Fine-Tuning (Epochs 6-20)
        else:
            p = (epoch - 5) / 15.0
            acc = 0.80 + 0.11 * p - 0.01 * (p**2) + np.random.normal(0, 0.005)
            val_acc = 0.73 + 0.16 * p - 0.02 * (p**2) + np.random.normal(0, 0.008)
            loss = 0.6 - 0.45 * p + np.random.normal(0, 0.01)
            val_loss = 0.8 - 0.55 * p + np.random.normal(0, 0.015)
            
        history['accuracy'].append(min(float(acc), 0.99))
        history['val_accuracy'].append(min(float(val_acc), 0.99))
        history['loss'].append(max(float(loss), 0.05))
        history['val_loss'].append(max(float(val_loss), 0.10))

    # Save training history JSON
    history_path = os.path.join(config.MODELS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  [SAVED] Simulated training history: {history_path}")

    # Generate graphs using src/evaluate.py plotting functions
    evaluate.plot_training_history(history)
    evaluate.plot_class_distribution()

    # 2. Confusion Matrix (Simulated)
    # 7 classes: akiec, bcc, bkl, df, mel, nv, vasc
    #nv has high counts, df has low counts
    class_labels = sorted(config.CLASS_TO_INDEX.keys())
    # nv class is class 5. df class is class 3.
    # Total count in test set ~ 2000 images
    cm = np.array([
        [50,  5,  8,  0,  2,  0,  0],  # akiec
        [ 4, 85,  5,  0,  6,  2,  1],  # bcc
        [ 6,  4,180,  1, 15, 12,  2],  # bkl
        [ 0,  0,  1, 20,  0,  2,  0],  # df
        [ 3,  8, 12,  0,175, 22,  0],  # mel
        [ 1,  2, 10,  1, 18,1310, 3],  # nv
        [ 0,  0,  1,  0,  0,  2, 25]   # vasc
    ])
    evaluate.plot_confusion_matrix(cm, class_labels)

    # 3. Model performance evaluation metrics
    results = {
        'accuracy': 0.892,
        'precision': 0.887,
        'recall': 0.892,
        'f1_score': 0.889
    }
    evaluate.plot_performance_metrics(results)
    
    # Save simulated evaluation results json
    results_path = os.path.join(config.STATIC_IMAGES_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'classification_report': {
                'accuracy': results['accuracy'],
                'macro avg': {'precision': 0.84, 'recall': 0.78, 'f1-score': 0.80, 'support': 2003},
                'weighted avg': {'precision': 0.89, 'recall': 0.89, 'f1-score': 0.89, 'support': 2003}
            }
        }, f, indent=2)
    print(f"  [SAVED] Simulated evaluation results: {results_path}")

def build_and_save_dummy_model():
    """
    Build a real MobileNetV2 Keras model with randomly initialized weights (untrained)
    and save it. This satisfies the model file requirement, compiles successfully,
    and supports standard inference and Grad-CAM execution.
    """
    print("[INFO] Building and saving dummy Keras model...")
    import tensorflow as tf
    from src.model import build_model, compile_model
    
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Build model using config variables, but with weights=None (to avoid download)
    print("  Initializing MobileNetV2 base (weights=None for fast creation)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    
    inputs = tf.keras.layers.Input(shape=config.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_norm')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_50')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_30')(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax', name='classification')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='SkinLesionClassifier')
    model = compile_model(model)
    
    # Save the dummy model to config path
    model.save(config.MODEL_SAVE_PATH)
    print(f"[SUCCESS] Dummy model saved to: {config.MODEL_SAVE_PATH}")
    print("  This model is fully functional for web app testing and Grad-CAM visualization!")

if __name__ == '__main__':
    print("=" * 60)
    print("  DUMMY MODEL AND ASSET GENERATION")
    print("=" * 60)
    
    # Ensure all directories exist
    config.ensure_directories()
    
    # Build Keras model
    try:
        build_and_save_dummy_model()
    except Exception as e:
        print(f"[ERROR] Failed to save dummy model: {e}")
        print("  Make sure tensorflow is installed and importable.")
        sys.exit(1)
        
    # Generate simulated plots
    try:
        generate_simulated_plots()
    except Exception as e:
        print(f"[ERROR] Failed to generate simulated plots: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("  DUMMY MODEL SETUP COMPLETED SUCCESSFULLY!")
    print("  You can now start the Flask web application:")
    print("  python app.py")
    print("=" * 60)
