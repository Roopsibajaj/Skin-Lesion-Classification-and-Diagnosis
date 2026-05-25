# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Training Pipeline Module
# ============================================================================
# This module orchestrates the complete model training process including:
#   - Phase 1: Feature extraction (frozen base, higher LR)
#   - Phase 2: Fine-tuning (unfrozen top layers, lower LR)
#   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#   - Training history saving for evaluation plots
#
# Training parameters as specified in Chapter 6.2:
#   - Optimizer: Adam
#   - Loss: Categorical Cross-Entropy
#   - Batch Size: 16
#   - Epochs: 10-20
# ============================================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from . import config
from .data_loader import prepare_dataset
from .augmentation import get_data_generators
from .model import build_model, compile_model, unfreeze_top_layers


def get_callbacks():
    """
    Create training callbacks for monitoring and optimization.
    
    Callbacks:
    1. ModelCheckpoint: Save the best model based on validation accuracy
    2. EarlyStopping: Stop training if validation loss doesn't improve
    3. ReduceLROnPlateau: Reduce learning rate when validation loss plateaus
    
    Returns:
        list: List of Keras callback instances
    """
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Stop training early if validation loss doesn't improve
        # for 5 consecutive epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        # Factor of 0.5, patience of 3 epochs
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model():
    """
    Execute the complete two-phase training pipeline.
    
    Phase 1 (Feature Extraction):
        - MobileNetV2 base layers are FROZEN
        - Only the custom classification head is trained
        - Higher learning rate (1e-3) for faster convergence
        - 5 epochs
    
    Phase 2 (Fine-Tuning):
        - Top 50 layers of MobileNetV2 are UNFROZEN
        - Entire model is trained with very low learning rate (1e-5)
        - Prevents destroying pretrained features
        - 15 epochs
    
    Returns:
        tuple: (model, history_phase1, history_phase2)
    """
    print("=" * 60)
    print("  SKIN LESION CLASSIFICATION — MODEL TRAINING")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Step 1: Prepare dataset
    # ------------------------------------------------------------------
    print("\n[STEP 1/5] Preparing dataset...")
    train_df, val_df, test_df, class_weights = prepare_dataset(oversample=True)
    
    # ------------------------------------------------------------------
    # Step 2: Create data generators with augmentation
    # ------------------------------------------------------------------
    print("\n[STEP 2/5] Creating data generators...")
    train_gen, val_gen, test_gen = get_data_generators(
        train_df, val_df, test_df
    )
    
    # ------------------------------------------------------------------
    # Step 3: Build and compile model (Phase 1 — frozen base)
    # ------------------------------------------------------------------
    print("\n[STEP 3/5] Building model (Phase 1 — Feature Extraction)...")
    model, base_model = build_model(freeze_base=True)
    model = compile_model(model, learning_rate=config.LEARNING_RATE_PHASE1)
    
    # Print full model summary
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # ------------------------------------------------------------------
    # Step 4: Phase 1 Training — Feature Extraction
    # ------------------------------------------------------------------
    print("\n[STEP 4/5] Phase 1: Training classification head...")
    print(f"  Epochs: {config.EPOCHS_PHASE1}")
    print(f"  Learning Rate: {config.LEARNING_RATE_PHASE1}")
    print(f"  Base Model: FROZEN")
    print("-" * 40)
    
    history_phase1 = model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # ------------------------------------------------------------------
    # Step 5: Phase 2 Training — Fine-Tuning
    # ------------------------------------------------------------------
    print("\n[STEP 5/5] Phase 2: Fine-tuning top layers...")
    print(f"  Epochs: {config.EPOCHS_PHASE2}")
    print(f"  Learning Rate: {config.LEARNING_RATE_PHASE2}")
    print(f"  Unfreezing top 50 layers")
    print("-" * 40)
    
    # Unfreeze top layers and recompile with lower learning rate
    model = unfreeze_top_layers(model, base_model, num_layers_to_unfreeze=50)
    
    history_phase2 = model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # ------------------------------------------------------------------
    # Save final model and training history
    # ------------------------------------------------------------------
    print("\n[INFO] Saving final model...")
    model.save(config.MODEL_SAVE_PATH)
    print(f"  Model saved to: {config.MODEL_SAVE_PATH}")
    
    # Combine training histories from both phases
    combined_history = combine_histories(history_phase1, history_phase2)
    
    # Save training history as JSON for evaluation plotting
    history_path = os.path.join(config.MODELS_DIR, 'training_history.json')
    save_training_history(combined_history, history_path)
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    
    # Print final metrics
    final_train_acc = combined_history['accuracy'][-1]
    final_val_acc = combined_history['val_accuracy'][-1]
    print(f"\n  Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    
    return model, combined_history


def combine_histories(history1, history2):
    """
    Combine training histories from Phase 1 and Phase 2 into a single
    dictionary for continuous plotting.
    
    Args:
        history1: Keras History object from Phase 1
        history2: Keras History object from Phase 2
    
    Returns:
        dict: Combined history dictionary
    """
    combined = {}
    
    for key in history1.history.keys():
        combined[key] = (
            history1.history[key] + history2.history[key]
        )
    
    return combined


def save_training_history(history, filepath):
    """
    Save training history to a JSON file for later evaluation plotting.
    
    Args:
        history (dict): Training history dictionary
        filepath (str): Path to save the JSON file
    """
    # Convert numpy values to Python floats for JSON serialization
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [float(v) for v in values]
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"  Training history saved to: {filepath}")


def load_training_history(filepath=None):
    """
    Load previously saved training history from JSON file.
    
    Args:
        filepath (str): Path to the history JSON file.
                        If None, uses default path.
    
    Returns:
        dict: Training history dictionary
    """
    if filepath is None:
        filepath = os.path.join(config.MODELS_DIR, 'training_history.json')
    
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    return history


# ============================================================================
# Entry point for standalone training
# ============================================================================
if __name__ == '__main__':
    # Ensure required directories exist
    config.ensure_directories()
    
    # Run training pipeline
    model, history = train_model()
