# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Model Architecture Module
# ============================================================================
# This module implements the CNN model architecture using transfer learning
# with MobileNetV2 pretrained on ImageNet, as described in Chapter 5
# (Table 5.4) and the proposed methodology.
#
# Architecture:
#   Input (224x224x3) -> MobileNetV2 (frozen base) -> GlobalAveragePooling2D
#   -> Dense(256, ReLU) -> BatchNorm -> Dropout(0.5)
#   -> Dense(128, ReLU) -> Dropout(0.3)
#   -> Dense(7, Softmax)
# ============================================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from . import config


def build_model(input_shape=config.INPUT_SHAPE,
                num_classes=config.NUM_CLASSES,
                freeze_base=True):
    """
    Build the skin lesion classification model using MobileNetV2
    transfer learning.
    
    The architecture follows the CNN structure described in Table 5.4:
    - Base: MobileNetV2 pretrained on ImageNet (feature extraction)
    - Head: GlobalAveragePooling2D -> Dense layers -> Softmax classification
    
    Transfer learning approach (as described in Chapter 5):
    - Phase 1: Freeze base layers, train only classification head
    - Phase 2: Unfreeze top layers for fine-tuning
    
    Args:
        input_shape (tuple): Input image dimensions (224, 224, 3)
        num_classes (int): Number of output classes (7)
        freeze_base (bool): Whether to freeze the MobileNetV2 base layers
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    print("[INFO] Building MobileNetV2 transfer learning model...")
    
    # -----------------------------------------------------------------------
    # Step 1: Load MobileNetV2 base model with ImageNet weights
    # -----------------------------------------------------------------------
    # include_top=False removes the original ImageNet classification head
    # This allows us to add our own classification layers for 7 classes
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # Remove ImageNet classification layers
        weights='imagenet'          # Load pretrained ImageNet weights
    )
    
    # Freeze/unfreeze base model layers
    base_model.trainable = not freeze_base
    
    if freeze_base:
        print(f"  Base model layers: {len(base_model.layers)} (FROZEN)")
    else:
        print(f"  Base model layers: {len(base_model.layers)} (TRAINABLE)")
    
    # -----------------------------------------------------------------------
    # Step 2: Build custom classification head
    # -----------------------------------------------------------------------
    # As described in Table 5.4:
    # GlobalAveragePooling2D -> Dense(256) -> Dropout -> Dense(7, Softmax)
    
    inputs = Input(shape=input_shape)
    
    # Pass through MobileNetV2 base
    x = base_model(inputs, training=False)
    
    # Global Average Pooling: reduces spatial dimensions to feature vector
    # Output shape: (batch_size, 1280) — MobileNetV2's final feature dim
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    
    # Dense layer 1: 256 units with ReLU activation
    # Learns task-specific feature combinations
    x = Dense(256, activation='relu', name='dense_256')(x)
    
    # Batch Normalization: stabilizes and accelerates training
    x = BatchNormalization(name='batch_norm')(x)
    
    # Dropout 1: 50% dropout to prevent overfitting
    x = Dropout(0.5, name='dropout_50')(x)
    
    # Dense layer 2: 128 units with ReLU activation
    x = Dense(128, activation='relu', name='dense_128')(x)
    
    # Dropout 2: 30% dropout
    x = Dropout(0.3, name='dropout_30')(x)
    
    # Output layer: 7 units with Softmax activation
    # Produces probability distribution over the 7 skin lesion classes
    outputs = Dense(num_classes, activation='softmax', name='classification')(x)
    
    # -----------------------------------------------------------------------
    # Step 3: Create the final model
    # -----------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs,
                  name='SkinLesionClassifier')
    
    # Print model summary
    print("\n  Model Architecture Summary:")
    print(f"  {'='*50}")
    print(f"  Input Shape:        {input_shape}")
    print(f"  Base Model:         MobileNetV2 (ImageNet)")
    print(f"  Pooling:            GlobalAveragePooling2D")
    print(f"  Dense Layers:       256 (ReLU) -> 128 (ReLU)")
    print(f"  Dropout:            0.5, 0.3")
    print(f"  Output:             {num_classes} classes (Softmax)")
    print(f"  {'='*50}")
    
    # Count trainable and non-trainable parameters
    trainable = sum([tf.keras.backend.count_params(w)
                     for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w)
                         for w in model.non_trainable_weights])
    print(f"  Trainable parameters:     {trainable:,}")
    print(f"  Non-trainable parameters: {non_trainable:,}")
    print(f"  Total parameters:         {trainable + non_trainable:,}")
    
    return model, base_model


def compile_model(model, learning_rate=config.LEARNING_RATE_PHASE1):
    """
    Compile the model with optimizer, loss function, and metrics.
    
    Training configuration as specified in Chapter 6.2:
    - Optimizer: Adam
    - Loss Function: Categorical Cross-Entropy
    - Metrics: Accuracy
    
    Args:
        model (tf.keras.Model): The model to compile
        learning_rate (float): Learning rate for Adam optimizer
    
    Returns:
        tf.keras.Model: Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Model compiled:")
    print(f"  Optimizer:      Adam (lr={learning_rate})")
    print(f"  Loss Function:  Categorical Cross-Entropy")
    print(f"  Metrics:        Accuracy")
    
    return model


def unfreeze_top_layers(model, base_model, num_layers_to_unfreeze=50):
    """
    Unfreeze the top layers of the base model for fine-tuning (Phase 2).
    
    Fine-tuning with a very low learning rate allows the pretrained
    features to be adapted to medical imaging patterns without
    destroying the learned ImageNet representations.
    
    Args:
        model (tf.keras.Model): The full model
        base_model (tf.keras.Model): The MobileNetV2 base model
        num_layers_to_unfreeze (int): Number of top layers to unfreeze
    
    Returns:
        tf.keras.Model: Model with unfrozen top layers
    """
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze all layers except the top N layers
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers_to_unfreeze
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    frozen_count = sum(1 for layer in base_model.layers if not layer.trainable)
    
    print(f"[INFO] Fine-tuning configuration:")
    print(f"  Total base layers:    {total_layers}")
    print(f"  Frozen layers:        {frozen_count}")
    print(f"  Trainable layers:     {trainable_count}")
    
    # Recompile with lower learning rate for fine-tuning
    model = compile_model(model, learning_rate=config.LEARNING_RATE_PHASE2)
    
    return model


def load_trained_model(model_path=config.MODEL_SAVE_PATH):
    """
    Load a previously trained model from disk.
    
    Args:
        model_path (str): Path to the saved .h5 model file
    
    Returns:
        tf.keras.Model: Loaded model ready for inference
        None: If model file does not exist
    """
    import os
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    
    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"  Model loaded successfully")
    
    return model
