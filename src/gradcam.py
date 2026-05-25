# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Grad-CAM (Gradient-weighted Class Activation Mapping) Module
# ============================================================================
# This module implements Grad-CAM for generating visual explanations of
# CNN predictions. Grad-CAM highlights the image regions that are most
# influential in the model's classification decision.
#
# As described in the project methodology:
# "Generates Grad-CAM heatmap overlays to highlight diagnostic regions
#  of interest" (Chapter 5.1, Classification Output Module)
# ============================================================================

import numpy as np
import cv2
import tensorflow as tf
from . import config


def get_gradcam_heatmap(model, image, pred_index=None):
    """
    Generate a Grad-CAM heatmap for the given image and model prediction.
    
    Grad-CAM computes the gradient of the predicted class score with respect
    to the feature maps of the last convolutional layer. These gradients are
    then used to weight the feature maps, producing a heatmap that highlights
    the image regions most important for the prediction.
    
    Algorithm:
    1. Extract the last convolutional layer output
    2. Compute gradients of the predicted class w.r.t. feature maps
    3. Global average pool the gradients to get importance weights
    4. Compute weighted sum of feature maps
    5. Apply ReLU to keep only positive influences
    6. Normalize to [0, 1]
    
    Args:
        model (tf.keras.Model): Trained classification model
        image (numpy.ndarray): Preprocessed image with shape (1, 224, 224, 3)
        pred_index (int): Class index to visualize. If None, uses the
                          predicted class (highest probability)
    
    Returns:
        numpy.ndarray: Grad-CAM heatmap with shape (H, W), values in [0, 1]
    """
    # Find the base model layer (which is a nested model containing other layers)
    base_model_layer = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and 'mobilenet' in layer.name:
            base_model_layer = layer
            break

    # Find the last convolutional layer in the model
    # For MobileNetV2, this is the last Conv2D layer in the base model
    last_conv_layer = find_last_conv_layer(model)
    
    if last_conv_layer is None:
        print("[ERROR] Could not find convolutional layer for Grad-CAM")
        return None
    
    # Trace gradients through nested/flat graph
    if base_model_layer is not None:
        # Create a model that outputs both the conv layer output and base model output
        base_grad_model = tf.keras.models.Model(
            inputs=base_model_layer.input,
            outputs=[last_conv_layer.output, base_model_layer.output]
        )
        
        # Compute the gradient of the predicted class with respect to the
        # feature maps of the last convolutional layer by running both parts in tape
        with tf.GradientTape() as tape:
            conv_outputs, base_features = base_grad_model(image)
            
            # Pass base features through the custom classification head
            x = base_features
            found_base = False
            for layer in model.layers:
                if layer == base_model_layer or layer.name == base_model_layer.name:
                    found_base = True
                    continue
                if found_base:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    x = layer(x)
            predictions = x
            
            # If no specific class specified, use the predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_score = predictions[:, pred_index]
        
        # Compute gradients of the class score w.r.t. conv layer outputs
        gradients = tape.gradient(class_score, conv_outputs)
    else:
        # Flat model: Create a model that outputs both the conv layer output and predictions
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute the gradient of the predicted class with respect to the
        # feature maps of the last convolutional layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            # If no specific class specified, use the predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_score = predictions[:, pred_index]
        
        # Compute gradients of the class score w.r.t. conv layer outputs
        gradients = tape.gradient(class_score, conv_outputs)
    
    # Global Average Pooling of gradients to get importance weights
    # Shape: (num_filters,)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    # Weight feature maps by their corresponding gradient importance
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    
    # Multiply each feature map by its importance weight
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU: keep only positive influences (features that increase
    # the predicted class score)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize heatmap to [0, 1] range
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    
    For MobileNetV2 wrapped in a functional model, we need to search
    through the layers of the base model.
    
    Args:
        model (tf.keras.Model): The classification model
    
    Returns:
        tf.keras.layers.Layer: Last convolutional layer, or None
    """
    # Search through all layers (including nested models)
    last_conv = None
    
    for layer in model.layers:
        # Check if this is a nested model (e.g., MobileNetV2 base)
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, (tf.keras.layers.Conv2D,
                                          tf.keras.layers.DepthwiseConv2D)):
                    last_conv = sub_layer
        elif isinstance(layer, (tf.keras.layers.Conv2D,
                                tf.keras.layers.DepthwiseConv2D)):
            last_conv = layer
    
    if last_conv is not None:
        print(f"  [Grad-CAM] Using layer: {last_conv.name}")
    
    return last_conv


def overlay_heatmap(original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the Grad-CAM heatmap on the original image.
    
    The heatmap is upscaled to match the original image dimensions,
    colorized using a colormap, and blended with the original image.
    
    Args:
        original_image (numpy.ndarray): Original RGB image (H, W, 3)
                                        with values in [0, 255] or [0, 1]
        heatmap (numpy.ndarray): Grad-CAM heatmap (h, w) in [0, 1]
        alpha (float): Blending factor (0 = only original, 1 = only heatmap)
        colormap (int): OpenCV colormap to apply
    
    Returns:
        numpy.ndarray: RGB image with heatmap overlay (H, W, 3) in [0, 255]
    """
    # Ensure original image is in uint8 format
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1],
                                            original_image.shape[0]))
    
    # Convert heatmap to uint8 for colormap application
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (JET: blue=cold/low, red=hot/high)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert from BGR (OpenCV) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend original image with colored heatmap
    overlaid = cv2.addWeighted(original_image, 1 - alpha,
                                heatmap_colored, alpha, 0)
    
    return overlaid


def generate_gradcam(model, image_preprocessed, original_image,
                     pred_index=None, save_path=None):
    """
    Complete Grad-CAM pipeline: generate heatmap and overlay on image.
    
    This is the main function called by the Flask app and prediction
    pipeline to produce visual explanations for model predictions.
    
    Args:
        model (tf.keras.Model): Trained classification model
        image_preprocessed (numpy.ndarray): Preprocessed image (1, 224, 224, 3)
        original_image (numpy.ndarray): Original RGB image for overlay
        pred_index (int): Class index to visualize (None = predicted class)
        save_path (str): Path to save the overlay image (optional)
    
    Returns:
        numpy.ndarray: Overlay image with Grad-CAM heatmap
        numpy.ndarray: Raw heatmap
    """
    # Generate Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, image_preprocessed, pred_index)
    
    if heatmap is None:
        print("[WARNING] Grad-CAM heatmap generation failed")
        return None, None
    
    # Overlay heatmap on original image
    overlay = overlay_heatmap(original_image, heatmap, alpha=0.4)
    
    # Save overlay image if path is provided
    if save_path is not None:
        # Convert RGB to BGR for OpenCV saving
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"  [SAVED] Grad-CAM overlay: {save_path}")
    
    return overlay, heatmap


def generate_gradcam_for_all_classes(model, image_preprocessed,
                                      original_image, save_dir=None):
    """
    Generate Grad-CAM heatmaps for all 7 classes.
    Useful for visualization in the notebook.
    
    Args:
        model (tf.keras.Model): Trained model
        image_preprocessed (numpy.ndarray): Preprocessed image
        original_image (numpy.ndarray): Original image for overlay
        save_dir (str): Directory to save all overlays
    
    Returns:
        dict: Dictionary mapping class name to overlay image
    """
    overlays = {}
    
    for idx, class_abbr in config.CLASS_NAMES.items():
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'gradcam_{class_abbr}.png')
        
        overlay, heatmap = generate_gradcam(
            model, image_preprocessed, original_image,
            pred_index=idx, save_path=save_path
        )
        
        if overlay is not None:
            overlays[class_abbr] = overlay
    
    return overlays
