# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Image Preprocessing Module
# ============================================================================
# This module implements image preprocessing techniques as described in
# Table 5.2 of the project report:
#   1. Image Resizing (224x224)
#   2. Hair Removal (DullRazor morphological operations)
#   3. Contrast Enhancement (CLAHE)
#   4. Image Normalization ([0,1] range)
# ============================================================================

import cv2
import numpy as np
from . import config


def resize_image(image, target_size=config.IMG_SIZE):
    """
    Resize image to the target dimensions (224x224) to match CNN input
    requirements as specified in Table 5.2.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        target_size (tuple): Target dimensions (height, width)
    
    Returns:
        numpy.ndarray: Resized image
    """
    resized = cv2.resize(image, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_AREA)
    return resized


def remove_hair(image):
    """
    Remove hair artifacts from dermoscopic images using DullRazor algorithm.
    
    This technique uses morphological blackhat filtering to detect dark hair
    strands against the skin background, then inpaints the detected regions
    to produce a clean image.
    
    Steps:
        1. Convert image to grayscale
        2. Apply blackhat morphological operation to detect hair
        3. Threshold the result to create a binary mask
        4. Inpaint the masked regions using the surrounding pixels
    
    As described in Table 5.2: "Morphological operations used to detect
    and inpaint hair artefacts"
    
    Args:
        image (numpy.ndarray): Input image in BGR format
    
    Returns:
        numpy.ndarray: Image with hair artifacts removed
    """
    # Convert to grayscale for morphological operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a kernel for the blackhat operation
    # The kernel size is chosen to match typical hair strand width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Apply blackhat morphological operation
    # Blackhat = closing(image) - image
    # This highlights dark thin structures (hair) on lighter background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Apply Gaussian blur to reduce noise in the blackhat result
    blackhat_blurred = cv2.GaussianBlur(blackhat, (3, 3), 0)
    
    # Threshold to create a binary mask of hair regions
    # Pixels above the threshold are considered hair
    _, hair_mask = cv2.threshold(blackhat_blurred, 10, 255,
                                 cv2.THRESH_BINARY)
    
    # Inpaint the hair regions using the surrounding pixel information
    # cv2.INPAINT_TELEA uses the Fast Marching Method for inpainting
    clean_image = cv2.inpaint(image, hair_mask, inpaintRadius=6,
                              flags=cv2.INPAINT_TELEA)
    
    return clean_image


def enhance_contrast(image):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization).
    
    CLAHE operates on the luminance channel (L) of the LAB color space,
    improving local contrast while preventing over-amplification of noise.
    
    As described in Table 5.2: "Adaptive histogram equalization improves
    lesion visibility"
    
    Args:
        image (numpy.ndarray): Input image in BGR format
    
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    # Convert BGR to LAB color space
    # LAB separates luminance (L) from color (A, B)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the luminance channel
    # clipLimit controls contrast amplification
    # tileGridSize defines the region size for adaptive equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge enhanced L channel with original A and B channels
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def normalize_image(image):
    """
    Normalize pixel values to [0, 1] range for stable gradient computation
    during model training.
    
    As described in Table 5.2: "Pixel values scaled to [0,1] range for
    stable gradient computation"
    
    Args:
        image (numpy.ndarray): Input image with pixel values in [0, 255]
    
    Returns:
        numpy.ndarray: Normalized image with pixel values in [0.0, 1.0]
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def preprocess_single_image(image_path, apply_hair_removal=True):
    """
    Apply the complete preprocessing pipeline to a single image.
    
    This function chains all preprocessing steps in order:
        1. Read image from file
        2. Resize to 224x224
        3. Remove hair artifacts (optional)
        4. Enhance contrast with CLAHE
        5. Normalize to [0, 1]
    
    Args:
        image_path (str): Path to the input image file
        apply_hair_removal (bool): Whether to apply hair removal step
    
    Returns:
        numpy.ndarray: Fully preprocessed image ready for model input
        None: If the image file cannot be read
    """
    # Read image from file
    image = cv2.imread(image_path)
    
    # Validate that the image was loaded successfully
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    
    # Step 1: Resize to target dimensions (224x224)
    image = resize_image(image)
    
    # Step 2: Remove hair artifacts using DullRazor algorithm
    if apply_hair_removal:
        image = remove_hair(image)
    
    # Step 3: Enhance contrast using CLAHE
    image = enhance_contrast(image)
    
    # Step 4: Convert BGR to RGB (OpenCV loads as BGR, models expect RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 5: Normalize pixel values to [0, 1] range
    image = normalize_image(image)
    
    return image


def preprocess_batch(image_paths, apply_hair_removal=True, verbose=True):
    """
    Apply the complete preprocessing pipeline to a batch of images.
    
    Args:
        image_paths (list): List of paths to input image files
        apply_hair_removal (bool): Whether to apply hair removal step
        verbose (bool): Whether to print progress updates
    
    Returns:
        numpy.ndarray: Array of preprocessed images with shape
                       (N, 224, 224, 3)
    """
    preprocessed_images = []
    total = len(image_paths)
    
    for idx, path in enumerate(image_paths):
        # Print progress every 500 images
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Preprocessing: {idx + 1}/{total} images completed")
        
        # Apply full preprocessing pipeline
        processed = preprocess_single_image(path, apply_hair_removal)
        
        if processed is not None:
            preprocessed_images.append(processed)
        else:
            print(f"  [WARNING] Skipping unreadable image: {path}")
    
    if verbose:
        print(f"  Preprocessing complete: {len(preprocessed_images)}/{total} "
              f"images processed successfully")
    
    return np.array(preprocessed_images)


def preprocess_for_prediction(image_path):
    """
    Preprocess a single image for model prediction (web app upload).
    Returns the image as a batch of 1 with shape (1, 224, 224, 3).
    
    Args:
        image_path (str): Path to the uploaded image
    
    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 224, 224, 3)
        numpy.ndarray: Original resized image (for Grad-CAM overlay)
        None: If the image cannot be read
    """
    # Read and validate
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    # Resize for display (original for Grad-CAM overlay)
    original_resized = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
    original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
    
    # Full preprocessing pipeline
    processed = preprocess_single_image(image_path, apply_hair_removal=True)
    
    if processed is None:
        return None, None
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    processed_batch = np.expand_dims(processed, axis=0)
    
    return processed_batch, original_rgb
