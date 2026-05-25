# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Configuration Module
# ============================================================================
# Central configuration file containing all hyperparameters, file paths,
# class mappings, and system settings used across the project.
# ============================================================================

import os

# ----------------------------------------------------------------------------
# Base Paths
# ----------------------------------------------------------------------------
# Get the project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
METADATA_FILE = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')

# Model save path
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'skin_lesion_model.h5')

# Upload and static paths
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
STATIC_IMAGES_DIR = os.path.join(STATIC_DIR, 'images')

# ----------------------------------------------------------------------------
# Image Configuration
# ----------------------------------------------------------------------------
# All images are resized to 224x224 pixels to match CNN input requirements
# as specified in Table 5.4 of the project report
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ----------------------------------------------------------------------------
# Training Hyperparameters
# ----------------------------------------------------------------------------
# Training parameters as specified in Chapter 6 (Experimental Setup)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 5          # Feature extraction phase (frozen base)
EPOCHS_PHASE2 = 15         # Fine-tuning phase (unfrozen top layers)
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2  # Total: 20 epochs

# Learning rates for two-phase training strategy
LEARNING_RATE_PHASE1 = 1e-3   # Higher LR for training classification head
LEARNING_RATE_PHASE2 = 1e-5   # Very low LR for fine-tuning pretrained layers

# Dataset split ratios
TRAIN_SPLIT = 0.80   # 80% for training (as per Chapter 6.2)
TEST_SPLIT = 0.20    # 20% for testing
VAL_SPLIT = 0.15     # 15% of training set used for validation

# ----------------------------------------------------------------------------
# Class Definitions
# ----------------------------------------------------------------------------
# HAM10000 Dataset — 7 Skin Lesion Classes
# As defined in Table 5.1 of the project report
NUM_CLASSES = 7

# Abbreviation to index mapping
CLASS_NAMES = {
    0: 'akiec',    # Actinic keratoses and intraepithelial carcinoma
    1: 'bcc',      # Basal cell carcinoma
    2: 'bkl',      # Benign keratosis-like lesions
    3: 'df',       # Dermatofibroma
    4: 'mel',      # Melanoma
    5: 'nv',       # Melanocytic nevi
    6: 'vasc'      # Vascular lesions
}

# Index to abbreviation mapping (reverse)
CLASS_TO_INDEX = {v: k for k, v in CLASS_NAMES.items()}

# Full descriptive names for display in the web application
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

# Clinical risk levels for each lesion type
# Used in the web application to display risk information
CLASS_RISK_LEVELS = {
    'akiec': 'Pre-cancerous / High Risk',
    'bcc': 'Malignant / Moderate Risk',
    'bkl': 'Benign / Low Risk',
    'df': 'Benign / Very Low Risk',
    'mel': 'Malignant / Very High Risk',
    'nv': 'Benign / Low Risk',
    'vasc': 'Benign / Very Low Risk'
}

# Risk level color coding for UI display
RISK_COLORS = {
    'Very High Risk': '#ff1744',
    'High Risk': '#ff6d00',
    'Moderate Risk': '#ffab00',
    'Low Risk': '#00c853',
    'Very Low Risk': '#00e676',
    'Pre-cancerous / High Risk': '#ff6d00',
    'Malignant / Very High Risk': '#ff1744',
    'Malignant / Moderate Risk': '#ffab00',
    'Benign / Low Risk': '#00c853',
    'Benign / Very Low Risk': '#00e676'
}

# HAM10000 class distribution (for reference and class weight computation)
# As per Table 5.1 in the report
CLASS_DISTRIBUTION = {
    'nv': 6705,
    'mel': 1113,
    'bkl': 1099,
    'bcc': 514,
    'akiec': 327,
    'vasc': 142,
    'df': 115
}

# ----------------------------------------------------------------------------
# Data Augmentation Parameters
# ----------------------------------------------------------------------------
# Augmentation parameters as defined in Table 5.3 of the project report
AUGMENTATION_PARAMS = {
    'rotation_range': 40,              # Random rotation between 0° and 40°
    'width_shift_range': 0.2,          # Horizontal shift
    'height_shift_range': 0.2,         # Vertical shift
    'horizontal_flip': True,           # Random horizontal flip
    'vertical_flip': True,             # Random vertical flip
    'zoom_range': 0.2,                 # Zoom factor 0.8–1.2x
    'brightness_range': [0.8, 1.2],    # Brightness jitter ±20%
    'fill_mode': 'nearest',            # Fill mode for new pixels
    'shear_range': 0.15                # Shear transformation
}

# ----------------------------------------------------------------------------
# Allowed File Extensions for Upload
# ----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Maximum upload file size (16 MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# ----------------------------------------------------------------------------
# Flask Configuration
# ----------------------------------------------------------------------------
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
FLASK_DEBUG = True

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

def ensure_directories():
    """
    Create all required project directories if they don't exist.
    This function is called during application startup.
    """
    directories = [
        DATASET_DIR,
        IMAGES_DIR,
        MODELS_DIR,
        UPLOADS_DIR,
        STATIC_IMAGES_DIR,
        os.path.join(BASE_DIR, 'notebooks')
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_class_name(index):
    """
    Get the full descriptive name for a class index.
    
    Args:
        index (int): Class index (0-6)
    
    Returns:
        str: Full descriptive class name
    """
    abbreviation = CLASS_NAMES.get(index, 'Unknown')
    return CLASS_FULL_NAMES.get(abbreviation, 'Unknown')


def get_risk_level(abbreviation):
    """
    Get the clinical risk level for a lesion type.
    
    Args:
        abbreviation (str): Class abbreviation (e.g., 'mel', 'nv')
    
    Returns:
        str: Risk level description
    """
    return CLASS_RISK_LEVELS.get(abbreviation, 'Unknown')
