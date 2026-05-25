# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Data Augmentation Module
# ============================================================================
# This module implements data augmentation techniques to artificially
# increase the dataset size and improve model generalization.
# Augmentation parameters are based on Table 5.3 of the project report.
# ============================================================================

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config


def create_train_generator():
    """
    Create an ImageDataGenerator for training data with augmentation.
    
    Implements augmentation techniques from Table 5.3:
    - Random Rotation: 0° to 40°
    - Horizontal & Vertical Flip: Random 50% probability
    - Random Zoom: Factor 0.8–1.2x
    - Brightness Jitter: ±20%
    - Width/Height Shift: 20%
    - Shear: 15%
    
    The generator also rescales pixel values to [0, 1] range.
    
    Returns:
        ImageDataGenerator: Configured training data generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,                                      # Normalize to [0,1]
        rotation_range=config.AUGMENTATION_PARAMS['rotation_range'],
        width_shift_range=config.AUGMENTATION_PARAMS['width_shift_range'],
        height_shift_range=config.AUGMENTATION_PARAMS['height_shift_range'],
        horizontal_flip=config.AUGMENTATION_PARAMS['horizontal_flip'],
        vertical_flip=config.AUGMENTATION_PARAMS['vertical_flip'],
        zoom_range=config.AUGMENTATION_PARAMS['zoom_range'],
        brightness_range=config.AUGMENTATION_PARAMS['brightness_range'],
        fill_mode=config.AUGMENTATION_PARAMS['fill_mode'],
        shear_range=config.AUGMENTATION_PARAMS['shear_range']
    )
    
    return train_datagen


def create_val_test_generator():
    """
    Create an ImageDataGenerator for validation and testing data.
    
    Only rescaling is applied — no augmentation — to ensure that
    evaluation metrics reflect actual model performance on unmodified
    images.
    
    Returns:
        ImageDataGenerator: Configured validation/test data generator
    """
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0  # Only normalize, no augmentation
    )
    
    return val_test_datagen


def create_flow_from_dataframe(datagen, dataframe, x_col='filepath',
                                y_col='dx', target_size=config.IMG_SIZE,
                                batch_size=config.BATCH_SIZE, shuffle=True):
    """
    Create a data flow from a DataFrame using the given ImageDataGenerator.
    
    This function connects the augmentation pipeline to the actual image
    files referenced in the DataFrame.
    
    Args:
        datagen (ImageDataGenerator): The data generator to use
        dataframe (pandas.DataFrame): DataFrame with file paths and labels
        x_col (str): Column name containing image file paths
        y_col (str): Column name containing class labels
        target_size (tuple): Target image dimensions (height, width)
        batch_size (int): Number of images per batch
        shuffle (bool): Whether to shuffle data each epoch
    
    Returns:
        DirectoryIterator: Configured data flow iterator
    """
    # Get sorted class list for consistent label encoding
    classes = sorted(config.CLASS_TO_INDEX.keys())
    
    flow = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',    # One-hot encoded labels for softmax
        classes=classes,
        shuffle=shuffle,
        seed=42
    )
    
    return flow


def get_data_generators(train_df, val_df, test_df):
    """
    Create complete data generation pipelines for training, validation,
    and testing.
    
    Training data uses augmentation to increase effective dataset size.
    Validation and test data use only normalization.
    
    Args:
        train_df (pandas.DataFrame): Training data DataFrame
        val_df (pandas.DataFrame): Validation data DataFrame
        test_df (pandas.DataFrame): Testing data DataFrame
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    print("[INFO] Creating data generators...")
    
    # Create augmented generator for training
    train_datagen = create_train_generator()
    
    # Create non-augmented generator for validation and testing
    val_test_datagen = create_val_test_generator()
    
    # Create data flows from DataFrames
    train_generator = create_flow_from_dataframe(
        train_datagen, train_df, shuffle=True
    )
    
    val_generator = create_flow_from_dataframe(
        val_test_datagen, val_df, shuffle=False
    )
    
    test_generator = create_flow_from_dataframe(
        val_test_datagen, test_df, shuffle=False,
        batch_size=1  # Process one image at a time for evaluation
    )
    
    print(f"  Training generator: {train_generator.samples} images, "
          f"{len(train_generator)} batches/epoch")
    print(f"  Validation generator: {val_generator.samples} images, "
          f"{len(val_generator)} batches/epoch")
    print(f"  Test generator: {test_generator.samples} images")
    
    return train_generator, val_generator, test_generator


def augment_single_image(image, num_augmented=5):
    """
    Generate multiple augmented versions of a single image.
    Useful for visualizing augmentation effects.
    
    Args:
        image (numpy.ndarray): Input image with shape (H, W, 3)
        num_augmented (int): Number of augmented images to generate
    
    Returns:
        list: List of augmented images
    """
    datagen = ImageDataGenerator(
        rotation_range=config.AUGMENTATION_PARAMS['rotation_range'],
        width_shift_range=config.AUGMENTATION_PARAMS['width_shift_range'],
        height_shift_range=config.AUGMENTATION_PARAMS['height_shift_range'],
        horizontal_flip=config.AUGMENTATION_PARAMS['horizontal_flip'],
        vertical_flip=config.AUGMENTATION_PARAMS['vertical_flip'],
        zoom_range=config.AUGMENTATION_PARAMS['zoom_range'],
        brightness_range=config.AUGMENTATION_PARAMS['brightness_range'],
        fill_mode=config.AUGMENTATION_PARAMS['fill_mode'],
        shear_range=config.AUGMENTATION_PARAMS['shear_range']
    )
    
    # Reshape for the generator: (H, W, 3) -> (1, H, W, 3)
    image_batch = np.expand_dims(image, axis=0)
    
    augmented_images = []
    augment_iter = datagen.flow(image_batch, batch_size=1)
    
    for _ in range(num_augmented):
        augmented = next(augment_iter)[0]
        # Clip values to valid range
        augmented = np.clip(augmented, 0, 1)
        augmented_images.append(augmented)
    
    return augmented_images
