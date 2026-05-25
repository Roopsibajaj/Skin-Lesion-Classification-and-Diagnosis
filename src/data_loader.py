# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Data Loading Module
# ============================================================================
# This module handles loading the HAM10000 dataset, addressing class
# imbalance through oversampling, and splitting the data into training
# and testing sets as described in Chapter 5 and Chapter 6.
# ============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from . import config


def load_metadata():
    """
    Load the HAM10000 metadata CSV file containing image IDs, diagnosis
    labels, and additional patient information.
    
    The CSV file contains columns: lesion_id, image_id, dx, dx_type,
    age, sex, localization
    
    Returns:
        pandas.DataFrame: Metadata DataFrame with added file path column
    """
    print("[INFO] Loading HAM10000 metadata...")
    
    # Read the metadata CSV file
    df = pd.read_csv(config.METADATA_FILE)
    
    # Add file path column by mapping image_id to actual file paths
    # HAM10000 images can be in .jpg format
    df['filepath'] = df['image_id'].apply(
        lambda x: find_image_path(x)
    )
    
    # Remove rows where image file was not found
    missing = df['filepath'].isna().sum()
    if missing > 0:
        print(f"  [WARNING] {missing} images not found in {config.IMAGES_DIR}")
        df = df.dropna(subset=['filepath'])
    
    # Map diagnosis labels to numeric indices
    df['label'] = df['dx'].map(config.CLASS_TO_INDEX)
    
    print(f"  Total images loaded: {len(df)}")
    print(f"  Class distribution:")
    for cls_name, count in df['dx'].value_counts().items():
        print(f"    {cls_name}: {count} images")
    
    return df


def find_image_path(image_id):
    """
    Find the full file path for a given image ID.
    Searches the images directory for jpg/png files matching the ID.
    
    Args:
        image_id (str): Image identifier from metadata
    
    Returns:
        str: Full file path if found, None otherwise
    """
    # Check common extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        filepath = os.path.join(config.IMAGES_DIR, f"{image_id}{ext}")
        if os.path.exists(filepath):
            return filepath
    return None


def oversample_minority_classes(df, target_count=None):
    """
    Address class imbalance by oversampling minority classes through
    random duplication of samples.
    
    The HAM10000 dataset is highly imbalanced (Table 5.1):
    - nv: 6705 images (majority)
    - df: 115 images (most underrepresented)
    
    This function duplicates minority class samples to reach a more
    balanced distribution, which helps prevent the model from being
    biased toward the majority class.
    
    Args:
        df (pandas.DataFrame): Original unbalanced DataFrame
        target_count (int): Target count per class. If None, uses the
                           mean of all class counts
    
    Returns:
        pandas.DataFrame: Balanced DataFrame with oversampled classes
    """
    print("[INFO] Oversampling minority classes to handle class imbalance...")
    
    # Calculate target count as the mean if not specified
    class_counts = df['dx'].value_counts()
    if target_count is None:
        target_count = int(class_counts.mean())
    
    print(f"  Target samples per class: {target_count}")
    
    balanced_dfs = []
    for class_name in config.CLASS_TO_INDEX.keys():
        class_df = df[df['dx'] == class_name]
        current_count = len(class_df)
        
        if current_count < target_count:
            # Oversample: duplicate samples to reach target count
            additional_samples = target_count - current_count
            oversampled = class_df.sample(
                n=additional_samples,
                replace=True,
                random_state=42
            )
            class_df = pd.concat([class_df, oversampled], ignore_index=True)
            print(f"    {class_name}: {current_count} -> {len(class_df)} "
                  f"(+{additional_samples} oversampled)")
        else:
            print(f"    {class_name}: {current_count} (no oversampling needed)")
        
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Balanced dataset size: {len(balanced_df)} images")
    return balanced_df


def split_dataset(df, test_size=config.TEST_SPLIT, val_size=config.VAL_SPLIT,
                  random_state=42):
    """
    Split the dataset into training, validation, and testing sets
    using stratified sampling to maintain class distribution.
    
    As per Chapter 6.2:
    - Training set: 80%
    - Testing set: 20%
    - Validation: 15% of training set
    
    Args:
        df (pandas.DataFrame): Full dataset DataFrame
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of training data for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames
    """
    print("[INFO] Splitting dataset into train/validation/test sets...")
    
    # First split: separate test set (20%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: separate validation from training (15% of remaining)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    print(f"  Training set:   {len(train_df)} images")
    print(f"  Validation set: {len(val_df)} images")
    print(f"  Testing set:    {len(test_df)} images")
    
    return train_df, val_df, test_df


def compute_class_weights(labels):
    """
    Compute class weights to handle remaining class imbalance during
    training. Higher weights are assigned to underrepresented classes.
    
    Args:
        labels (numpy.ndarray): Array of class labels
    
    Returns:
        dict: Dictionary mapping class index to weight value
    """
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    class_weights = dict(zip(unique_classes, weights))
    
    print("[INFO] Computed class weights:")
    for cls_idx, weight in class_weights.items():
        cls_name = config.CLASS_NAMES[cls_idx]
        print(f"    {cls_name} (class {cls_idx}): {weight:.4f}")
    
    return class_weights


def prepare_dataset(oversample=True):
    """
    Complete data preparation pipeline:
    1. Load metadata
    2. Optionally oversample minority classes
    3. Split into train/val/test
    4. Compute class weights
    
    Args:
        oversample (bool): Whether to oversample minority classes
    
    Returns:
        tuple: (train_df, val_df, test_df, class_weights)
    """
    # Step 1: Load metadata
    df = load_metadata()
    
    # Step 2: Handle class imbalance
    if oversample:
        df = oversample_minority_classes(df)
    
    # Step 3: Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Step 4: Compute class weights for training
    class_weights = compute_class_weights(train_df['label'].values)
    
    return train_df, val_df, test_df, class_weights
