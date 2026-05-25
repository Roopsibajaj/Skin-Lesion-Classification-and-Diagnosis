# ============================================================================
# Deep Learning Based Skin Lesion Classification and Diagnosis
# Dataset Download Helper Script
# ============================================================================
# This script downloads the HAM10000 dataset from Kaggle and organizes
# it into the expected folder structure.
#
# Prerequisites:
#   1. Install the Kaggle API: pip install kaggle
#   2. Place your Kaggle API key at: ~/.kaggle/kaggle.json
#      (Download from https://www.kaggle.com/settings → API → Create New Token)
#
# Alternatively, manually download the dataset from:
#   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# ============================================================================

import os
import sys
import shutil
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config


def download_from_kaggle():
    """
    Download the HAM10000 dataset using the Kaggle API.
    
    Requires:
    - kaggle package installed (pip install kaggle)
    - Kaggle API key at ~/.kaggle/kaggle.json
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] Kaggle API not installed.")
        print("  Install it with: pip install kaggle")
        print("  Then place your API key at: ~/.kaggle/kaggle.json")
        return False
    
    print("[INFO] Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        print("  Make sure your API key is at: ~/.kaggle/kaggle.json")
        print("  Download it from: https://www.kaggle.com/settings")
        return False
    
    print("[INFO] Downloading HAM10000 dataset from Kaggle...")
    print("  This may take several minutes (~2.4 GB)...")
    
    # Ensure dataset directory exists
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    
    try:
        api.dataset_download_files(
            'kmader/skin-cancer-mnist-ham10000',
            path=config.DATASET_DIR,
            unzip=True
        )
        print("[SUCCESS] Dataset downloaded and extracted!")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def organize_dataset():
    """
    Organize downloaded files into the expected folder structure.
    
    Expected structure after organization:
        dataset/
        ├── HAM10000_images/          # All 10,015 images
        └── HAM10000_metadata.csv     # Labels and metadata
    """
    print("\n[INFO] Organizing dataset files...")
    
    # Create images directory
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    
    # Find and move metadata CSV
    csv_patterns = [
        os.path.join(config.DATASET_DIR, 'HAM10000_metadata.csv'),
        os.path.join(config.DATASET_DIR, 'HAM10000_metadata'),
        os.path.join(config.DATASET_DIR, '**', 'HAM10000_metadata.csv'),
    ]
    
    csv_found = False
    for pattern in csv_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            dst = config.METADATA_FILE
            if src != dst:
                shutil.copy2(src, dst)
            print(f"  Metadata CSV: {dst}")
            csv_found = True
            break
    
    if not csv_found:
        print("  [WARNING] HAM10000_metadata.csv not found!")
        print("  Please ensure it's placed at:", config.METADATA_FILE)
    
    # Find and move image files
    # HAM10000 images may be in subdirectories
    image_patterns = [
        os.path.join(config.DATASET_DIR, '**', '*.jpg'),
        os.path.join(config.DATASET_DIR, '**', '*.jpeg'),
        os.path.join(config.DATASET_DIR, '**', '*.png'),
    ]
    
    image_count = 0
    for pattern in image_patterns:
        for src in glob.glob(pattern, recursive=True):
            filename = os.path.basename(src)
            dst = os.path.join(config.IMAGES_DIR, filename)
            
            # Skip if already in the correct location
            if os.path.abspath(src) == os.path.abspath(dst):
                image_count += 1
                continue
            
            # Copy image to images directory
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                image_count += 1
    
    print(f"  Total images organized: {image_count}")
    
    # Verify
    actual_images = len(glob.glob(os.path.join(config.IMAGES_DIR, '*.*')))
    print(f"  Images in {config.IMAGES_DIR}: {actual_images}")
    
    if actual_images >= 10000:
        print("\n[SUCCESS] Dataset organized successfully!")
        return True
    else:
        print(f"\n[WARNING] Expected ~10,015 images, found {actual_images}")
        print("  Some images may be missing.")
        return actual_images > 0


def verify_dataset():
    """
    Verify that the dataset is properly set up.
    """
    print("\n[INFO] Verifying dataset...")
    
    # Check metadata file
    if os.path.exists(config.METADATA_FILE):
        import pandas as pd
        df = pd.read_csv(config.METADATA_FILE)
        print(f"  ✓ Metadata file found: {len(df)} entries")
        print(f"    Classes: {df['dx'].unique().tolist()}")
    else:
        print(f"  ✗ Metadata file NOT found: {config.METADATA_FILE}")
        return False
    
    # Check images
    image_count = len(glob.glob(os.path.join(config.IMAGES_DIR, '*.*')))
    if image_count > 0:
        print(f"  ✓ Images found: {image_count}")
    else:
        print(f"  ✗ No images found in: {config.IMAGES_DIR}")
        return False
    
    print("\n[SUCCESS] Dataset verification passed!")
    return True


def print_manual_instructions():
    """Print instructions for manual dataset download."""
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
If the automatic download doesn't work, follow these steps:

1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

2. Click "Download" (you'll need a free Kaggle account)

3. Extract the downloaded ZIP file

4. Place the files in the following structure:
   
   dataset/
   ├── HAM10000_images/          ← All .jpg images go here
   │   ├── ISIC_0024306.jpg
   │   ├── ISIC_0024307.jpg
   │   └── ... (10,015 images)
   └── HAM10000_metadata.csv     ← Metadata CSV file

5. Run this script again with --verify to check:
   python download_dataset.py --verify
""")
    print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and setup the HAM10000 dataset'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Only verify the existing dataset without downloading'
    )
    parser.add_argument(
        '--organize', action='store_true',
        help='Only organize already downloaded files'
    )
    parser.add_argument(
        '--manual', action='store_true',
        help='Show manual download instructions'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  HAM10000 DATASET SETUP")
    print("=" * 60)
    
    # Ensure directories exist
    config.ensure_directories()
    
    if args.manual:
        print_manual_instructions()
    elif args.verify:
        verify_dataset()
    elif args.organize:
        organize_dataset()
        verify_dataset()
    else:
        # Full download pipeline
        success = download_from_kaggle()
        
        if success:
            organize_dataset()
            verify_dataset()
        else:
            print_manual_instructions()
