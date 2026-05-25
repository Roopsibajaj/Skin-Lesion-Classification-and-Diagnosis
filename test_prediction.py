import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.model import load_trained_model
from src.predict import predict_image

def main():
    print("[INFO] Starting prediction test...")
    # Load model
    model = load_trained_model(config.MODEL_SAVE_PATH)
    if model is None:
        print("[ERROR] Model could not be loaded!")
        return

    # Create a dummy image
    dummy_img_path = os.path.join(config.UPLOADS_DIR, 'test_dummy.jpg')
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    
    # Create 300x300 BGR image (random color)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    cv2.imwrite(dummy_img_path, img)
    print(f"  Saved dummy image to: {dummy_img_path}")

    try:
        # Run prediction
        results = predict_image(model, dummy_img_path, generate_heatmap=True)
        print("[SUCCESS] Prediction completed!")
        print(f"  Prediction: {results['predicted_class_full']}")
        print(f"  Confidence: {results['confidence_percentage']}")
    except Exception as e:
        print("[ERROR] Prediction failed with exception:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
