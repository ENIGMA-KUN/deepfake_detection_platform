import os
import random
from pathlib import Path

import numpy as np

from detectors.image_detector.vit_detector import ViTImageDetector

TEST_IMAGE_DIR = Path(__file__).parent / "test_data" / "images"


def get_random_image_path() -> str:
    images = list(TEST_IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError("No test images found in test_data/images")
    return str(random.choice(images))


def test_vit_image_detector_prediction():
    image_path = get_random_image_path()

    detector = ViTImageDetector(model_name="google/vit-base-patch16-224", confidence_threshold=0.5)
    result = detector.detect(image_path)

    assert isinstance(result, dict)
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "is_deepfake" in result
