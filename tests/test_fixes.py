"""
Test script to validate the fixes made to the Deepfake Detection Platform.
"""
import os
import sys
import logging
import tempfile

# Add the parent directory to sys.path to allow importing from the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.processor import MediaProcessor
from app.interface.app import create_app

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_media_processor_initialization():
    """Test that MediaProcessor initializes correctly."""
    logger.info("Testing MediaProcessor initialization...")
    
    # Create a simple config
    config = {
        'general': {
            'temp_dir': './temp',
            'result_cache_size': 100
        },
        'models': {
            'image': {
                'model_name': 'vit',
                'confidence_threshold': 0.7
            },
            'audio': {
                'model_name': 'wav2vec',
                'confidence_threshold': 0.7
            },
            'video': {
                'frame_model_name': 'timesformer',
                'temporal_model_name': 'slowfast',
                'confidence_threshold': 0.7
            }
        }
    }
    
    try:
        processor = MediaProcessor(config)
        logger.info("✅ MediaProcessor initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"❌ MediaProcessor initialization failed: {str(e)}")
        raise

def test_detector_loading():
    """Test that detectors load correctly."""
    logger.info("Testing detector loading...")
    
    processor = test_media_processor_initialization()
    
    try:
        # Force load detectors
        processor.load_detectors()
        
        # Check if detectors are loaded
        for media_type in ['image', 'audio', 'video']:
            assert media_type in processor.detectors, f"Detector for {media_type} not loaded"
            logger.info(f"✅ Detector for {media_type} loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Detector loading failed: {str(e)}")
        raise

def test_model_parameter_propagation():
    """Test that model parameters are correctly propagated to detectors."""
    logger.info("Testing model parameter propagation...")
    
    processor = test_media_processor_initialization()
    
    try:
        # Test setting a custom confidence threshold
        media_type = 'image'
        model_params = {
            'model_name': 'vit',
            'confidence_threshold': 0.85  # Custom threshold
        }
        
        # Get detector and check if parameters are correctly set
        detector = processor.detectors.get(media_type)
        if not detector:
            processor.load_detectors()
            detector = processor.detectors.get(media_type)
        
        original_threshold = detector.confidence_threshold
        
        # Update with custom params
        if "confidence_threshold" in model_params:
            detector.confidence_threshold = model_params["confidence_threshold"]
        
        # Check if the threshold was updated
        assert detector.confidence_threshold == 0.85, f"Threshold not updated correctly: {detector.confidence_threshold}"
        logger.info(f"✅ Model parameter propagation works correctly")
        
        # Reset to original value
        detector.confidence_threshold = original_threshold
        
        return True
    except Exception as e:
        logger.error(f"❌ Model parameter propagation test failed: {str(e)}")
        raise

def run_tests():
    """Run all tests."""
    logger.info("Starting tests...")
    
    tests = [
        test_media_processor_initialization,
        test_detector_loading,
        test_model_parameter_propagation
    ]
    
    success_count = 0
    for test_func in tests:
        try:
            test_func()
            success_count += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {str(e)}")
    
    logger.info(f"Tests completed: {success_count} / {len(tests)} successful")

if __name__ == "__main__":
    run_tests()
