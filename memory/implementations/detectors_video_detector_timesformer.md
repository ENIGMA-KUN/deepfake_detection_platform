# TimeSformerVideoDetector Implementation

## Overview
This file implements a video deepfake detector using the TimeSformer (Time-Space Transformer) model, which is designed to analyze temporal features in videos for deepfake detection. The implementation includes real model loading via the model_loader utility, frame extraction from videos, temporal analysis, and graceful fallback mechanisms.

## Key Components

### Model Loading
- Uses `model_loader.py` to load models from Hugging Face
- Includes API key validation for premium models
- Implements fallback to GenConViT model when premium models are unavailable
- Provides an advanced fallback mechanism called "Temporal Oracle" when no models can be loaded

### Video Processing
- Extracts frames from videos using OpenCV
- Processes frames through the model with proper preprocessing
- Handles both temporal analysis (for TimeSformer) and frame-by-frame analysis (for fallback models)
- Analyzes anomalies across video frames to identify potential deepfake regions

### Error Handling
- Comprehensive error handling throughout the detection pipeline
- Graceful degradation to simplified models when primary models fail
- Mock implementations that use video metadata to generate plausible results
- Detailed logging for diagnostics and troubleshooting

## Classes

### TimeSformerVideoDetector
- Extends BaseDetector
- Constructor parameters:
  - `model_name`: Name of the pretrained TimeSformer model to use (default: "timesformer")
  - `confidence_threshold`: Threshold for classifying video as deepfake (default: 0.5)
  - `device`: Device to run the model on ('cuda' or 'cpu')
- Main methods:
  - `load_model()`: Loads the TimeSformer model using model_loader
  - `detect(media_path)`: Main detection method that processes a video file
  - `_process_video(video_path)`: Processes video frames through model
  - `_process_video_mock(video_path)`: Fallback mock implementation
  - `_extract_frames(video_path)`: Extracts frames from video for processing
  - `_process_frames_temporal(frames)`: Processes frames with temporal awareness
  - `_process_single_frame(frame)`: Processes a single frame individually

## Input/Output Format

### Input
- Path to a video file (mp4, avi, mov, etc.)

### Output
- JSON object containing:
  - `is_deepfake`: Boolean indicating whether the video is a deepfake
  - `confidence`: Confidence score (0-1) of the detection
  - `model_name`: Name of the model used for detection
  - `processing_time`: Time taken to process the video
  - `frame_scores`: List of confidence scores for each frame
  - `timestamps`: List of timestamps for each analyzed frame
  - `anomaly_frames`: List of timestamps for frames flagged as anomalous

## Dependencies
- PyTorch for model inference
- OpenCV for video processing
- NumPy for numerical operations
- PIL for image processing
- Transformers for model loading

## Implementation Details
- Uses CUDA if available
- Extracts a fixed number of frames (8 by default) from videos
- Resizes frames to 224x224 for model compatibility
- Implements both temporal mode (whole clip analysis) and frame-by-frame mode
- Provides deterministic fallback for consistent results when model is unavailable
