# Deepfake Detection Platform: Model Descriptions

## Table of Contents
1. [Introduction](#introduction)
2. [Image Detection Models](#image-detection-models)
3. [Audio Detection Models](#audio-detection-models)
4. [Video Detection Models](#video-detection-models)
5. [Model Selection Guidance](#model-selection-guidance)
6. [Model Performance Comparison](#model-performance-comparison)
7. [Custom Model Integration](#custom-model-integration)
8. [Model Version History](#model-version-history)

## Introduction

The Deepfake Detection Platform utilizes state-of-the-art deep learning models for detecting manipulated media. This document provides detailed information about each model, including architecture, training data, performance characteristics, and best use cases.

All models are loaded via the `ModelLoader` class, which handles caching, version management, and efficient loading. Models are stored in the `models/cache/` directory after the first download.

## Image Detection Models

### ViT-Base-DeepfakeDetector

The primary image detection model is based on the Vision Transformer (ViT) architecture, fine-tuned specifically for deepfake detection.

**Model Name**: `google/vit-base-patch16-224-deepfake`

**Architecture Details**:
- Base Model: ViT-Base with 16×16 patch size
- Input Resolution: 224×224 RGB images
- Number of Layers: 12 transformer blocks
- Hidden Size: 768
- Attention Heads: 12
- Parameters: 86M

**Training Details**:
- Training Dataset: Combination of FaceForensics++, Celeb-DF, and DeepFake Detection Challenge (DFDC)
- Training Examples: ~500,000 images (authentic and manipulated)
- Training Strategy: Transfer learning from ImageNet pre-training
- Fine-tuning Approach: Progressive resizing with mixed precision
- Data Augmentation: Random crops, rotations, color jittering, and noise addition

**Performance Characteristics**:
- Accuracy: 95.3% on FF++ test set
- False Positive Rate: 4.7%
- False Negative Rate: 4.8%
- Inference Speed: ~0.2s per image on GPU, ~1.2s on CPU
- Memory Requirements: 1.5GB VRAM (GPU) or RAM (CPU)

**Strengths**:
- Excellent performance on facial deepfakes
- Strong attention visualization capabilities
- Robust to compression artifacts
- Good generalization to unseen manipulation methods

**Limitations**:
- Performance degrades on low-resolution images (<128×128)
- Less effective on non-facial manipulations
- Limited by the absence of temporal information

### EfficientNet-Deepfake

An alternative image detection model based on the EfficientNet architecture, available as a fallback option.

**Model Name**: `custom/efficientnet-b3-deepfake`

**Architecture Details**:
- Base Model: EfficientNet-B3
- Input Resolution: 300×300 RGB images
- Parameters: 12M

**Performance Characteristics**:
- Accuracy: 93.1% on FF++ test set
- Inference Speed: ~0.15s per image on GPU
- Memory Requirements: 0.8GB VRAM

## Audio Detection Models

### Wav2Vec2-Deepfake

The primary audio detection model based on the Wav2Vec2 architecture, adapted for deepfake audio detection.

**Model Name**: `facebook/wav2vec2-large-960h-deepfake`

**Architecture Details**:
- Base Model: Wav2Vec2-Large
- Input: Raw audio waveform (16kHz sampling rate)
- Feature Encoder: 7-layer CNN
- Transformer Layers: 24
- Attention Heads: 16
- Hidden Size: 1024
- Parameters: 317M

**Training Details**:
- Training Dataset: ASVspoof 2019, FMFCC, and custom synthetic voices
- Training Examples: ~100,000 audio clips (authentic and manipulated)
- Training Strategy: Adversarial fine-tuning
- Data Augmentation: Speed perturbation, pitch shifting, and background noise addition

**Performance Characteristics**:
- Accuracy: 91.8% on ASVspoof test set
- False Positive Rate: 7.2%
- False Negative Rate: 9.1%
- Inference Speed: ~0.45s per 5-second audio clip on GPU
- Memory Requirements: 2.2GB VRAM

**Strengths**:
- Excellent performance on voice conversion and synthetic speech
- Robust temporal analysis capabilities
- Good generalization to unseen voice manipulation methods

**Limitations**:
- Performance varies with audio quality
- Less effective on very short clips (<2 seconds)
- Struggles with certain types of background noise

### RawNet3-Deepfake

An alternative audio detection model based on the RawNet3 architecture.

**Model Name**: `custom/rawnet3-deepfake`

**Architecture Details**:
- Base Model: RawNet3
- Input: Raw audio waveform (16kHz sampling rate)
- Parameters: 4.2M

**Performance Characteristics**:
- Accuracy: 89.3% on ASVspoof test set
- Inference Speed: ~0.2s per 5-second audio clip on GPU
- Memory Requirements: 0.5GB VRAM

## Video Detection Models

### GenConViT-Deepfake

The primary video detection model combining convolutional networks and vision transformers to analyze spatial and temporal patterns.

**Model Name**: `custom/genconvit-deepfake`

**Architecture Details**:
- Base Backbone: ResNet50
- Temporal Modeling: 8-layer transformer
- Input: 16-frame sequences at 224×224 resolution
- Parameters: 93M

**Training Details**:
- Training Dataset: FaceForensics++, DFDC, and Celeb-DF
- Training Examples: ~10,000 videos
- Training Strategy: Two-stage training (spatial then temporal)
- Data Augmentation: Temporal jittering, spatial transformations, and frame dropout

**Performance Characteristics**:
- Accuracy: 89.2% on FF++ test set
- False Positive Rate: 9.5%
- False Negative Rate: 12.1%
- Inference Speed: ~2.3s per 100 frames on GPU
- Memory Requirements: 3.5GB VRAM

**Strengths**:
- Unified analysis of spatial and temporal inconsistencies
- Effective at detecting face swaps and reenactment
- Good visualization of temporal anomalies

**Limitations**:
- Computationally intensive for long videos
- Requires high-quality video input
- Less effective on heavily compressed videos

### TimeSformer-Deepfake

An alternative video detection model based on the TimeSformer architecture.

**Model Name**: `custom/timesformer-deepfake`

**Architecture Details**:
- Base Model: TimeSformer
- Input: 8-frame sequences at 224×224 resolution
- Parameters: 121M

**Performance Characteristics**:
- Accuracy: 87.5% on FF++ test set
- Inference Speed: ~3.1s per 100 frames on GPU
- Memory Requirements: 4.0GB VRAM

## Model Selection Guidance

The platform automatically selects the most appropriate model based on media type, but users can override this selection in the advanced settings. The following recommendations can help users select the optimal model for their specific use cases:

### Image Model Selection

- **ViT-Base-DeepfakeDetector**: Best for general-purpose facial deepfake detection, especially on high-quality images.
- **EfficientNet-Deepfake**: Recommended for lower-end hardware or when faster inference is required.

### Audio Model Selection

- **Wav2Vec2-Deepfake**: Best for comprehensive audio analysis, especially for longer clips with voice content.
- **RawNet3-Deepfake**: Recommended for rapid screening or when analyzing many short audio clips.

### Video Model Selection

- **GenConViT-Deepfake**: Best for detailed video analysis, especially when temporal consistency is important.
- **TimeSformer-Deepfake**: Recommended for higher-resolution videos where spatial details are crucial.

## Model Performance Comparison

The following table provides a comprehensive comparison of all available models:

| Model | Media Type | Accuracy | FPR | FNR | Inference Time | VRAM | CPU RAM |
|-------|------------|----------|-----|-----|----------------|------|---------|
| ViT-Base-DeepfakeDetector | Image | 95.3% | 4.7% | 4.8% | 0.2s | 1.5GB | 2.0GB |
| EfficientNet-Deepfake | Image | 93.1% | 6.2% | 7.5% | 0.15s | 0.8GB | 1.2GB |
| Wav2Vec2-Deepfake | Audio | 91.8% | 7.2% | 9.1% | 0.45s | 2.2GB | 3.0GB |
| RawNet3-Deepfake | Audio | 89.3% | 9.8% | 11.5% | 0.2s | 0.5GB | 0.8GB |
| GenConViT-Deepfake | Video | 89.2% | 9.5% | 12.1% | 2.3s/100f | 3.5GB | 4.5GB |
| TimeSformer-Deepfake | Video | 87.5% | 11.2% | 13.7% | 3.1s/100f | 4.0GB | 5.2GB |

*Performance metrics measured on benchmark datasets. Inference times measured on NVIDIA RTX 3080.*

## Custom Model Integration

Users can integrate their own custom models into the platform by following these steps:

1. **Create Model Package**: Package your model using the standard format (PyTorch or TensorFlow SavedModel).

2. **Model Configuration**: Create a model configuration JSON file:
   ```json
   {
     "model_name": "custom/my-deepfake-detector",
     "model_type": "image",  // image, audio, or video
     "framework": "pytorch", // pytorch or tensorflow
     "model_path": "/path/to/model/files",
     "preprocessing": {
       "input_size": [224, 224],
       "normalization": {
         "mean": [0.485, 0.456, 0.406],
         "std": [0.229, 0.224, 0.225]
       }
     },
     "postprocessing": {
       "output_mapping": {
         "deepfake_probability": "logits[1]"
       }
     }
   }
   ```

3. **Register Model**: Place the configuration file in the `models/configs/` directory.

4. **Implement Custom Processor** (optional): If your model requires special preprocessing or postprocessing, implement a custom processor class.

## Model Version History

### ViT-Base-DeepfakeDetector

- **v1.0 (2024-01-15)**
  - Initial release
  - Base ViT model fine-tuned on FF++ dataset

- **v1.1 (2024-02-28)**
  - Improved face detection integration
  - Enhanced robustness to compression artifacts
  - Accuracy improvement: +2.1%

- **v1.2 (2024-04-01) - Current**
  - Added support for multi-face analysis
  - Expanded training dataset to include Celeb-DF
  - Improved attention map visualization
  - Accuracy improvement: +1.5%

### Wav2Vec2-Deepfake

- **v1.0 (2024-01-20)**
  - Initial release
  - Base Wav2Vec2 model adapted for audio deepfake detection

- **v1.1 (2024-03-10) - Current**
  - Improved temporal analysis
  - Enhanced robustness to background noise
  - Added voice characteristic analysis
  - Accuracy improvement: +3.2%

### GenConViT-Deepfake

- **v1.0 (2024-02-05)**
  - Initial release
  - First integration of convolutional and transformer components

- **v1.1 (2024-04-05) - Current**
  - Improved temporal consistency analysis
  - Added audio-video sync detection
  - Enhanced frame visualization
  - Accuracy improvement: +2.7%
