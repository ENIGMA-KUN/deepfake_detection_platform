# Deepfake Detection Platform: Detector Details

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Base Detector](#base-detector)
3. [Image Detector](#image-detector)
4. [Audio Detector](#audio-detector)
5. [Video Detector](#video-detector)
6. [Model Internals](#model-internals)
7. [Custom Detection Models](#custom-detection-models)
8. [Ensemble Methods](#ensemble-methods)
9. [Performance Metrics](#performance-metrics)

## Architecture Overview

The Deepfake Detection Platform employs a modular architecture with specialized detectors for different media types (image, audio, and video). All detectors share a common base interface to provide a consistent API while implementing media-specific detection algorithms.

```
detectors/
├── base_detector.py           # Abstract base class with common functionality
├── image_detector/
│   ├── vit_detector.py        # ViT-based image detector
│   └── ensemble.py            # Ensemble of multiple image detectors
├── audio_detector/
│   ├── wav2vec_detector.py    # Wav2Vec2-based audio detector
│   └── spectrogram_analyzer.py # Complementary spectrogram analysis
└── video_detector/
    ├── genconvit.py           # GenConViT hybrid detector
    └── frame_analyzer.py      # Frame-by-frame analysis utilities
```

## Base Detector

All detectors extend the `BaseDetector` abstract class, which defines the common interface and provides utility methods for result formatting and confidence normalization.

### Interface

```python
class BaseDetector(ABC):
    def __init__(self, model_name: str, confidence_threshold: float = 0.5):
        """Initialize detector with model name and confidence threshold."""
        pass
    
    @abstractmethod
    def detect(self, media_path: str) -> Dict[str, Any]:
        """
        Detect if media is a deepfake.
        
        Args:
            media_path: Path to media file
            
        Returns:
            Dictionary with detection result, including:
            - is_deepfake: Boolean indicating if media is a deepfake
            - confidence: Confidence score (0.0-1.0)
            - metadata: Additional detection metadata
        """
        pass
    
    def normalize_confidence(self, raw_score: float) -> float:
        """Normalize confidence score to range [0, 1]."""
        pass
    
    def format_result(self, is_deepfake: bool, confidence: float, metadata: Dict) -> Dict[str, Any]:
        """Format detection result into standardized output format."""
        pass
```

## Image Detector

The ViTImageDetector uses the Vision Transformer (ViT) architecture for detecting manipulated images, with a focus on facial deepfakes.

### Key Features

1. **Face-Focused Analysis**: Prioritizes face regions where manipulations are most common
2. **Patch-Based Processing**: Divides image into patches for transformer processing
3. **Attention Map Visualization**: Generates heatmaps showing suspicious regions
4. **Multi-Face Support**: Can analyze multiple faces in a single image
5. **Whole-Image Fallback**: Falls back to whole-image analysis when no faces are detected

### Detection Pipeline

1. **Face Detection**: Using MTCNN to identify and extract faces
2. **Preprocessing**: Resizing, normalization, and transformation
3. **Feature Extraction**: Using Vision Transformer to extract features
4. **Classification**: Binary classification (authentic/deepfake)
5. **Attention Map Generation**: Creating visual heatmaps of suspicious areas
6. **Result Aggregation**: Combining individual face results for multi-face images

### Technical Details

- **Model Architecture**: Vision Transformer (ViT-Base with 16×16 patches, 12 transformer layers)
- **Input Size**: 224×224 RGB images
- **Training Dataset**: Combination of FF++, CelebDF, and DFDC datasets
- **Fine-tuning Strategy**: Progressive resizing with mixed precision training
- **Augmentation**: Random cropping, flipping, color jittering, and noise addition

## Audio Detector

The Wav2VecAudioDetector uses the Wav2Vec2 model architecture, originally designed for speech recognition, adapted for deepfake detection.

### Key Features

1. **Temporal Analysis**: Segment-by-segment analysis to detect inconsistencies over time
2. **Spectrogram Analysis**: Complementary approach analyzing visual patterns in spectrograms
3. **Multi-level Detection**: Combines waveform and frequency-domain analysis
4. **Voice Characteristic Analysis**: Focuses on voice attributes like pitch, timbre, and articulation

### Detection Pipeline

1. **Audio Loading**: Loading and resampling audio to 16kHz
2. **Segmentation**: Dividing audio into overlapping segments
3. **Feature Extraction**: Using Wav2Vec2 to extract contextual representations
4. **Classification**: Per-segment classification and temporal consistency evaluation
5. **Spectrogram Analysis**: Generation and analysis of spectrograms for visual patterns
6. **Result Aggregation**: Combining temporal and spectrogram analyses

### Technical Details

- **Model Architecture**: Wav2Vec2-Large with 24 transformer layers
- **Input Format**: 16kHz mono audio
- **Training Dataset**: ASVspoof 2019, FMFCC, and in-house synthetic voices
- **Fine-tuning Strategy**: Adversarial training with voice conversion examples
- **Segment Duration**: 4-second overlapping windows with 2-second stride
- **Frequency Analysis**: Mel-frequency cepstral coefficients (MFCCs) and log-mel spectrograms

## Video Detector

The GenConViTVideoDetector combines a generalized convolutional network with vision transformers to analyze spatial and temporal aspects of videos.

### Key Features

1. **Frame-Level Analysis**: Individual frame analysis for spatial inconsistencies
2. **Temporal Consistency**: Analysis of consistency between consecutive frames
3. **Audio-Video Synchronization**: Detection of audio-video synchronization issues
4. **Multi-Modal Integration**: Combined analysis of visual and audio features
5. **Temporal Attention**: Special focus on temporal transitions between frames

### Detection Pipeline

1. **Video Decoding**: Extracting frames and audio
2. **Frame Sampling**: Selecting representative frames for analysis
3. **Face Tracking**: Tracking faces across frames
4. **Frame-Level Detection**: Analyzing each selected frame
5. **Temporal Analysis**: Calculating consistency between consecutive frames
6. **A/V Sync Analysis**: Measuring audio-visual synchronization
7. **Result Aggregation**: Combining frame-level, temporal, and A/V sync results

### Technical Details

- **Model Architecture**: Hybrid GenConViT (combination of ResNet50 backbone with transformer encoder)
- **Input Format**: Frame sequences at 224×224 resolution
- **Training Dataset**: FaceForensics++, DeepFake Detection Challenge, and Celeb-DF
- **Temporal Window**: Analysis of 16-frame sequences with 8-frame overlap
- **Optical Flow**: Dense optical flow calculation for motion consistency analysis
- **A/V Sync**: Feature alignment between audio and visual modalities with cross-attention

## Model Internals

### Vision Transformer (ViT)

The ViT model used in the image detector divides input images into fixed-size patches, linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard transformer encoder.

```
Input Image
    ↓
Patch Embedding + Position Embedding
    ↓
Transformer Encoder (12 layers)
    ↓
Classification Head
```

Key implementation details:
- Patch size: 16×16 pixels
- Hidden dimension: 768
- MLP dimension: 3072
- Attention heads: 12
- Dropout rate: 0.1

### Wav2Vec2

The Wav2Vec2 model used in the audio detector processes raw audio waveforms through a multi-layer CNN feature encoder followed by a transformer context network.

```
Raw Audio
    ↓
CNN Feature Encoder
    ↓
Transformer Context Network (24 layers)
    ↓
Classification Head
```

Key implementation details:
- Feature encoder: 7-layer CNN with 512 channels
- Context network: 24-layer transformer with 16 attention heads
- Hidden dimension: 1024
- Mask token rate during training: 0.065

### GenConViT

The GenConViT model used in the video detector combines convolutional layers for spatial feature extraction with transformers for temporal modeling.

```
Frame Sequence
    ↓
ResNet50 Backbone (per-frame)
    ↓
Temporal Feature Aggregation
    ↓
Transformer Encoder (8 layers)
    ↓
Classification Head
```

Key implementation details:
- Backbone: ResNet50 (pretrained on ImageNet)
- Temporal features: 16 frames
- Transformer layers: 8
- Attention heads: 8
- Hidden dimension: 512

## Custom Detection Models

The platform supports integration of custom detection models through the following steps:

1. **Implement Detector Class**: Create a new detector class that extends the appropriate base detector
2. **Interface Compliance**: Ensure the new detector implements all required methods
3. **Model Integration**: Load your custom model using the platform's model loader
4. **Registration**: Register the detector in the configuration file

Example of a custom image detector implementation:

```python
from detectors.image_detector.vit_detector import ViTImageDetector

class CustomImageDetector(ViTImageDetector):
    def __init__(self, model_name: str, confidence_threshold: float = 0.5):
        super().__init__(model_name, confidence_threshold)
        # Additional initialization
        
    def detect(self, media_path: str) -> Dict[str, Any]:
        # Custom detection implementation
        # ...
        return self.format_result(is_deepfake, confidence, metadata)
```

## Ensemble Methods

The platform implements ensemble detection methods to improve accuracy and robustness. Ensembles combine the predictions of multiple detector models using various strategies.

### Ensemble Types

1. **Majority Voting**: Each model votes, final decision based on majority
2. **Weighted Averaging**: Models' predictions weighted by confidence and reliability
3. **Stacking**: Using a meta-model to learn optimal combination of base models
4. **Boosting**: Sequentially training models to focus on previous models' weaknesses

### Implementation

```python
class EnsembleImageDetector(BaseDetector):
    def __init__(self, model_configs: List[Dict], ensemble_type: str = 'weighted_avg'):
        self.detectors = [
            ViTImageDetector(config['model_name'], config['threshold']) 
            for config in model_configs
        ]
        self.ensemble_type = ensemble_type
        
    def detect(self, media_path: str) -> Dict[str, Any]:
        # Collect individual detector results
        results = [detector.detect(media_path) for detector in self.detectors]
        
        # Apply ensemble method
        if self.ensemble_type == 'majority_vote':
            return self._majority_vote(results)
        elif self.ensemble_type == 'weighted_avg':
            return self._weighted_average(results)
        # ...
```

## Performance Metrics

The platform measures detector performance using the following metrics:

### Accuracy Metrics

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Percentage of true deepfakes among predicted deepfakes
- **Recall**: Percentage of detected deepfakes among all actual deepfakes
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve

### Efficiency Metrics

- **Inference Time**: Time required to analyze a single media file
- **Memory Usage**: Peak RAM usage during detection
- **GPU Memory**: VRAM required for model execution
- **Throughput**: Media files processed per minute

### Benchmark Results

| Detector           | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Inference Time |
|--------------------|----------|-----------|--------|----------|---------|----------------|
| ViTImageDetector   | 95.3%    | 94.1%     | 96.5%  | 95.3%    | 0.983   | 215ms          |
| Wav2VecAudioDetector | 91.8%  | 89.3%     | 93.6%  | 91.4%    | 0.957   | 450ms          |
| GenConViTVideoDetector | 89.2% | 87.5%    | 90.1%  | 88.8%    | 0.936   | 2.3s/100 frames |

*Benchmarks conducted on NVIDIA RTX 3080, AMD Ryzen 9 5900X, 32GB RAM
