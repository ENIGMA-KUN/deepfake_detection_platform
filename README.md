# Deepfake Detection Platform

A comprehensive platform for analyzing and detecting deepfakes in images, audio, and video media using advanced deep learning models.

## Features

- **Image Deepfake Detection**: Powered by Vision Transformer (ViT) models with region-specific analysis and heatmaps
- **Audio Deepfake Detection**: Uses Wav2Vec2 models and spectrogram analysis to detect manipulation in audio files
- **Video Deepfake Detection**: Implements GenConViT/TimeSformer hybrid approach with temporal consistency analysis
- **Tron Legacy-Inspired UI**: Modern, sleek interface with dedicated tabs for different media types
- **Comprehensive Reports**: Detailed analysis results with visualizations showing detected inconsistencies

## Installation

```bash
# Clone the repository
git clone https://github.com/enigma-kun/deepfake-detection-platform.git
cd deepfake-detection-platform

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python -m app.main
```

## Project Structure

```
deepfake-detection/
├── app/              # Main application code
│   ├── interface/    # UI components
│   ├── core/         # Core processing logic
│   └── utils/        # Utility functions
├── detectors/        # Deepfake detection modules
│   ├── image_detector/
│   ├── audio_detector/
│   └── video_detector/
├── models/           # Model definitions and loaders
├── data/             # Data preprocessing and augmentation
├── reports/          # Report templates and output
├── tests/            # Test cases
└── docs/             # Documentation
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.5+
- See `requirements.txt` for a complete list

## License

MIT
