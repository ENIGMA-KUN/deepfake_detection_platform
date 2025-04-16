# Deepfake Detection Platform: User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Interface](#using-the-interface)
   - [Image Analysis](#image-analysis)
   - [Audio Analysis](#audio-analysis)
   - [Video Analysis](#video-analysis)
   - [Reports](#reports)
4. [Understanding Results](#understanding-results)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

## Introduction

The Deepfake Detection Platform is a comprehensive tool designed to analyze media files (images, audio, and video) for signs of AI manipulation or "deepfake" content. Leveraging state-of-the-art deep learning models, the platform provides detailed analysis and visualization to help users determine the authenticity of digital content.

## Getting Started

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for video analysis)
- NVIDIA GPU with CUDA support recommended for faster processing
- 5GB free disk space

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection-platform.git
   cd deepfake-detection-platform
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python app/main.py
   ```

4. Access the web interface by navigating to:
   ```
   http://localhost:8050
   ```

## Using the Interface

The platform features a Tron Legacy-themed interface divided into tabs for different media types.

### Image Analysis

1. **Select the Image Tab**: Click on the "Image" tab in the navigation bar.

2. **Upload an Image**: Click the upload button or drag and drop an image file. Supported formats include JPG, PNG, and WEBP.

3. **Advanced Settings (Optional)**:
   - Adjust detection threshold: Higher values reduce false positives but may miss subtle manipulations
   - Select visualization type: Heatmap, region highlight, or side-by-side comparison

4. **Analyze**: Click the "Analyze Image" button and wait for processing to complete.

5. **View Results**: The analysis results will display below the upload area, showing:
   - Overall verdict (authentic or deepfake)
   - Confidence score
   - Visual heatmap highlighting potentially manipulated regions
   - Face detection results (if applicable)
   - Technical details about the detection

6. **Save Report**: Click "Generate Report" to create a detailed analysis report in HTML format.

### Audio Analysis

1. **Select the Audio Tab**: Click on the "Audio" tab in the navigation bar.

2. **Upload Audio**: Click the upload button or drag and drop an audio file. Supported formats include WAV, MP3, and M4A.

3. **Advanced Settings (Optional)**:
   - Adjust detection threshold
   - Select temporal analysis granularity
   - Enable/disable spectrogram analysis

4. **Analyze**: Click the "Analyze Audio" button and wait for processing to complete.

5. **View Results**: Results will display with:
   - Overall verdict (authentic or deepfake)
   - Confidence score
   - Temporal analysis graph showing manipulation probability over time
   - Spectrogram visualization with highlighted anomalies
   - Technical details about the detection

6. **Playback Controls**: Listen to the audio and observe how the manipulation probability changes in real-time.

7. **Save Report**: Click "Generate Report" to create a detailed analysis report.

### Video Analysis

1. **Select the Video Tab**: Click on the "Video" tab in the navigation bar.

2. **Upload Video**: Click the upload button or drag and drop a video file. Supported formats include MP4, AVI, and MOV.

3. **Advanced Settings (Optional)**:
   - Adjust detection threshold
   - Set frame sampling rate (analyzing every Nth frame)
   - Enable/disable audio-video sync analysis
   - Select visualization style

4. **Analyze**: Click the "Analyze Video" button and wait for processing to complete.
   - Note: Video analysis is computationally intensive and may take several minutes depending on video length and your hardware.

5. **View Results**: Results will display with:
   - Overall verdict (authentic or deepfake)
   - Confidence score
   - Frame-by-frame analysis showing manipulation probability
   - Temporal inconsistency visualization
   - Audio-video sync analysis
   - Highlighted frames with potential manipulation

6. **Interactive Timeline**: Scrub through the video timeline to see detection results for specific sections.

7. **Save Report**: Click "Generate Report" to create a detailed analysis report.

### Reports

1. **Select the Reports Tab**: Click on the "Reports" tab to view and manage previous analysis reports.

2. **Browse Reports**: View a list of all previously generated reports, sorted by date.

3. **Filter Reports**: Use the filter options to search by:
   - Media type (image, audio, video)
   - Date range
   - Detection result (deepfake or authentic)
   - Confidence threshold

4. **View Report**: Click on any report entry to view the detailed report.

5. **Export Options**: Export reports in various formats:
   - HTML (default)
   - PDF
   - JSON (for integration with other systems)

6. **Batch Operations**: Select multiple reports for batch export or deletion.

## Understanding Results

### Confidence Score

The confidence score (0-100%) indicates how certain the system is about its verdict:
- 0-20%: Low confidence - result should be treated with caution
- 20-50%: Moderate confidence
- 50-80%: High confidence
- 80-100%: Very high confidence

### Visual Indicators

- **Green**: Indicates authentic content or regions
- **Red**: Indicates potentially manipulated content or regions
- **Yellow**: Indicates uncertain areas that require closer inspection

### Detailed Metrics

- **Temporal Inconsistency**: Measures how consistent the manipulation is across time (for audio and video). Higher values indicate potential deepfakes.
- **Spatial Inconsistency**: Measures how consistent the manipulation is across different regions of an image.
- **A/V Sync Score**: Measures synchronization between audio and video. Desynchronization can indicate manipulation.

## Troubleshooting

### Common Issues

1. **Slow Processing**:
   - Enable GPU acceleration in config.yaml
   - Reduce video resolution or analyze fewer frames
   - Close other GPU-intensive applications

2. **Model Loading Errors**:
   - Check your internet connection (models are downloaded on first use)
   - Ensure you have sufficient disk space
   - Try restarting the application

3. **False Positives/Negatives**:
   - Adjust the confidence threshold
   - Try a different detector model (in Advanced Settings)
   - Ensure the media doesn't contain artifacts from compression or low quality

4. **Interface Not Loading**:
   - Check that port 8050 is not in use by another application
   - Clear your browser cache
   - Try a different browser

### Getting Support

If you encounter problems not covered here, please:
1. Check the project's GitHub issues page
2. Consult the detailed technical documentation
3. Submit a new issue with detailed reproduction steps

## FAQ

**Q: How accurate is the detection?**
A: Detection accuracy varies by media type and manipulation technique. Our image detector achieves ~95% accuracy on benchmark datasets, while audio and video detectors achieve ~90% and ~85% respectively. However, as deepfake technology evolves, so must detection methods.

**Q: Can the platform detect all types of deepfakes?**
A: The platform is designed to detect common manipulation techniques, but may not catch every type, especially newer methods. We regularly update our models to improve detection capabilities.

**Q: How is my data handled?**
A: All processing happens locally on your machine. No media files or analysis results are sent to external servers unless you explicitly enable cloud processing in the settings.

**Q: Can I integrate this with my own systems?**
A: Yes, the platform offers a REST API for integration. See the API documentation for details.

**Q: How often are detection models updated?**
A: We aim to release updated models quarterly. Check the settings panel for available updates.

**Q: Does detection work on compressed or low-quality media?**
A: Detection accuracy may be reduced on highly compressed or low-quality media. For best results, use the highest quality source available.

**Q: Can I train the system on my own data?**
A: Advanced users can fine-tune the models on custom datasets. See the detector_details.md documentation for instructions.
