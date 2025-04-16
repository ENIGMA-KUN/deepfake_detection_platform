#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for audio deepfake detection results.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.detection_result import AudioFeatures, DetectionResult, TimeSegment

# Configure logger
logger = logging.getLogger(__name__)


def visualize_audio_detection(
    audio_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    show_waveform: bool = True,
    show_spectrogram: bool = True,
    highlight_segments: bool = True,
    colormap: str = 'viridis',
    output_format: str = 'png',
    dpi: int = 100
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Create visualization for audio detection results.
    
    Args:
        audio_path: Path to the original audio file
        result: Detection result to visualize
        output_path: Path to save the visualization (None = don't save)
        show_waveform: Whether to show waveform visualization
        show_spectrogram: Whether to show spectrogram visualization
        highlight_segments: Whether to highlight detected segments
        colormap: Colormap to use for spectrogram
        output_format: Image format for saving
        dpi: DPI for output image
        
    Returns:
        Tuple of (waveform, spectrogram) visualizations if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Create figure with appropriate subplots
        if show_waveform and show_spectrogram:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
            if show_spectrogram:
                ax2 = ax1
        
        # Plot waveform if requested
        if show_waveform:
            _plot_waveform(ax1, audio_data, sr, result)
        
        # Plot spectrogram if requested
        if show_spectrogram:
            if show_waveform:
                spec_ax = ax2
            else:
                spec_ax = ax1
            _plot_spectrogram(spec_ax, audio_data, sr, result, colormap)
        
        # Highlight detected segments if requested
        if highlight_segments and result.time_segments:
            if show_waveform:
                _highlight_segments(ax1, result.time_segments)
            if show_spectrogram:
                if show_waveform:
                    _highlight_segments(ax2, result.time_segments)
                else:
                    _highlight_segments(ax1, result.time_segments)
        
        # Add title with overall result
        if result.is_deepfake:
            title = f"DEEPFAKE DETECTED (Confidence: {result.confidence_score:.2f})"
            title_color = 'red'
        else:
            title = f"LIKELY AUTHENTIC (Confidence: {1 - result.confidence_score:.2f})"
            title_color = 'green'
        
        fig.suptitle(title, fontsize=16, color=title_color)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, format=output_format, dpi=dpi)
            logger.info(f"Visualization saved to {output_path}")
        
        # Get image data from figure
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        if show_waveform and show_spectrogram:
            # Split image into waveform and spectrogram parts
            height = image_data.shape[0]
            waveform = image_data[:height//2, :, :]
            spectrogram = image_data[height//2:, :, :]
            return waveform, spectrogram
        else:
            # Return the single visualization
            return image_data, None
        
    except Exception as e:
        logger.error(f"Error creating audio visualization: {e}")
        return None


def _plot_waveform(
    ax: Any,
    audio_data: np.ndarray,
    sr: int,
    result: DetectionResult
) -> None:
    """Plot audio waveform.
    
    Args:
        ax: Matplotlib axis
        audio_data: Audio data as numpy array
        sr: Sample rate
        result: Detection result
    """
    import librosa.display
    
    # Calculate time array
    times = np.arange(len(audio_data)) / sr
    
    # Plot waveform
    ax.plot(times, audio_data, color='blue', alpha=0.7)
    
    # Set axis labels and limits
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.set_xlim(0, len(audio_data) / sr)
    
    # Add grid
    ax.grid(True, alpha=0.3)


def _plot_spectrogram(
    ax: Any,
    audio_data: np.ndarray,
    sr: int,
    result: DetectionResult,
    colormap: str = 'viridis'
) -> None:
    """Plot audio spectrogram.
    
    Args:
        ax: Matplotlib axis
        audio_data: Audio data as numpy array
        sr: Sample rate
        result: Detection result
        colormap: Colormap to use
    """
    import librosa
    import librosa.display
    
    # Calculate spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data)), 
        ref=np.max
    )
    
    # Plot spectrogram
    img = librosa.display.specshow(
        D, 
        y_axis='log', 
        x_axis='time',
        sr=sr,
        ax=ax,
        cmap=colormap
    )
    
    # Add colorbar
    import matplotlib.pyplot as plt
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    
    # Set title
    ax.set_title('Spectrogram')


def _highlight_segments(
    ax: Any,
    time_segments: List[TimeSegment],
    alpha: float = 0.3
) -> None:
    """Highlight detected segments in the visualization.
    
    Args:
        ax: Matplotlib axis
        time_segments: List of detected time segments
        alpha: Transparency for highlighting
    """
    import matplotlib.patches as patches
    
    # Define colors for different confidence levels
    colors = {
        "high": 'red',       # High confidence
        "medium": 'orange',  # Medium confidence
        "low": 'yellow'      # Low confidence
    }
    
    for segment in time_segments:
        # Determine color based on confidence
        if segment.confidence >= 0.7:
            color = colors["high"]
        elif segment.confidence >= 0.5:
            color = colors["medium"]
        else:
            color = colors["low"]
        
        # Add rectangle patch
        rect = patches.Rectangle(
            (segment.start_time, ax.get_ylim()[0]), 
            segment.end_time - segment.start_time, 
            ax.get_ylim()[1] - ax.get_ylim()[0],
            linewidth=1, 
            edgecolor=color, 
            facecolor=color, 
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Add label at the top
        label = f"{segment.label} ({segment.confidence:.2f})"
        ax.text(
            segment.start_time, 
            ax.get_ylim()[1] * 0.95, 
            label,
            fontsize=8,
            color=color,
            backgroundcolor='white',
            alpha=0.7
        )


def create_audio_analysis_grid(
    audio_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    output_format: str = 'png',
    dpi: int = 150
) -> Optional[np.ndarray]:
    """Create a grid visualization comparing different audio analysis components.
    
    Args:
        audio_path: Path to the original audio file
        result: Detection result
        output_path: Path to save the visualization (None = don't save)
        output_format: Image format for saving
        dpi: DPI for output image
        
    Returns:
        Grid visualization as numpy array if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Create figure with grid layout
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Create axes for different visualizations
        ax_waveform = fig.add_subplot(gs[0, :])  # Waveform (full width)
        ax_spec = fig.add_subplot(gs[1, 0])      # Regular spectrogram
        ax_mel = fig.add_subplot(gs[1, 1])       # Mel spectrogram
        ax_chroma = fig.add_subplot(gs[2, 0])    # Chroma features
        ax_info = fig.add_subplot(gs[2, 1])      # Text information
        
        # Plot waveform
        _plot_waveform(ax_waveform, audio_data, sr, result)
        
        # Highlight detected segments in waveform
        if result.time_segments:
            _highlight_segments(ax_waveform, result.time_segments)
        
        # Plot regular spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)), 
            ref=np.max
        )
        librosa.display.specshow(
            D, 
            y_axis='log', 
            x_axis='time',
            sr=sr,
            ax=ax_spec,
            cmap='viridis'
        )
        ax_spec.set_title('Spectrogram')
        
        # Plot mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db,
            y_axis='mel',
            x_axis='time',
            sr=sr,
            ax=ax_mel,
            cmap='magma'
        )
        ax_mel.set_title('Mel Spectrogram')
        
        # Plot chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        librosa.display.specshow(
            chroma,
            y_axis='chroma',
            x_axis='time',
            sr=sr,
            ax=ax_chroma,
            cmap='coolwarm'
        )
        ax_chroma.set_title('Chroma Features')
        
        # Highlight detected segments in other plots
        if result.time_segments:
            _highlight_segments(ax_spec, result.time_segments)
            _highlight_segments(ax_mel, result.time_segments)
            _highlight_segments(ax_chroma, result.time_segments)
        
        # Add information text
        ax_info.axis('off')  # Turn off axis
        
        # Overall result
        if result.is_deepfake:
            result_text = f"DEEPFAKE DETECTED\nConfidence: {result.confidence_score:.2f}"
            result_color = 'red'
        else:
            result_text = f"LIKELY AUTHENTIC\nConfidence: {1 - result.confidence_score:.2f}"
            result_color = 'green'
        
        ax_info.text(0.05, 0.95, result_text, fontsize=14, color=result_color,
                    va='top', transform=ax_info.transAxes)
        
        # Add detected categories
        if result.categories:
            categories_text = "Detected categories:"
            ax_info.text(0.05, 0.85, categories_text, fontsize=12,
                        va='top', transform=ax_info.transAxes)
            
            for i, category in enumerate(result.categories):
                ax_info.text(0.1, 0.8 - i * 0.05, f"- {category.value}", fontsize=11,
                            va='top', transform=ax_info.transAxes)
        
        # Add synthesis markers if available
        if result.audio_features and result.audio_features.synthesis_markers:
            markers_text = "Synthesis markers:"
            ax_info.text(0.05, 0.6, markers_text, fontsize=12,
                        va='top', transform=ax_info.transAxes)
            
            for i, marker in enumerate(result.audio_features.synthesis_markers):
                ax_info.text(0.1, 0.55 - i * 0.05, f"- {marker}", fontsize=11,
                            va='top', transform=ax_info.transAxes)
        
        # Add file information
        file_info = (
            f"Filename: {os.path.basename(audio_path)}\n"
            f"Duration: {len(audio_data) / sr:.2f} seconds\n"
            f"Sample rate: {sr} Hz"
        )
        ax_info.text(0.05, 0.3, file_info, fontsize=11,
                    va='top', transform=ax_info.transAxes)
        
        # Add title
        fig.suptitle("Audio Deepfake Analysis", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, format=output_format, dpi=dpi)
            logger.info(f"Analysis grid saved to {output_path}")
        
        # Get image data from figure
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error creating audio analysis grid: {e}")
        return None


def create_time_domain_features_plot(
    audio_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    output_format: str = 'png',
    dpi: int = 100
) -> Optional[np.ndarray]:
    """Create a visualization of time-domain features.
    
    Args:
        audio_path: Path to the original audio file
        result: Detection result
        output_path: Path to save the visualization (None = don't save)
        output_format: Image format for saving
        dpi: DPI for output image
        
    Returns:
        Visualization as numpy array if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Plot waveform
        _plot_waveform(axes[0], audio_data, sr, result)
        
        # Plot RMS energy
        rms = librosa.feature.rms(y=audio_data)[0]
        frames = np.arange(len(rms))
        t = librosa.frames_to_time(frames, sr=sr)
        axes[1].plot(t, rms, color='green')
        axes[1].set_ylabel('RMS Energy')
        axes[1].set_title('Root Mean Square Energy')
        
        # Plot zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        t = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
        axes[2].plot(t, zcr, color='purple')
        axes[2].set_ylabel('ZCR')
        axes[2].set_title('Zero Crossing Rate')
        
        # Plot spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        t = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
        axes[3].plot(t, spectral_centroids, color='orange')
        axes[3].set_ylabel('Frequency (Hz)')
        axes[3].set_title('Spectral Centroid')
        axes[3].set_xlabel('Time (s)')
        
        # Highlight detected segments in all plots
        if result.time_segments:
            for ax in axes:
                _highlight_segments(ax, result.time_segments)
        
        # Add overall title
        if result.is_deepfake:
            title = f"DEEPFAKE DETECTED (Confidence: {result.confidence_score:.2f})"
            title_color = 'red'
        else:
            title = f"LIKELY AUTHENTIC (Confidence: {1 - result.confidence_score:.2f})"
            title_color = 'green'
        
        fig.suptitle(title, fontsize=16, color=title_color)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, format=output_format, dpi=dpi)
            logger.info(f"Time-domain features plot saved to {output_path}")
        
        # Get image data from figure
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error creating time-domain features plot: {e}")
        return None