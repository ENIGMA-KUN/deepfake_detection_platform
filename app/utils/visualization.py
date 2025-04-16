"""
Visualization utilities for the Deepfake Detection Platform.
Provides functions for visualizing detection results for different media types.
"""
import os
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont

class VisualizationManager:
    """
    Manager class for visualization of deepfake detection results.
    Provides methods for visualizing different media types and results.
    """
    
    def __init__(self, output_dir: str = None, 
                theme: str = "tron_legacy",
                dpi: int = 100):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualization outputs
            theme: Visual theme to use ('tron_legacy', 'default', etc.)
            dpi: DPI for saved images
        """
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    'reports', 'output')
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Set theme-specific colors
        self.theme = theme
        self.dpi = dpi
        
        if theme == "tron_legacy":
            self.colors = {
                "background": "#000000",
                "grid": "#0C141F",
                "primary": "#4CC9F0",
                "secondary": "#4361EE",
                "accent": "#3A0CA3",
                "text": "#CDE7FB",
                "authentic": "#0AFF16",
                "deepfake": "#F72585",
                "uncertain": "#FCA311"
            }
        else:
            # Default color scheme
            self.colors = {
                "background": "#FFFFFF",
                "grid": "#E0E0E0",
                "primary": "#1976D2",
                "secondary": "#388E3C",
                "accent": "#FBC02D",
                "text": "#212121",
                "authentic": "#00C853",
                "deepfake": "#D32F2F",
                "uncertain": "#FF9800"
            }
            
        self.logger.info(f"VisualizationManager initialized with theme: {theme}")
    
    def create_detection_overlay(self, image: Union[str, np.ndarray, Image.Image], 
                               detection_result: Dict[str, Any]) -> Image.Image:
        """
        Create a visual overlay on an image showing deepfake detection results.
        
        Args:
            image: Image file path, numpy array, or PIL Image
            detection_result: Detection result dictionary from detector
            
        Returns:
            PIL Image with detection visualization overlay
        """
        # Load image if needed
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise ValueError("Unsupported image type")
        
        # Create a draw object
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        width, height = img.size
        
        # Try to load a font, use default if unable
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw overall result banner
        is_deepfake = detection_result.get("is_deepfake", False)
        confidence = detection_result.get("confidence", 0.0)
        
        banner_color = self.colors["deepfake"] if is_deepfake else self.colors["authentic"]
        banner_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC"
        
        # Draw top banner
        banner_height = 30
        draw.rectangle([(0, 0), (width, banner_height)], fill=banner_color)
        text_width = draw.textlength(banner_text, font=font)
        draw.text(
            ((width - text_width) // 2, 5),
            banner_text,
            fill="#FFFFFF",
            font=font
        )
        
        # Draw confidence text
        conf_text = f"Confidence: {confidence:.2f}"
        draw.text(
            (10, banner_height + 10),
            conf_text,
            fill=banner_color,
            font=small_font
        )
        
        # Draw face boxes if available
        if "details" in detection_result:
            details = detection_result["details"]
            
            # Check if there are face results
            if "face_results" in details:
                for face in details["face_results"]:
                    # Draw bounding box
                    if "bounding_box" in face:
                        bbox = face["bounding_box"]
                        x1, y1, x2, y2 = bbox[:4]
                        
                        face_conf = face.get("confidence", 0.0)
                        bbox_color = self._get_confidence_color(face_conf)
                        
                        # Draw rectangle
                        draw.rectangle(
                            [(x1, y1), (x2, y2)],
                            outline=bbox_color,
                            width=2
                        )
                        
                        # Draw confidence text
                        face_text = f"{face_conf:.2f}"
                        draw.text(
                            (x1 + 5, y1 + 5),
                            face_text,
                            fill=bbox_color,
                            font=small_font
                        )
        
        return img
    
    def create_heatmap_overlay(self, image: Union[str, np.ndarray, Image.Image],
                             attention_map: np.ndarray,
                             alpha: float = 0.7) -> Image.Image:
        """
        Create a heatmap overlay on an image showing areas of interest.
        
        Args:
            image: Image file path, numpy array, or PIL Image
            attention_map: 2D numpy array of attention weights
            alpha: Opacity of heatmap overlay (0-1)
            
        Returns:
            PIL Image with heatmap overlay
        """
        # Load image if needed
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise ValueError("Unsupported image type")
        
        # Get image dimensions
        width, height = img.size
        
        # Resize attention map to match image dimensions
        attention_resized = cv2.resize(
            attention_map, 
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize the attention map
        if np.max(attention_resized) > 0:
            attention_normalized = attention_resized / np.max(attention_resized)
        else:
            attention_normalized = attention_resized
        
        # Create heatmap using jet colormap
        heatmap = cv2.applyColorMap(
            (255 * attention_normalized).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Convert back to PIL Image
        heatmap_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        
        # Blend images using alpha
        blended = Image.blend(img, heatmap_img, alpha)
        
        return blended
    
    def create_audio_spectrogram(self, spectrogram: np.ndarray,
                               is_deepfake: bool = False,
                               confidence: float = 0.0) -> plt.Figure:
        """
        Create a spectrogram visualization for audio.
        
        Args:
            spectrogram: Mel spectrogram as numpy array
            is_deepfake: Whether the audio is detected as deepfake
            confidence: Detection confidence
            
        Returns:
            Matplotlib figure with spectrogram visualization
        """
        # Create figure with theme styling
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set background color
        fig.patch.set_facecolor(self.colors["background"])
        ax.set_facecolor(self.colors["grid"])
        
        # Display spectrogram
        img = ax.imshow(
            spectrogram,
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        
        # Add a color bar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('dB', color=self.colors["text"])
        cbar.ax.yaxis.set_tick_params(color=self.colors["text"])
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=self.colors["text"])
        
        # Set labels
        ax.set_xlabel('Time', color=self.colors["text"])
        ax.set_ylabel('Mel Frequency', color=self.colors["text"])
        
        # Set tick colors
        ax.tick_params(axis='x', colors=self.colors["text"])
        ax.tick_params(axis='y', colors=self.colors["text"])
        
        # Add title with detection result
        result_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC"
        color = self.colors["deepfake"] if is_deepfake else self.colors["authentic"]
        
        title = f"Audio Spectrogram: {result_text} (Confidence: {confidence:.2f})"
        ax.set_title(title, color=color, fontweight='bold')
        
        # Add grid
        ax.grid(color=self.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def create_temporal_analysis(self, scores: List[float], 
                               times: List[float],
                               threshold: float = 0.5) -> plt.Figure:
        """
        Create a temporal analysis visualization showing detection scores over time.
        
        Args:
            scores: List of detection scores
            times: List of time points corresponding to scores
            threshold: Confidence threshold for classification
            
        Returns:
            Matplotlib figure with temporal analysis visualization
        """
        # Create figure with theme styling
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set background color
        fig.patch.set_facecolor(self.colors["background"])
        ax.set_facecolor(self.colors["grid"])
        
        # Create colors based on threshold
        colors = []
        for score in scores:
            if score >= threshold:
                colors.append(self.colors["deepfake"])
            else:
                colors.append(self.colors["authentic"])
        
        # Plot scores
        ax.scatter(times, scores, c=colors, s=50, alpha=0.8)
        ax.plot(times, scores, color=self.colors["primary"], alpha=0.6)
        
        # Add threshold line
        ax.axhline(y=threshold, color=self.colors["uncertain"], 
                 linestyle='--', alpha=0.7, label=f"Threshold ({threshold})")
        
        # Set labels
        ax.set_xlabel('Time (seconds)', color=self.colors["text"])
        ax.set_ylabel('Deepfake Confidence Score', color=self.colors["text"])
        
        # Set tick colors
        ax.tick_params(axis='x', colors=self.colors["text"])
        ax.tick_params(axis='y', colors=self.colors["text"])
        
        # Add title
        ax.set_title('Temporal Analysis', color=self.colors["text"], fontweight='bold')
        
        # Add grid
        ax.grid(color=self.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Set y-axis limits
        ax.set_ylim([0, 1])
        
        # Add legend
        ax.legend(facecolor=self.colors["background"], edgecolor=self.colors["grid"],
                labelcolor=self.colors["text"])
        
        plt.tight_layout()
        return fig
    
    def create_video_analysis_dashboard(self, detection_result: Dict[str, Any],
                                      include_frames: int = 5) -> Dict[str, Any]:
        """
        Create a comprehensive dashboard visualization for video analysis.
        
        Args:
            detection_result: Detection result dictionary for a video
            include_frames: Number of key frames to include in visualization
            
        Returns:
            Dictionary of visualization elements:
            - temporal_plot: Temporal analysis as base64 image
            - frame_overlays: List of key frames with detection overlays
            - summary_stats: Summary statistics and visualization
        """
        dashboard = {}
        
        # Extract required data
        is_deepfake = detection_result.get("is_deepfake", False)
        confidence = detection_result.get("confidence", 0.0)
        details = detection_result.get("details", {})
        
        # Create temporal analysis visualization
        if "frame_scores" in details:
            frame_scores = details["frame_scores"]
            frame_times = details.get("frame_times", list(range(len(frame_scores))))
            
            temporal_fig = self.create_temporal_analysis(
                frame_scores, 
                frame_times,
                detection_result.get("threshold", 0.5)
            )
            
            # Convert to base64
            buf = BytesIO()
            temporal_fig.savefig(buf, format='png', dpi=self.dpi, 
                               facecolor=self.colors["background"])
            buf.seek(0)
            temporal_plot = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(temporal_fig)
            
            dashboard["temporal_plot"] = temporal_plot
        
        # Process key frames
        frame_overlays = []
        
        # Select key frames (most confident deepfake frames and some authentic frames)
        if "frames" in details and "frame_scores" in details:
            frames = details["frames"]
            scores = details["frame_scores"]
            
            # Sort frames by score
            frame_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # Select top deepfake frames and some authentic frames
            selected_indices = []
            deepfake_count = 0
            authentic_count = 0
            
            for idx in frame_indices:
                if len(selected_indices) >= include_frames:
                    break
                    
                if scores[idx] >= detection_result.get("threshold", 0.5):
                    if deepfake_count < include_frames // 2 + 1:
                        selected_indices.append(idx)
                        deepfake_count += 1
                else:
                    if authentic_count < include_frames // 2:
                        selected_indices.append(idx)
                        authentic_count += 1
            
            # Sort by original order
            selected_indices.sort()
            
            # Create overlays for selected frames
            for idx in selected_indices:
                frame = frames[idx]
                
                # Create visualization with detection overlay
                frame_result = {
                    "is_deepfake": scores[idx] >= detection_result.get("threshold", 0.5),
                    "confidence": scores[idx],
                    "details": {}
                }
                
                # If face detection results are available
                if "face_detections" in details:
                    frame_result["details"]["face_results"] = details["face_detections"].get(str(idx), [])
                
                overlay = self.create_detection_overlay(frame, frame_result)
                
                # Convert to base64
                buf = BytesIO()
                overlay.save(buf, format='PNG')
                buf.seek(0)
                overlay_b64 = base64.b64encode(buf.read()).decode('utf-8')
                
                frame_overlays.append({
                    "frame_index": idx,
                    "time": details.get("frame_times", [0] * len(frames))[idx],
                    "score": scores[idx],
                    "overlay": overlay_b64
                })
        
        dashboard["frame_overlays"] = frame_overlays
        
        # Create summary statistics
        summary_stats = {
            "is_deepfake": is_deepfake,
            "confidence": confidence,
            "threshold": detection_result.get("threshold", 0.5),
            "temporal_consistency": details.get("temporal_inconsistency", 0.0),
            "av_sync_score": details.get("av_sync_score", 0.0),
            "total_frames_analyzed": len(details.get("frame_scores", [])),
            "deepfake_frame_percentage": (
                sum(1 for s in details.get("frame_scores", []) 
                    if s >= detection_result.get("threshold", 0.5)) / 
                max(1, len(details.get("frame_scores", [])))
            ) * 100
        }
        
        dashboard["summary_stats"] = summary_stats
        
        return dashboard
    
    def create_confidence_gauge_chart(self, confidence: float, 
                                    is_deepfake: bool) -> Dict[str, Any]:
        """
        Create a gauge chart visualization of detection confidence.
        
        Args:
            confidence: Detection confidence score (0-1)
            is_deepfake: Whether the media is detected as deepfake
            
        Returns:
            Plotly figure data as JSON
        """
        # Determine color based on confidence and result
        if is_deepfake:
            color = self.colors["deepfake"]
            title = "DEEPFAKE DETECTED"
        else:
            color = self.colors["authentic"]
            title = "AUTHENTIC"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,  # Convert to percentage
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'color': color, 'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': self.colors["text"]},
                'bar': {'color': color},
                'bgcolor': self.colors["grid"],
                'borderwidth': 2,
                'bordercolor': self.colors["text"],
                'steps': [
                    {'range': [0, 50], 'color': self.colors["authentic"]},
                    {'range': [50, 100], 'color': self.colors["deepfake"]}
                ],
                'threshold': {
                    'line': {'color': self.colors["uncertain"], 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            number={'suffix': '%', 'font': {'color': self.colors["text"], 'size': 20}}
        ))
        
        # Update layout with theme
        fig.update_layout(
            paper_bgcolor=self.colors["background"],
            font={'color': self.colors["text"], 'family': 'Arial'},
            margin=dict(l=20, r=20, t=60, b=20),
            height=300
        )
        
        return fig.to_dict()
    
    def save_image(self, image: Union[Image.Image, np.ndarray], 
                 filename: str, directory: str = None) -> str:
        """
        Save an image to file.
        
        Args:
            image: PIL Image or numpy array
            filename: Filename to save as
            directory: Directory to save in, defaults to self.output_dir
            
        Returns:
            Path to saved file
        """
        if directory is None:
            directory = self.output_dir
            
        os.makedirs(directory, exist_ok=True)
        
        # Ensure the filename has an extension
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
            
        filepath = os.path.join(directory, filename)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
        # Save the image
        image.save(filepath)
        self.logger.debug(f"Saved image to {filepath}")
        
        return filepath
    
    def _get_confidence_color(self, confidence: float) -> str:
        """
        Get a color based on confidence value.
        
        Args:
            confidence: Confidence value (0-1)
            
        Returns:
            Hex color code
        """
        if confidence >= 0.7:
            return self.colors["deepfake"]
        elif confidence <= 0.3:
            return self.colors["authentic"]
        else:
            return self.colors["uncertain"]
    
    def encode_image_base64(self, image: Union[Image.Image, np.ndarray, str]) -> str:
        """
        Encode an image as base64 string.
        
        Args:
            image: PIL Image, numpy array, or path to image
            
        Returns:
            Base64 encoded string
        """
        # Load image if it's a path
        if isinstance(image, str):
            img = Image.open(image)
        # Convert numpy array to PIL Image
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
        # Use PIL Image directly
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("Unsupported image type")
            
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')

# Helper functions
def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return base64_str

def generate_plot_colors(n: int, theme: str = "tron_legacy") -> List[str]:
    """
    Generate n distinct colors for plots based on theme.
    
    Args:
        n: Number of colors to generate
        theme: Theme name
        
    Returns:
        List of hex color codes
    """
    if theme == "tron_legacy":
        base_colors = [
            "#4CC9F0", "#4361EE", "#3A0CA3", "#7209B7", 
            "#F72585", "#B5179E", "#560BAD", "#480CA8"
        ]
    else:
        base_colors = [
            "#1976D2", "#388E3C", "#FBC02D", "#D32F2F",
            "#7B1FA2", "#00796B", "#FFA000", "#5D4037"
        ]
    
    # If we need more colors than base_colors, generate additional ones
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # Use HSV color space to generate evenly distributed colors
        colors = []
        for i in range(n):
            hue = i / n
            rgb = tuple(int(x * 255) for x in plt.cm.hsv(hue)[:3])
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
            colors.append(hex_color)
        return colors
