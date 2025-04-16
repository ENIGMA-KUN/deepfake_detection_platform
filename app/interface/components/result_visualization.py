"""
Result visualization components for the Deepfake Detection Platform.

This module provides specialized components for visualizing detection results
with a Tron Legacy-inspired design, including confidence meters, heatmaps,
and region highlighting.
"""

import tkinter as tk
from tkinter import ttk
import colorsys
from PIL import Image, ImageTk, ImageDraw
import logging
import os
import math
import numpy as np

# Import utilities from parent modules
from app.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger('result_visualization')

class TronTheme:
    """Color constants for Tron theme."""
    DEEP_BLACK = "#000000"
    DARK_TEAL = "#57A3B7"
    NEON_CYAN = "#00FFFF"
    SKY_BLUE = "#BFD4FF"
    POWDER_BLUE = "#BBFEEF"
    MAX_BLUE = "#48B3B6"
    ELECTRIC_PURPLE = "#3B1AAA"
    SUNRAY = "#DE945B"
    MAGENTA = "#B20D58"
    
    @staticmethod
    def get_gradient_color(value, start_color, end_color):
        """
        Get a color interpolated between start_color and end_color.
        
        Args:
            value: Value between 0 and 1
            start_color: Starting color in hex format
            end_color: Ending color in hex format
            
        Returns:
            Interpolated color in hex format
        """
        # Convert hex to RGB
        r1, g1, b1 = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
        r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
        
        # Interpolate
        r = int(r1 + (r2 - r1) * value)
        g = int(g1 + (g2 - g1) * value)
        b = int(b1 + (b2 - b1) * value)
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def get_confidence_color(confidence):
        """
        Get color based on confidence score.
        
        Args:
            confidence: Confidence value between 0 and 1
            
        Returns:
            Color in hex format
        """
        if confidence < 0.5:
            # Red to yellow gradient
            return TronTheme.get_gradient_color(confidence * 2, TronTheme.MAGENTA, TronTheme.SUNRAY)
        else:
            # Yellow to cyan gradient
            return TronTheme.get_gradient_color((confidence - 0.5) * 2, TronTheme.SUNRAY, TronTheme.NEON_CYAN)

class ConfidenceMeter:
    """
    Component for displaying a confidence score meter.
    
    This component visualizes confidence scores with a glowing Tron-themed
    meter and numerical display.
    """
    
    def __init__(self, parent, value=0.0):
        """
        Initialize the confidence meter.
        
        Args:
            parent: Parent widget
            value: Initial confidence value (0.0 to 1.0)
        """
        self.parent = parent
        self.value = value
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas for meter
        self.canvas = tk.Canvas(
            self.frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.DARK_TEAL,
            highlightthickness=1,
            height=30
        )
        self.canvas.pack(fill=tk.X)
        
        # Create label for value
        self.label = ttk.Label(
            self.frame,
            text=f"{int(self.value * 100)}%",
            anchor=tk.CENTER,
            background=TronTheme.DEEP_BLACK,
            foreground="white"
        )
        self.label.pack(fill=tk.X, pady=(5, 0))
        
        # Draw meter
        self.draw_meter()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def draw_meter(self):
        """Draw the confidence meter."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Get dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Default dimensions if not drawn yet
        if width <= 1:
            width = 200
        
        # Draw background
        self.canvas.create_rectangle(
            0, 0,
            width, height,
            fill=TronTheme.DEEP_BLACK,
            outline=TronTheme.DARK_TEAL,
            width=1
        )
        
        # Calculate filled width
        filled_width = int(width * self.value)
        
        # Get color based on confidence
        color = TronTheme.get_confidence_color(self.value)
        
        # Draw filled portion with gradient
        if filled_width > 0:
            # Create gradient fill
            for x in range(0, filled_width):
                # Calculate gradient position
                pos = x / width
                
                # Get color based on position
                if pos < 0.5:
                    line_color = TronTheme.get_gradient_color(pos * 2, TronTheme.MAGENTA, TronTheme.SUNRAY)
                else:
                    line_color = TronTheme.get_gradient_color((pos - 0.5) * 2, TronTheme.SUNRAY, TronTheme.NEON_CYAN)
                
                self.canvas.create_line(
                    x, 0,
                    x, height,
                    fill=line_color,
                    width=1
                )
            
            # Add glow effect
            self.canvas.create_line(
                0, height // 2,
                filled_width, height // 2,
                fill=color,
                width=5,
                stipple="gray50"  # Stipple pattern for glow effect
            )
        
        # Add tick marks
        for i in range(0, 101, 10):
            x = int(width * i / 100)
            
            self.canvas.create_line(
                x, height - 5,
                x, height,
                fill=TronTheme.SKY_BLUE,
                width=1
            )
            
            if i % 20 == 0:
                self.canvas.create_text(
                    x, height - 10,
                    text=str(i),
                    fill=TronTheme.SKY_BLUE,
                    font=("Helvetica", 7),
                    anchor=tk.S
                )
        
        # Add value label
        self.label.config(text=f"Confidence: {int(self.value * 100)}%")
    
    def set_value(self, value):
        """
        Set the confidence value.
        
        Args:
            value: Confidence value (0.0 to 1.0)
        """
        self.value = max(0.0, min(1.0, value))
        self.draw_meter()
    
    def get_value(self):
        """
        Get the current confidence value.
        
        Returns:
            Current confidence value
        """
        return self.value

class HeatmapVisualization:
    """
    Component for displaying detection heatmaps.
    
    This component provides visualization of detection results as 
    heatmaps overlaid on the original media.
    """
    
    def __init__(self, parent):
        """
        Initialize the heatmap visualization.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas for visualization
        self.canvas = tk.Canvas(
            self.frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Current visualization data
        self.current_image = None
        self.current_heatmap = None
        self.original_size = (0, 0)
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def display_image_heatmap(self, image_path, regions=None, alpha=0.7):
        """
        Display image with heatmap overlay for detected regions.
        
        Args:
            image_path: Path to image file
            regions: List of region dictionaries (x, y, width, height, confidence, type)
            alpha: Opacity of heatmap overlay (0.0 to 1.0)
        """
        try:
            # Load image
            image = Image.open(image_path)
            self.original_size = image.size
            
            # Create heatmap overlay
            heatmap = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(heatmap)
            
            # Draw regions if provided
            if regions:
                for region in regions:
                    # Get region coordinates
                    x = int(region['x'] * image.size[0])
                    y = int(region['y'] * image.size[1])
                    width = int(region['width'] * image.size[0])
                    height = int(region['height'] * image.size[1])
                    
                    # Get color based on confidence
                    confidence = region['confidence']
                    base_color = TronTheme.get_confidence_color(confidence)
                    
                    # Convert hex to RGBA with alpha
                    r = int(base_color[1:3], 16)
                    g = int(base_color[3:5], 16)
                    b = int(base_color[5:7], 16)
                    a = int(alpha * 255)
                    
                    # Draw filled rectangle with alpha
                    draw.rectangle(
                        [x, y, x + width, y + height],
                        fill=(r, g, b, a)
                    )
                    
                    # Draw border
                    draw.rectangle(
                        [x, y, x + width, y + height],
                        outline=(r, g, b, 255),
                        width=2
                    )
            
            # Resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Default dimensions if not drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 300
            
            # Calculate resize ratio
            ratio = min(canvas_width / image.size[0], canvas_height / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            
            # Resize image and heatmap
            image = image.resize(new_size, Image.LANCZOS)
            heatmap = heatmap.resize(new_size, Image.LANCZOS)
            
            # Composite image with heatmap
            result = Image.alpha_composite(image.convert('RGBA'), heatmap)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(result)
            
            # Store references
            self.current_image = photo
            self.current_heatmap = heatmap
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Display image
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            
            # Draw legend
            self.draw_legend(regions)
            
        except Exception as e:
            logger.error(f"Error displaying image heatmap: {e}")
            self.show_error(str(e))
    
    def display_audio_heatmap(self, audio_path, regions=None):
        """
        Display audio waveform with heatmap overlay for detected regions.
        
        Args:
            audio_path: Path to audio file
            regions: List of region dictionaries (start_time, end_time, confidence, type)
        """
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Default dimensions if not drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 200
            
            # Draw background grid
            for x in range(0, canvas_width, 20):
                self.canvas.create_line(
                    x, 0,
                    x, canvas_height,
                    fill=TronTheme.DEEP_BLACK,
                    width=1
                )
            
            for y in range(0, canvas_height, 20):
                self.canvas.create_line(
                    0, y,
                    canvas_width, y,
                    fill=TronTheme.DEEP_BLACK,
                    width=1
                )
            
            # Draw time axis
            axis_y = canvas_height - 30
            self.canvas.create_line(
                50, axis_y,
                canvas_width - 50, axis_y,
                fill=TronTheme.NEON_CYAN,
                width=2
            )
            
            # Draw time markers
            max_time = 60  # Default: 1 minute (or from regions)
            if regions:
                max_time = max([region['end_time'] for region in regions]) + 5
            
            time_scale = (canvas_width - 100) / max_time
            
            for t in range(0, int(max_time) + 1, 5):
                x = 50 + t * time_scale
                
                self.canvas.create_line(
                    x, axis_y - 5,
                    x, axis_y + 5,
                    fill=TronTheme.NEON_CYAN,
                    width=1
                )
                
                self.canvas.create_text(
                    x, axis_y + 15,
                    text=f"{t}s",
                    fill=TronTheme.NEON_CYAN,
                    font=("Helvetica", 8),
                    anchor=tk.CENTER
                )
            
            # Draw simulated waveform
            waveform_height = axis_y - 50
            center_y = waveform_height // 2 + 25
            
            # Generate waveform from audio file name (deterministic)
            filename = os.path.basename(audio_path)
            seed = sum(ord(c) for c in filename)
            
            points = []
            for x in range(50, canvas_width - 50, 2):
                # Time position
                t = (x - 50) / time_scale
                
                # Generate amplitude using sine waves
                amp = 0
                for i in range(1, 5):
                    freq = (i * 0.1 + (seed % 10) * 0.01) * (1 + i * 0.1)
                    amp += math.sin(t * freq) * (20 / i)
                
                # Add noise
                amp += (hash(f"{seed}_{x}") % 1000) / 1000 * 10 - 5
                
                # Add point
                points.append(x)
                points.append(center_y + amp)
            
            # Draw waveform
            self.canvas.create_line(
                *points,
                fill=TronTheme.SKY_BLUE,
                width=1,
                smooth=True
            )
            
            # Draw regions if provided
            if regions:
                for region in regions:
                    # Get region coordinates
                    start_x = 50 + region['start_time'] * time_scale
                    end_x = 50 + region['end_time'] * time_scale
                    
                    # Get color based on confidence
                    confidence = region['confidence']
                    color = TronTheme.get_confidence_color(confidence)
                    
                    # Draw region
                    self.canvas.create_rectangle(
                        start_x, 25,
                        end_x, waveform_height + 25,
                        fill="",
                        outline=color,
                        width=2
                    )
                    
                    # Add stipple overlay
                    self.canvas.create_rectangle(
                        start_x, 25,
                        end_x, waveform_height + 25,
                        fill=color,
                        stipple="gray25",
                        outline=""
                    )
                    
                    # Add label if enough space
                    if end_x - start_x > 40:
                        self.canvas.create_text(
                            (start_x + end_x) / 2, 15,
                            text=region['type'],
                            fill=color,
                            font=("Helvetica", 8),
                            anchor=tk.CENTER
                        )
            
            # Draw legend
            self.draw_legend(regions)
            
        except Exception as e:
            logger.error(f"Error displaying audio heatmap: {e}")
            self.show_error(str(e))
    
    def display_video_heatmap(self, video_path, regions=None):
        """
        Display video keyframes with heatmap overlay for detected regions.
        
        Args:
            video_path: Path to video file
            regions: List of region dictionaries (start_frame, end_frame, confidence, type)
        """
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Default dimensions if not drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 300
            
            # Draw background
            self.canvas.create_rectangle(
                0, 0,
                canvas_width, canvas_height,
                fill=TronTheme.DEEP_BLACK,
                outline=TronTheme.NEON_CYAN,
                width=1
            )
            
            # In a real implementation, we would extract keyframes from the video
            # and display them with heatmap overlays
            
            # For now, draw a simulated timeline with frame markers
            timeline_y = canvas_height - 50
            self.canvas.create_line(
                50, timeline_y,
                canvas_width - 50, timeline_y,
                fill=TronTheme.NEON_CYAN,
                width=2
            )
            
            # Determine max frame from regions or use default
            max_frame = 300  # Default
            if regions:
                max_frame = max([region['end_frame'] for region in regions])
            
            frame_scale = (canvas_width - 100) / max_frame
            
            # Draw frame markers
            for frame in range(0, max_frame + 1, 60):
                x = 50 + frame * frame_scale
                
                self.canvas.create_line(
                    x, timeline_y - 5,
                    x, timeline_y + 5,
                    fill=TronTheme.NEON_CYAN,
                    width=1
                )
                
                self.canvas.create_text(
                    x, timeline_y + 15,
                    text=f"Frame {frame}",
                    fill=TronTheme.NEON_CYAN,
                    font=("Helvetica", 8),
                    anchor=tk.CENTER
                )
            
            # Create a simulated keyframe mosaic
            if regions:
                # Create keyframe boxes based on regions
                box_width = 80
                box_height = 60
                box_y = 50
                
                for i, region in enumerate(regions):
                    # Limit to 4 keyframes
                    if i >= 4:
                        break
                    
                    # Calculate position
                    box_x = 50 + i * (box_width + 20)
                    
                    # Get color based on confidence
                    color = TronTheme.get_confidence_color(region['confidence'])
                    
                    # Draw box
                    self.canvas.create_rectangle(
                        box_x, box_y,
                        box_x + box_width, box_y + box_height,
                        fill=TronTheme.DEEP_BLACK,
                        outline=color,
                        width=2
                    )
                    
                    # Add frame label
                    self.canvas.create_text(
                        box_x + box_width // 2, box_y + box_height + 15,
                        text=f"Frame {region['start_frame']}",
                        fill=color,
                        font=("Helvetica", 8),
                        anchor=tk.CENTER
                    )
                    
                    # Add type label
                    self.canvas.create_text(
                        box_x + box_width // 2, box_y - 10,
                        text=region['type'],
                        fill=color,
                        font=("Helvetica", 8),
                        anchor=tk.CENTER
                    )
                    
                    # Draw fake content in box
                    self.canvas.create_line(
                        box_x + 10, box_y + 10,
                        box_x + box_width - 10, box_y + box_height - 10,
                        fill=color,
                        width=1
                    )
                    self.canvas.create_line(
                        box_x + 10, box_y + box_height - 10,
                        box_x + box_width - 10, box_y + 10,
                        fill=color,
                        width=1
                    )
                    self.canvas.create_oval(
                        box_x + 20, box_y + 15,
                        box_x + box_width - 20, box_y + box_height - 15,
                        outline=color,
                        width=1
                    )
            
            # Draw regions on timeline if provided
            if regions:
                for region in regions:
                    # Get region coordinates
                    start_x = 50 + region['start_frame'] * frame_scale
                    end_x = 50 + region['end_frame'] * frame_scale
                    
                    # Get color based on confidence
                    confidence = region['confidence']
                    color = TronTheme.get_confidence_color(confidence)
                    
                    # Draw region on timeline
                    self.canvas.create_rectangle(
                        start_x, timeline_y - 10,
                        end_x, timeline_y + 10,
                        fill=color,
                        outline="white",
                        width=1
                    )
            
            # Show info text
            if not regions:
                self.canvas.create_text(
                    canvas_width // 2, 100,
                    text="No suspicious regions detected in video",
                    fill=TronTheme.NEON_CYAN,
                    font=("Helvetica", 12),
                    anchor=tk.CENTER
                )
            
            # Draw legend
            self.draw_legend(regions)
            
        except Exception as e:
            logger.error(f"Error displaying video heatmap: {e}")
            self.show_error(str(e))
    
    def draw_legend(self, regions):
        """
        Draw a legend for the heatmap visualization.
        
        Args:
            regions: List of region dictionaries
        """
        if not regions:
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Default dimensions if not drawn yet
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # Draw legend background
        legend_width = 150
        legend_height = 20 * len(regions) + 30
        legend_x = canvas_width - legend_width - 10
        legend_y = 10
        
        self.canvas.create_rectangle(
            legend_x, legend_y,
            legend_x + legend_width, legend_y + legend_height,
            fill=TronTheme.DEEP_BLACK,
            outline=TronTheme.NEON_CYAN,
            width=1,
            stipple="gray12"  # Stipple pattern for semi-transparency
        )
        
        # Draw legend title
        self.canvas.create_text(
            legend_x + 5, legend_y + 15,
            text="Detection Regions",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10, "bold"),
            anchor=tk.W
        )
        
        # Draw legend items
        for i, region in enumerate(regions):
            # Get color based on confidence
            confidence = region['confidence']
            color = TronTheme.get_confidence_color(confidence)
            
            # Draw color swatch
            self.canvas.create_rectangle(
                legend_x + 10, legend_y + 35 + i * 20,
                legend_x + 25, legend_y + 50 + i * 20,
                fill=color,
                outline="white",
                width=1
            )
            
            # Draw type label
            self.canvas.create_text(
                legend_x + 30, legend_y + 42 + i * 20,
                text=f"{region['type']} ({int(confidence * 100)}%)",
                fill="white",
                font=("Helvetica", 8),
                anchor=tk.W
            )
    
    def show_error(self, error_message):
        """
        Show error message in the canvas.
        
        Args:
            error_message: Error message to display
        """
        # Clear canvas
        self.canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Default dimensions if not drawn yet
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # Show error message
        self.canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=f"Error displaying visualization:\n{error_message}",
            fill=TronTheme.MAGENTA,
            font=("Helvetica", 10),
            width=canvas_width - 40,
            justify=tk.CENTER
        )
    
    def clear(self):
        """Clear the visualization."""
        self.canvas.delete("all")
        self.current_image = None
        self.current_heatmap = None

class ResultSummaryPanel:
    """
    Component for displaying detection result summaries.
    
    This component provides a summary panel with key metrics and confidence
    score visualization.
    """
    
    def __init__(self, parent):
        """
        Initialize the result summary panel.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        
        # Create frame with Tron styling
        self.frame = ttk.Frame(parent)
        
        # Create header
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(
            header_frame,
            text="Detection Results",
            style="Subheader.TLabel"
        )
        self.title_label.pack(anchor=tk.W)
        
        # Create content area
        content_frame = ttk.Frame(self.frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status panel
        status_frame = ttk.Frame(content_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="No results available")
        
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Helvetica", 12)
        )
        status_label.pack(anchor=tk.W)
        
        # Confidence meter
        self.confidence_meter = ConfidenceMeter(content_frame)
        self.confidence_meter.pack(fill=tk.X, pady=(0, 10))
        
        # Statistics panel
        stats_frame = ttk.Frame(content_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(
            stats_frame,
            height=6,
            width=40,
            bg=TronTheme.DEEP_BLACK,
            fg="white",
            bd=1,
            highlightbackground=TronTheme.DARK_TEAL,
            highlightthickness=1,
            font=("Helvetica", 9)
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.stats_text.tag_configure("header", foreground=TronTheme.NEON_CYAN, font=("Helvetica", 10, "bold"))
        self.stats_text.tag_configure("highlight", foreground=TronTheme.SKY_BLUE)
        self.stats_text.tag_configure("warning", foreground=TronTheme.SUNRAY)
        self.stats_text.tag_configure("danger", foreground=TronTheme.MAGENTA)
        
        # Action buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        detail_btn = ttk.Button(
            button_frame,
            text="View Details",
            command=self.show_details
        )
        detail_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        report_btn = ttk.Button(
            button_frame,
            text="Generate Report",
            command=self.generate_report
        )
        report_btn.pack(side=tk.LEFT)
        
        # Callback for showing details
        self.on_show_details = None
        
        # Callback for generating report
        self.on_generate_report = None
        
        # Initialize with empty state
        self.update_stats_text()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def update_summary(self, result):
        """
        Update the summary panel with detection results.
        
        Args:
            result: Detection result dictionary
        """
        if not result:
            return
        
        # Update title with filename
        if 'filename' in result:
            self.title_label.config(text=f"Results: {result['filename']}")
        
        # Update status based on detection
        if result.get('is_fake', False):
            self.status_var.set("⚠️ LIKELY MANIPULATED CONTENT")
            self.confidence_meter.set_value(result.get('confidence', 0.5))
        else:
            self.status_var.set("✓ LIKELY AUTHENTIC CONTENT")
            self.confidence_meter.set_value(result.get('confidence', 0.5))
        
        # Update statistics text
        self.update_stats_text(result)
    
    def update_stats_text(self, result=None):
        """
        Update the statistics text with result information.
        
        Args:
            result: Detection result dictionary (optional)
        """
        # Enable editing
        self.stats_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.stats_text.delete(1.0, tk.END)
        
        if result:
            # Insert header
            self.stats_text.insert(tk.END, "Analysis Summary\n", "header")
            
            # Insert classification
            classification = "AUTHENTIC" if not result.get('is_fake', False) else "MANIPULATED/FAKE"
            classification_tag = "highlight" if not result.get('is_fake', False) else "danger"
            self.stats_text.insert(tk.END, f"Classification: ", "highlight")
            self.stats_text.insert(tk.END, f"{classification}\n", classification_tag)
            
            # Insert confidence
            confidence = result.get('confidence', 0.0) * 100
            confidence_tag = "highlight" if confidence > 70 else "warning"
            self.stats_text.insert(tk.END, f"Confidence: ", "highlight")
            self.stats_text.insert(tk.END, f"{confidence:.1f}%\n", confidence_tag)
            
            # Insert media info
            media_type = result.get('media_type', 'unknown')
            self.stats_text.insert(tk.END, f"Media Type: ", "highlight")
            self.stats_text.insert(tk.END, f"{media_type.capitalize()}\n", "")
            
            # Insert detected regions
            regions = result.get('regions', [])
            if regions:
                self.stats_text.insert(tk.END, f"Detected Regions: ", "highlight")
                self.stats_text.insert(tk.END, f"{len(regions)}\n", "")
                
                # List region types
                region_types = set(region['type'] for region in regions)
                self.stats_text.insert(tk.END, f"Types: ", "highlight")
                self.stats_text.insert(tk.END, f"{', '.join(region_types)}\n", "")
            else:
                self.stats_text.insert(tk.END, f"Detected Regions: ", "highlight")
                self.stats_text.insert(tk.END, "None\n", "")
                
        else:
            # Show placeholder text
            self.stats_text.insert(tk.END, "No analysis results available.\n\n")
            self.stats_text.insert(tk.END, "Run detection on media files to view results.")
        
        # Disable editing
        self.stats_text.config(state=tk.DISABLED)
    
    def show_details(self):
        """Show detailed results."""
        if self.on_show_details:
            self.on_show_details()
    
    def generate_report(self):
        """Generate detection report."""
        if self.on_generate_report:
            self.on_generate_report()
    
    def set_callbacks(self, on_show_details=None, on_generate_report=None):
        """
        Set callbacks for button actions.
        
        Args:
            on_show_details: Callback for showing details
            on_generate_report: Callback for generating report
        """
        self.on_show_details = on_show_details
        self.on_generate_report = on_generate_report