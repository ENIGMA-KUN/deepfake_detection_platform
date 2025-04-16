"""
Media upload components for the Deepfake Detection Platform.

This module provides specialized components for handling image, audio,
and video uploads with a Tron Legacy-inspired design.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import threading
from PIL import Image, ImageTk
import logging

# Import utilities from parent modules
from app.utils.file_handler import validate_file, save_uploaded_file
from app.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger('upload_component')

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

class MediaUploadComponent:
    """
    Base class for media upload components.
    
    This class provides common functionality for all media upload components,
    including file selection, validation, and preview generation.
    """
    
    def __init__(self, parent, media_type, on_files_selected=None, on_analyze=None):
        """
        Initialize the upload component.
        
        Args:
            parent: Parent widget
            media_type: Type of media ('image', 'audio', 'video')
            on_files_selected: Callback when files are selected
            on_analyze: Callback when analysis is requested
        """
        self.parent = parent
        self.media_type = media_type
        self.on_files_selected = on_files_selected
        self.on_analyze = on_analyze
        self.files = []
        
        # Create frame
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create components
        self.create_header()
        self.create_upload_section()
        self.create_file_list()
        self.create_action_buttons()
    
    def create_header(self):
        """Create the component header."""
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title = f"{self.media_type.capitalize()} Upload"
        title_label = ttk.Label(
            header_frame,
            text=title,
            style="Subheader.TLabel"
        )
        title_label.pack(anchor=tk.W)
        
        # Explanation text
        media_desc = {
            'image': "Upload image files to analyze for potential deepfakes.",
            'audio': "Upload audio files to analyze for voice deepfakes and manipulations.",
            'video': "Upload video files to analyze for deepfake content."
        }
        
        desc_label = ttk.Label(
            header_frame,
            text=media_desc.get(self.media_type, "Upload files for analysis."),
            wraplength=400
        )
        desc_label.pack(anchor=tk.W)
    
    def create_upload_section(self):
        """Create the upload drop area and buttons."""
        upload_frame = ttk.Frame(self.frame)
        upload_frame.pack(fill=tk.X, pady=10)
        
        # Upload area
        self.upload_area = tk.Canvas(
            upload_frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            height=120
        )
        self.upload_area.pack(fill=tk.X)
        
        # Upload text and icon
        self.upload_area.create_text(
            200, 40,
            text=f"Drop {self.media_type} files here",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 14)
        )
        
        self.upload_area.create_text(
            200, 70,
            text="or click to select files",
            fill=TronTheme.SKY_BLUE,
            font=("Helvetica", 10)
        )
        
        # Bind upload area events
        self.upload_area.bind("<Button-1>", self.select_files)
        self.upload_area.bind("<Enter>", self.on_upload_area_enter)
        self.upload_area.bind("<Leave>", self.on_upload_area_leave)
        
        # Drag and drop bindings
        self.upload_area.drop_target_register("files")
        self.upload_area.dnd_bind("<<Drop>>", self.on_drop)
        
        # Upload buttons
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        select_btn = ttk.Button(
            button_frame,
            text="Select Files",
            command=self.select_files
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_files
        )
        clear_btn.pack(side=tk.LEFT)
    
    def create_file_list(self):
        """Create the file list section."""
        # Container frame
        list_frame = ttk.Frame(self.frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Header
        list_label = ttk.Label(list_frame, text="Selected Files:")
        list_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Scrollable list
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_list = tk.Listbox(
            list_container,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            selectbackground=TronTheme.NEON_CYAN,
            selectforeground=TronTheme.DEEP_BLACK,
            yscrollcommand=scrollbar.set,
            borderwidth=1,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_list.yview)
        
        # Bind selection event
        self.file_list.bind("<<ListboxSelect>>", self.on_file_select)
    
    def create_action_buttons(self):
        """Create action buttons."""
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        analyze_btn = ttk.Button(
            button_frame,
            text=f"▶️ Analyze {self.media_type.capitalize()}",
            command=self.analyze_files
        )
        analyze_btn.pack(fill=tk.X)
    
    def select_files(self, event=None):
        """Open file dialog to select files."""
        filetypes = []
        
        if self.media_type == 'image':
            filetypes = [
                ('Image Files', '*.jpg *.jpeg *.png *.gif *.bmp'),
                ('All Files', '*.*')
            ]
        elif self.media_type == 'audio':
            filetypes = [
                ('Audio Files', '*.wav *.mp3 *.flac'),
                ('All Files', '*.*')
            ]
        elif self.media_type == 'video':
            filetypes = [
                ('Video Files', '*.mp4 *.avi *.mov'),
                ('All Files', '*.*')
            ]
        
        files = filedialog.askopenfilenames(
            title=f"Select {self.media_type.capitalize()} Files",
            filetypes=filetypes
        )
        
        if not files:
            return
        
        # Process selected files
        threading.Thread(
            target=self.process_selected_files,
            args=(files,),
            daemon=True
        ).start()
    
    def process_selected_files(self, file_paths):
        """
        Process and validate selected files.
        
        Args:
            file_paths: List of file paths
        """
        valid_files = []
        
        for file_path in file_paths:
            is_valid, message, detected_type = validate_file(file_path)
            
            if is_valid and detected_type == self.media_type:
                valid_files.append(file_path)
                
                # Update UI (must be done in main thread)
                self.parent.after(0, lambda p=file_path: self.add_file_to_list(p))
            else:
                # Show error (must be done in main thread)
                self.parent.after(0, lambda m=message, p=file_path: self.show_file_error(p, m))
        
        # Store valid files
        self.files.extend(valid_files)
        
        # Call callback if provided
        if self.on_files_selected and valid_files:
            self.parent.after(0, lambda: self.on_files_selected(valid_files))
    
    def add_file_to_list(self, file_path):
        """
        Add a file to the file list.
        
        Args:
            file_path: Path to the file
        """
        filename = os.path.basename(file_path)
        self.file_list.insert(tk.END, filename)
    
    def show_file_error(self, file_path, message):
        """
        Show file validation error.
        
        Args:
            file_path: Path to the file
            message: Error message
        """
        filename = os.path.basename(file_path)
        import tkinter.messagebox as messagebox
        messagebox.showwarning(
            "Invalid File",
            f"File {filename} is not valid: {message}"
        )
    
    def on_file_select(self, event):
        """
        Handle file selection in the list.
        
        Args:
            event: Selection event
        """
        # To be implemented by subclasses
        pass
    
    def on_upload_area_enter(self, event):
        """
        Handle mouse enter event for upload area.
        
        Args:
            event: Enter event
        """
        self.upload_area.config(highlightbackground=TronTheme.SKY_BLUE)
    
    def on_upload_area_leave(self, event):
        """
        Handle mouse leave event for upload area.
        
        Args:
            event: Leave event
        """
        self.upload_area.config(highlightbackground=TronTheme.NEON_CYAN)
    
    def on_drop(self, event):
        """
        Handle file drop event.
        
        Args:
            event: Drop event
        """
        file_paths = event.data
        
        if not file_paths:
            return
        
        # Process dropped files
        threading.Thread(
            target=self.process_selected_files,
            args=(file_paths,),
            daemon=True
        ).start()
    
    def clear_files(self):
        """Clear all selected files."""
        self.files = []
        self.file_list.delete(0, tk.END)
        self.clear_preview()
    
    def clear_preview(self):
        """Clear the preview area."""
        # To be implemented by subclasses
        pass
    
    def analyze_files(self):
        """Start analysis of selected files."""
        if not self.files:
            import tkinter.messagebox as messagebox
            messagebox.showinfo(
                "No Files",
                f"Please select {self.media_type} files to analyze"
            )
            return
        
        # Call the analyze callback if provided
        if self.on_analyze:
            self.on_analyze(self.files)

class ImageUploadComponent(MediaUploadComponent):
    """
    Component for image upload and preview.
    
    This component extends the base upload component with image-specific
    functionality, including image preview and thumbnail generation.
    """
    
    def __init__(self, parent, on_files_selected=None, on_analyze=None):
        """
        Initialize the image upload component.
        
        Args:
            parent: Parent widget
            on_files_selected: Callback when files are selected
            on_analyze: Callback when analysis is requested
        """
        super().__init__(parent, 'image', on_files_selected, on_analyze)
        
        # Add image-specific components
        self.create_preview_section()
        
        # Image cache for previews
        self.image_cache = {}
        self.current_photo = None
    
    def create_preview_section(self):
        """Create the image preview section."""
        preview_frame = ttk.Frame(self.frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(10, 0))
        
        # Title
        preview_label = ttk.Label(
            preview_frame,
            text="Image Preview",
            style="Subheader.TLabel"
        )
        preview_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            width=300,
            height=300
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.preview_canvas.create_text(
            150, 150,
            text="Select an image to preview",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )
    
    def on_file_select(self, event):
        """
        Handle file selection for image preview.
        
        Args:
            event: Selection event
        """
        selection = self.file_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.files):
            return
        
        # Get selected file path
        file_path = self.files[index]
        
        # Load and display image preview
        threading.Thread(
            target=self.load_image_preview,
            args=(file_path,),
            daemon=True
        ).start()
    
    def load_image_preview(self, file_path):
        """
        Load image preview in a background thread.
        
        Args:
            file_path: Path to the image file
        """
        try:
            # Check if we already have this image in cache
            if file_path in self.image_cache:
                # Use cached image
                photo = self.image_cache[file_path]
                self.parent.after(0, lambda: self.display_image(photo))
                return
            
            # Load image
            image = Image.open(file_path)
            
            # Get canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Default dimensions if canvas not yet drawn
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 300
                canvas_height = 300
            
            # Resize to fit canvas
            img_width, img_height = image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Store in cache
            self.image_cache[file_path] = photo
            
            # Display in main thread
            self.parent.after(0, lambda: self.display_image(photo))
            
        except Exception as e:
            logger.error(f"Error loading image preview: {e}")
            self.parent.after(0, lambda: self.show_preview_error(str(e)))
    
    def display_image(self, photo):
        """
        Display image in the preview canvas.
        
        Args:
            photo: PhotoImage to display
        """
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Store reference to prevent garbage collection
        self.current_photo = photo
        
        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Default dimensions if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 300
            canvas_height = 300
        
        # Display image centered
        self.preview_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )
    
    def show_preview_error(self, error_message):
        """
        Show error in preview area.
        
        Args:
            error_message: Error message to display
        """
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Show error message
        self.preview_canvas.create_text(
            150, 150,
            text=f"Error loading image:\n{error_message}",
            fill=TronTheme.MAGENTA,
            font=("Helvetica", 10),
            width=280,
            justify=tk.CENTER
        )
    
    def clear_preview(self):
        """Clear the image preview."""
        self.preview_canvas.delete("all")
        self.current_photo = None
        
        # Show default message
        self.preview_canvas.create_text(
            150, 150,
            text="Select an image to preview",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )

class AudioUploadComponent(MediaUploadComponent):
    """
    Component for audio upload and visualization.
    
    This component extends the base upload component with audio-specific
    functionality, including waveform visualization and playback controls.
    """
    
    def __init__(self, parent, on_files_selected=None, on_analyze=None):
        """
        Initialize the audio upload component.
        
        Args:
            parent: Parent widget
            on_files_selected: Callback when files are selected
            on_analyze: Callback when analysis is requested
        """
        super().__init__(parent, 'audio', on_files_selected, on_analyze)
        
        # Add audio-specific components
        self.create_visualization_section()
        self.create_playback_controls()
        
        # Audio playback state
        self.is_playing = False
        self.current_audio = None
    
    def create_visualization_section(self):
        """Create the audio visualization section."""
        viz_frame = ttk.Frame(self.frame)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Title
        viz_label = ttk.Label(
            viz_frame,
            text="Audio Visualization",
            style="Subheader.TLabel"
        )
        viz_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Visualization canvas
        self.viz_canvas = tk.Canvas(
            viz_frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            height=150
        )
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.viz_canvas.create_text(
            200, 75,
            text="Select an audio file to visualize",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )
    
    def create_playback_controls(self):
        """Create audio playback controls."""
        controls_frame = ttk.Frame(self.frame)
        controls_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Play button
        self.play_btn = ttk.Button(
            controls_frame,
            text="▶️ Play",
            command=self.toggle_playback
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Stop button
        stop_btn = ttk.Button(
            controls_frame,
            text="⏹️ Stop",
            command=self.stop_playback
        )
        stop_btn.pack(side=tk.LEFT)
        
        # File info
        self.audio_info = ttk.Label(
            controls_frame,
            text="No file selected",
            font=("Helvetica", 9)
        )
        self.audio_info.pack(side=tk.RIGHT)
    
    def on_file_select(self, event):
        """
        Handle file selection for audio visualization.
        
        Args:
            event: Selection event
        """
        selection = self.file_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.files):
            return
        
        # Get selected file path
        file_path = self.files[index]
        
        # Stop any current playback
        self.stop_playback()
        
        # Load and display audio visualization
        threading.Thread(
            target=self.load_audio_visualization,
            args=(file_path,),
            daemon=True
        ).start()
    
    def load_audio_visualization(self, file_path):
        """
        Load audio visualization in a background thread.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            # In a real implementation, this would use libraries like librosa
            # to analyze the audio file and create a waveform visualization
            
            # For now, we'll create a simulated waveform
            filename = os.path.basename(file_path)
            
            # Update UI in main thread
            self.parent.after(0, lambda: self.display_audio_visualization(file_path, filename))
            
        except Exception as e:
            logger.error(f"Error loading audio visualization: {e}")
            self.parent.after(0, lambda: self.show_visualization_error(str(e)))
    
    def display_audio_visualization(self, file_path, filename):
        """
        Display audio visualization in the canvas.
        
        Args:
            file_path: Path to the audio file
            filename: Filename for display
        """
        # Clear canvas
        self.viz_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.viz_canvas.winfo_width()
        canvas_height = self.viz_canvas.winfo_height()
        
        # Default dimensions if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 150
        
        # Draw simulated waveform
        import random
        import math
        
        # Use filename as seed for consistent visualization
        seed = sum(ord(c) for c in filename)
        random.seed(seed)
        
        # Draw waveform
        center_y = canvas_height // 2
        
        # Background lines
        for y in range(0, canvas_height, 10):
            self.viz_canvas.create_line(
                0, y,
                canvas_width, y,
                fill=TronTheme.DEEP_BLACK,
                width=1
            )
        
        # Waveform segments
        points = []
        for x in range(0, canvas_width, 2):
            # Generate amplitude using sine wave with noise
            freq1 = random.uniform(0.01, 0.05)
            freq2 = random.uniform(0.01, 0.05)
            
            amp1 = math.sin(x * freq1) * 30
            amp2 = math.sin(x * freq2 + 1) * 15
            
            amp = amp1 + amp2
            
            # Add point
            points.append(x)
            points.append(center_y + amp)
        
        # Draw line
        if points:
            self.viz_canvas.create_line(
                *points,
                fill=TronTheme.NEON_CYAN,
                width=2,
                smooth=True
            )
        
        # Update file info
        self.audio_info.config(text=filename)
        
        # Store current audio file
        self.current_audio = file_path
    
    def show_visualization_error(self, error_message):
        """
        Show error in visualization area.
        
        Args:
            error_message: Error message to display
        """
        # Clear canvas
        self.viz_canvas.delete("all")
        
        # Show error message
        self.viz_canvas.create_text(
            200, 75,
            text=f"Error loading audio:\n{error_message}",
            fill=TronTheme.MAGENTA,
            font=("Helvetica", 10),
            width=380,
            justify=tk.CENTER
        )
        
        # Update file info
        self.audio_info.config(text="Error loading file")
    
    def toggle_playback(self):
        """Toggle audio playback."""
        if not self.current_audio:
            return
        
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start audio playback."""
        # In a real implementation, this would use libraries like pygame.mixer
        # to play the audio file
        
        # For now, we'll just simulate playback
        self.is_playing = True
        self.play_btn.config(text="⏸️ Pause")
        
        # Show playback position indicator
        self.update_playback_indicator()
    
    def stop_playback(self):
        """Stop audio playback."""
        self.is_playing = False
        self.play_btn.config(text="▶️ Play")
        
        # Remove playback indicator
        self.viz_canvas.delete("playback_indicator")
    
    def update_playback_indicator(self):
        """Update playback position indicator."""
        if not self.is_playing:
            return
        
        # Remove previous indicator
        self.viz_canvas.delete("playback_indicator")
        
        # Calculate position (simulated progress)
        import time
        t = time.time() % 10  # Cycle every 10 seconds
        progress = t / 10
        
        canvas_width = self.viz_canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 400
        
        x_pos = int(progress * canvas_width)
        
        # Draw indicator
        self.viz_canvas.create_line(
            x_pos, 0,
            x_pos, self.viz_canvas.winfo_height(),
            fill=TronTheme.SUNRAY,
            width=2,
            tags=["playback_indicator"]
        )
        
        # Schedule next update
        if self.is_playing:
            self.parent.after(50, self.update_playback_indicator)
    
    def clear_preview(self):
        """Clear the audio visualization."""
        self.viz_canvas.delete("all")
        self.stop_playback()
        self.current_audio = None
        self.audio_info.config(text="No file selected")
        
        # Show default message
        self.viz_canvas.create_text(
            200, 75,
            text="Select an audio file to visualize",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )

class VideoUploadComponent(MediaUploadComponent):
    """
    Component for video upload and preview.
    
    This component extends the base upload component with video-specific
    functionality, including thumbnail preview and playback controls.
    """
    
    def __init__(self, parent, on_files_selected=None, on_analyze=None):
        """
        Initialize the video upload component.
        
        Args:
            parent: Parent widget
            on_files_selected: Callback when files are selected
            on_analyze: Callback when analysis is requested
        """
        super().__init__(parent, 'video', on_files_selected, on_analyze)
        
        # Add video-specific components
        self.create_preview_section()
        self.create_playback_controls()
        
        # Video playback state
        self.is_playing = False
        self.current_video = None
        self.thumbnail_cache = {}
    
    def create_preview_section(self):
        """Create the video preview section."""
        preview_frame = ttk.Frame(self.frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(10, 0))
        
        # Title
        preview_label = ttk.Label(
            preview_frame,
            text="Video Preview",
            style="Subheader.TLabel"
        )
        preview_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            width=320,
            height=240
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.preview_canvas.create_text(
            160, 120,
            text="Select a video to preview",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )
    
    def create_playback_controls(self):
        """Create video playback controls."""
        controls_frame = ttk.Frame(self.frame)
        controls_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Play button
        self.play_btn = ttk.Button(
            controls_frame,
            text="▶️ Play",
            command=self.toggle_playback
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Stop button
        stop_btn = ttk.Button(
            controls_frame,
            text="⏹️ Stop",
            command=self.stop_playback
        )
        stop_btn.pack(side=tk.LEFT)
        
        # Video info
        self.video_info = ttk.Label(
            controls_frame,
            text="No file selected",
            font=("Helvetica", 9)
        )
        self.video_info.pack(side=tk.RIGHT)
    
    def on_file_select(self, event):
        """
        Handle file selection for video preview.
        
        Args:
            event: Selection event
        """
        selection = self.file_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.files):
            return
        
        # Get selected file path
        file_path = self.files[index]
        
        # Stop any current playback
        self.stop_playback()
        
        # Load and display video thumbnail
        threading.Thread(
            target=self.load_video_thumbnail,
            args=(file_path,),
            daemon=True
        ).start()
    
    def load_video_thumbnail(self, file_path):
        """
        Load video thumbnail in a background thread.
        
        Args:
            file_path: Path to the video file
        """
        try:
            # Check if we already have this thumbnail in cache
            if file_path in self.thumbnail_cache:
                # Use cached thumbnail
                photo = self.thumbnail_cache[file_path]
                self.parent.after(0, lambda: self.display_video_thumbnail(file_path, photo))
                return
            
            # In a real implementation, this would use libraries like cv2 or moviepy
            # to extract a frame from the video for thumbnail
            
            # For now, we'll create a simulated thumbnail
            # Create a blank image with Tron theme colors
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image
            img = Image.new('RGB', (320, 240), TronTheme.DEEP_BLACK)
            draw = ImageDraw.Draw(img)
            
            # Draw grid lines
            for x in range(0, 320, 20):
                draw.line([(x, 0), (x, 240)], fill=TronTheme.DARK_TEAL, width=1)
            
            for y in range(0, 240, 20):
                draw.line([(0, y), (320, y)], fill=TronTheme.DARK_TEAL, width=1)
            
            # Draw border
            draw.rectangle([(0, 0), (319, 239)], outline=TronTheme.NEON_CYAN, width=2)
            
            # Draw text
            filename = os.path.basename(file_path)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                
            draw.text((160, 100), "Video Preview", fill=TronTheme.NEON_CYAN, font=font, anchor="mm")
            draw.text((160, 130), filename, fill=TronTheme.SKY_BLUE, font=font, anchor="mm")
            
            # Draw play button
            draw.ellipse([(130, 150), (190, 210)], outline=TronTheme.NEON_CYAN, width=2)
            draw.polygon([(150, 165), (150, 195), (180, 180)], fill=TronTheme.NEON_CYAN)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Store in cache
            self.thumbnail_cache[file_path] = photo
            
            # Display in main thread
            self.parent.after(0, lambda: self.display_video_thumbnail(file_path, photo))
            
        except Exception as e:
            logger.error(f"Error loading video thumbnail: {e}")
            self.parent.after(0, lambda: self.show_preview_error(str(e)))
    
    def display_video_thumbnail(self, file_path, photo):
        """
        Display video thumbnail in the preview canvas.
        
        Args:
            file_path: Path to the video file
            photo: PhotoImage thumbnail
        """
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Store reference to prevent garbage collection
        self.current_thumbnail = photo
        
        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Default dimensions if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 320
            canvas_height = 240
        
        # Display image centered
        self.preview_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )
        
        # Update video info
        filename = os.path.basename(file_path)
        self.video_info.config(text=filename)
        
        # Store current video file
        self.current_video = file_path
    
    def show_preview_error(self, error_message):
        """
        Show error in preview area.
        
        Args:
            error_message: Error message to display
        """
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Show error message
        self.preview_canvas.create_text(
            160, 120,
            text=f"Error loading video:\n{error_message}",
            fill=TronTheme.MAGENTA,
            font=("Helvetica", 10),
            width=280,
            justify=tk.CENTER
        )
        
        # Update video info
        self.video_info.config(text="Error loading file")
    
    def toggle_playback(self):
        """Toggle video playback."""
        if not self.current_video:
            return
        
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback simulation."""
        # In a real implementation, this would use libraries like cv2 or moviepy
        # to play the video
        
        # For now, we'll just simulate playback with a simple animation
        self.is_playing = True
        self.play_btn.config(text="⏸️ Pause")
        
        # Start frame animation
        self.frame_index = 0
        self.animate_playback()
    
    def stop_playback(self):
        """Stop video playback."""
        self.is_playing = False
        self.play_btn.config(text="▶️ Play")
        
        # If we have a thumbnail, restore it
        if self.current_video and self.current_video in self.thumbnail_cache:
            photo = self.thumbnail_cache[self.current_video]
            self.display_video_thumbnail(self.current_video, photo)
    
    def animate_playback(self):
        """Animate video playback (simulated)."""
        if not self.is_playing:
            return
        
        # Create simulated frame
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image
        img = Image.new('RGB', (320, 240), TronTheme.DEEP_BLACK)
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines
        for x in range(0, 320, 20):
            draw.line([(x, 0), (x, 240)], fill=TronTheme.DARK_TEAL, width=1)
        
        for y in range(0, 240, 20):
            draw.line([(0, y), (320, y)], fill=TronTheme.DARK_TEAL, width=1)
        
        # Draw border
        draw.rectangle([(0, 0), (319, 239)], outline=TronTheme.NEON_CYAN, width=2)
        
        # Draw animated element (based on frame index)
        import math
        
        # Moving circle
        x = 160 + int(math.sin(self.frame_index / 10) * 100)
        y = 120 + int(math.cos(self.frame_index / 15) * 60)
        
        draw.ellipse([(x-20, y-20), (x+20, y+20)], outline=TronTheme.NEON_CYAN, width=2)
        
        # Draw text
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        draw.text((160, 20), "Video Playback", fill=TronTheme.NEON_CYAN, font=font, anchor="mm")
        draw.text((160, 220), f"Frame: {self.frame_index}", fill=TronTheme.SKY_BLUE, font=font, anchor="mm")
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Store reference to prevent garbage collection
        self.current_thumbnail = photo
        
        # Display image
        self.preview_canvas.create_image(
            160, 120,
            image=photo,
            anchor=tk.CENTER
        )
        
        # Increment frame index
        self.frame_index += 1
        
        # Schedule next frame
        if self.is_playing:
            self.parent.after(50, self.animate_playback)
    
    def clear_preview(self):
        """Clear the video preview."""
        self.preview_canvas.delete("all")
        self.stop_playback()
        self.current_video = None
        self.video_info.config(text="No file selected")
        
        # Show default message
        self.preview_canvas.create_text(
            160, 120,
            text="Select a video to preview",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )