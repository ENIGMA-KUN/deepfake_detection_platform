"""
Main GUI implementation for the Deepfake Detection Platform.

This module defines the main application interface with Tron Legacy-inspired
design and separate tabs for image, audio, and video upload and analysis.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from pathlib import Path
import yaml
import logging

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from app.core.processor import MediaProcessor
from app.core.queue_manager import QueueManager
from app.core.result_handler import ResultHandler
from app.utils.file_handler import validate_file, save_uploaded_file, get_file_info
from app.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger('interface')

class TronTheme:
    """
    Theme manager for Tron Legacy-inspired styling.
    
    This class provides color definitions and styling methods for
    applying the Tron Legacy theme to tkinter widgets.
    """
    
    # Color definitions
    DEEP_BLACK = "#000000"
    DARK_TEAL = "#57A3B7"
    NEON_CYAN = "#00FFFF"
    SKY_BLUE = "#BFD4FF"
    POWDER_BLUE = "#BBFEEF"
    MAX_BLUE = "#48B3B6"
    ELECTRIC_PURPLE = "#3B1AAA"
    SUNRAY = "#DE945B"
    MAGENTA = "#B20D58"
    
    @classmethod
    def setup_theme(cls):
        """Configure ttk styles for the application theme."""
        style = ttk.Style()
        
        # Configure basic styles
        style.configure("TFrame", background=cls.DEEP_BLACK)
        style.configure("TLabel", 
                       background=cls.DEEP_BLACK, 
                       foreground="white", 
                       font=("Helvetica", 10))
        style.configure("TButton", 
                       background=cls.DEEP_BLACK, 
                       foreground=cls.NEON_CYAN, 
                       borderwidth=1, 
                       font=("Helvetica", 10, "bold"),
                       focuscolor=cls.NEON_CYAN)
        
        # Header style
        style.configure("Header.TLabel", 
                       foreground=cls.NEON_CYAN, 
                       font=("Helvetica", 16, "bold"))
        
        # Subheader style
        style.configure("Subheader.TLabel", 
                       foreground=cls.SKY_BLUE, 
                       font=("Helvetica", 12, "bold"))
        
        # Button styles
        style.map("TButton",
                 background=[("active", cls.NEON_CYAN)],
                 foreground=[("active", cls.DEEP_BLACK)])
        
        # Tab styles
        style.configure("TNotebook", 
                       background=cls.DEEP_BLACK,
                       tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", 
                       background=cls.DEEP_BLACK, 
                       foreground=cls.SKY_BLUE,
                       padding=[10, 4],
                       font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                 background=[("selected", cls.DEEP_BLACK)],
                 foreground=[("selected", cls.NEON_CYAN)])
        
        # Progress bar style
        style.configure("Tron.Horizontal.TProgressbar", 
                       background=cls.NEON_CYAN, 
                       troughcolor=cls.DEEP_BLACK,
                       bordercolor=cls.DARK_TEAL,
                       lightcolor=cls.MAX_BLUE,
                       darkcolor=cls.SKY_BLUE)
        
        # Status indicators
        style.configure("Pending.TLabel", foreground=cls.DARK_TEAL)
        style.configure("Processing.TLabel", foreground=cls.SUNRAY)
        style.configure("Complete.TLabel", foreground=cls.NEON_CYAN)
        style.configure("Failed.TLabel", foreground=cls.MAGENTA)
        
        return style

class DeepfakeDetectionApp:
    """
    Main application class for the Deepfake Detection Platform.
    
    This class sets up the GUI with separate tabs for image, audio, and video
    upload and analysis, with a Tron Legacy-inspired visual style.
    """
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.setup_app_window()
        self.configure_theme()
        self.setup_variables()
        self.create_widgets()
        self.setup_event_queue()
        
        # Initialize backend components
        self.queue_manager = QueueManager()
        self.result_handler = ResultHandler()
        self.media_processor = MediaProcessor(self.queue_manager, self.result_handler)
        
        logger.info("Application initialized")
    
    def setup_app_window(self):
        """Configure the main application window."""
        self.root.title("Deepfake Detection Platform")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set window icon if available
        icon_path = os.path.join(os.path.dirname(__file__), "static", "icon.png")
        if os.path.exists(icon_path):
            try:
                icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon)
            except Exception as e:
                logger.warning(f"Could not load application icon: {e}")
    
    def configure_theme(self):
        """Apply the Tron Legacy theme to the application."""
        self.root.configure(background=TronTheme.DEEP_BLACK)
        self.style = TronTheme.setup_theme()
        
        # Load CSS file for web components (future use)
        self.css_path = os.path.join(os.path.dirname(__file__), "static", "tron_theme.css")
    
    def setup_variables(self):
        """Initialize application variables."""
        self.current_files = {
            'image': [],
            'audio': [],
            'video': []
        }
        self.processing_queue = []
        self.current_results = {}
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
    
    def create_widgets(self):
        """Create all application widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header()
        
        # Main content area with tabs
        self.create_tabs()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create the application header."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # App title
        title_label = ttk.Label(
            header_frame, 
            text="Deepfake Detection Platform", 
            style="Header.TLabel"
        )
        title_label.pack(side=tk.LEFT)
        
        # Right side controls
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        # Settings button
        settings_btn = ttk.Button(
            controls_frame, 
            text="⚙️ Settings", 
            command=self.show_settings
        )
        settings_btn.pack(side=tk.RIGHT, padx=5)
        
        # Help button
        help_btn = ttk.Button(
            controls_frame, 
            text="❓ Help", 
            command=self.show_help
        )
        help_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_tabs(self):
        """Create tabs for different media types."""
        self.tabs = ttk.Notebook(self.main_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for each media type
        self.image_tab = ttk.Frame(self.tabs)
        self.audio_tab = ttk.Frame(self.tabs)
        self.video_tab = ttk.Frame(self.tabs)
        self.results_tab = ttk.Frame(self.tabs)
        
        self.tabs.add(self.image_tab, text="Image Analysis")
        self.tabs.add(self.audio_tab, text="Audio Analysis")
        self.tabs.add(self.video_tab, text="Video Analysis")
        self.tabs.add(self.results_tab, text="Results & Reports")
        
        # Set up content for each tab
        self.setup_image_tab()
        self.setup_audio_tab()
        self.setup_video_tab()
        self.setup_results_tab()
    
    def setup_image_tab(self):
        """Set up the image analysis tab."""
        # Split into left (upload) and right (preview) panels
        panel_frame = ttk.Frame(self.image_tab)
        panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Upload controls
        upload_frame = ttk.Frame(panel_frame)
        upload_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Upload title
        upload_title = ttk.Label(
            upload_frame, 
            text="Upload Images", 
            style="Subheader.TLabel"
        )
        upload_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload instructions
        instructions = ttk.Label(
            upload_frame,
            text="Select one or more image files to analyze for potential deepfakes.",
            wraplength=300
        )
        instructions.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload buttons
        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        select_btn = ttk.Button(
            btn_frame,
            text="Select Files",
            command=lambda: self.select_files('image')
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_btn = ttk.Button(
            btn_frame,
            text="Clear All",
            command=lambda: self.clear_files('image')
        )
        clear_btn.pack(side=tk.LEFT)
        
        # File list
        list_frame = ttk.Frame(upload_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        list_label = ttk.Label(list_frame, text="Selected Files:")
        list_label.pack(anchor=tk.W)
        
        # Scrollable list
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_list = tk.Listbox(
            list_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            selectbackground=TronTheme.NEON_CYAN,
            selectforeground=TronTheme.DEEP_BLACK,
            height=10,
            yscrollcommand=list_scroll.set
        )
        self.image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.image_list.yview)
        
        # Analysis buttons
        analyze_frame = ttk.Frame(upload_frame)
        analyze_frame.pack(fill=tk.X, pady=10)
        
        analyze_btn = ttk.Button(
            analyze_frame,
            text="▶️ Analyze Images",
            command=lambda: self.start_analysis('image')
        )
        analyze_btn.pack(fill=tk.X)
        
        # Right panel - Preview
        preview_frame = ttk.Frame(panel_frame)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        preview_title = ttk.Label(
            preview_frame, 
            text="Image Preview", 
            style="Subheader.TLabel"
        )
        preview_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Preview canvas with border
        preview_canvas_frame = ttk.Frame(
            preview_frame,
            borderwidth=1,
            relief=tk.SOLID
        )
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_preview = tk.Canvas(
            preview_canvas_frame,
            background=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.image_preview.pack(fill=tk.BOTH, expand=True)
        
        # Preview instructions
        preview_instructions = ttk.Label(
            preview_frame,
            text="Select an image from the list to preview",
            wraplength=300
        )
        preview_instructions.pack(pady=10)
        
        # Bind selection event to update preview
        self.image_list.bind('<<ListboxSelect>>', self.update_image_preview)
    
    def setup_audio_tab(self):
        """Set up the audio analysis tab."""
        # Split into top (upload) and bottom (visualization) panels
        panel_frame = ttk.Frame(self.audio_tab)
        panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - Upload controls
        upload_frame = ttk.Frame(panel_frame)
        upload_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Upload title
        upload_title = ttk.Label(
            upload_frame, 
            text="Upload Audio Files", 
            style="Subheader.TLabel"
        )
        upload_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload instructions
        instructions = ttk.Label(
            upload_frame,
            text="Select one or more audio files to analyze for potential deepfakes.",
            wraplength=600
        )
        instructions.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload buttons
        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        select_btn = ttk.Button(
            btn_frame,
            text="Select Files",
            command=lambda: self.select_files('audio')
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_btn = ttk.Button(
            btn_frame,
            text="Clear All",
            command=lambda: self.clear_files('audio')
        )
        clear_btn.pack(side=tk.LEFT)
        
        # File list
        list_frame = ttk.Frame(upload_frame)
        list_frame.pack(fill=tk.X)
        
        list_label = ttk.Label(list_frame, text="Selected Files:")
        list_label.pack(anchor=tk.W)
        
        # Scrollable list
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.audio_list = tk.Listbox(
            list_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            selectbackground=TronTheme.NEON_CYAN,
            selectforeground=TronTheme.DEEP_BLACK,
            height=6,
            yscrollcommand=list_scroll.set
        )
        self.audio_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_scroll.config(command=self.audio_list.yview)
        
        # Analysis buttons
        analyze_frame = ttk.Frame(upload_frame)
        analyze_frame.pack(fill=tk.X, pady=10)
        
        analyze_btn = ttk.Button(
            analyze_frame,
            text="▶️ Analyze Audio",
            command=lambda: self.start_analysis('audio')
        )
        analyze_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        play_btn = ttk.Button(
            analyze_frame,
            text="🔊 Play Selected",
            command=self.play_selected_audio
        )
        play_btn.pack(side=tk.LEFT)
        
        # Bottom panel - Visualization
        viz_frame = ttk.Frame(panel_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        viz_title = ttk.Label(
            viz_frame, 
            text="Audio Visualization", 
            style="Subheader.TLabel"
        )
        viz_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Visualization canvas with border
        viz_canvas_frame = ttk.Frame(
            viz_frame,
            borderwidth=1,
            relief=tk.SOLID
        )
        viz_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.audio_viz = tk.Canvas(
            viz_canvas_frame,
            background=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.audio_viz.pack(fill=tk.BOTH, expand=True)
        
        # Visualization instructions
        viz_instructions = ttk.Label(
            viz_frame,
            text="Select an audio file from the list to visualize its waveform and spectrogram",
            wraplength=600
        )
        viz_instructions.pack(pady=10)
        
        # Bind selection event to update visualization
        self.audio_list.bind('<<ListboxSelect>>', self.update_audio_visualization)
    
    def setup_video_tab(self):
        """Set up the video analysis tab."""
        # Split into left (upload) and right (preview) panels
        panel_frame = ttk.Frame(self.video_tab)
        panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Upload controls
        upload_frame = ttk.Frame(panel_frame)
        upload_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Upload title
        upload_title = ttk.Label(
            upload_frame, 
            text="Upload Videos", 
            style="Subheader.TLabel"
        )
        upload_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload instructions
        instructions = ttk.Label(
            upload_frame,
            text="Select one or more video files to analyze for potential deepfakes.",
            wraplength=300
        )
        instructions.pack(anchor=tk.W, pady=(0, 10))
        
        # Upload buttons
        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        select_btn = ttk.Button(
            btn_frame,
            text="Select Files",
            command=lambda: self.select_files('video')
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_btn = ttk.Button(
            btn_frame,
            text="Clear All",
            command=lambda: self.clear_files('video')
        )
        clear_btn.pack(side=tk.LEFT)
        
        # File list
        list_frame = ttk.Frame(upload_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        list_label = ttk.Label(list_frame, text="Selected Files:")
        list_label.pack(anchor=tk.W)
        
        # Scrollable list
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.video_list = tk.Listbox(
            list_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            selectbackground=TronTheme.NEON_CYAN,
            selectforeground=TronTheme.DEEP_BLACK,
            height=10,
            yscrollcommand=list_scroll.set
        )
        self.video_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.video_list.yview)
        
        # Analysis buttons
        analyze_frame = ttk.Frame(upload_frame)
        analyze_frame.pack(fill=tk.X, pady=10)
        
        analyze_btn = ttk.Button(
            analyze_frame,
            text="▶️ Analyze Videos",
            command=lambda: self.start_analysis('video')
        )
        analyze_btn.pack(fill=tk.X)
        
        # Right panel - Preview
        preview_frame = ttk.Frame(panel_frame)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        preview_title = ttk.Label(
            preview_frame, 
            text="Video Preview", 
            style="Subheader.TLabel"
        )
        preview_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Preview canvas with border
        preview_canvas_frame = ttk.Frame(
            preview_frame,
            borderwidth=1,
            relief=tk.SOLID
        )
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_preview = tk.Canvas(
            preview_canvas_frame,
            background=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.video_preview.pack(fill=tk.BOTH, expand=True)
        
        # Video controls
        controls_frame = ttk.Frame(preview_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        play_btn = ttk.Button(
            controls_frame,
            text="▶️ Play",
            command=self.play_selected_video
        )
        play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        stop_btn = ttk.Button(
            controls_frame,
            text="⏹️ Stop",
            command=self.stop_video
        )
        stop_btn.pack(side=tk.LEFT)
        
        # Bind selection event to update preview
        self.video_list.bind('<<ListboxSelect>>', self.update_video_preview)
    
    def setup_results_tab(self):
        """Set up the results and reports tab."""
        # Results header
        results_title = ttk.Label(
            self.results_tab, 
            text="Detection Results", 
            style="Subheader.TLabel"
        )
        results_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Results notebook (tabs for different results)
        self.results_notebook = ttk.Notebook(self.results_tab)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Summary")
        
        # Detailed analysis tab
        self.detailed_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.detailed_tab, text="Detailed Analysis")
        
        # Visualizations tab
        self.viz_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.viz_tab, text="Visualizations")
        
        # Report tab
        self.report_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.report_tab, text="Report")
        
        # Setup contents of each tab
        self.setup_summary_tab()
        self.setup_detailed_tab()
        self.setup_viz_tab()
        self.setup_report_tab()
        
        # Export buttons
        export_frame = ttk.Frame(self.results_tab)
        export_frame.pack(fill=tk.X, pady=10)
        
        export_pdf_btn = ttk.Button(
            export_frame,
            text="Export as PDF",
            command=self.export_pdf
        )
        export_pdf_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        export_csv_btn = ttk.Button(
            export_frame,
            text="Export as CSV",
            command=self.export_csv
        )
        export_csv_btn.pack(side=tk.LEFT)
    
    def setup_summary_tab(self):
        """Set up the summary tab in results."""
        # Container frame with padding
        container = ttk.Frame(self.summary_tab, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Results overview section
        overview_frame = ttk.Frame(container)
        overview_frame.pack(fill=tk.X, pady=(0, 10))
        
        overview_label = ttk.Label(
            overview_frame,
            text="Result Overview",
            style="Subheader.TLabel"
        )
        overview_label.pack(anchor=tk.W)
        
        # Overview content frame with border
        overview_content = ttk.Frame(
            overview_frame,
            borderwidth=1,
            relief=tk.SOLID
        )
        overview_content.pack(fill=tk.X, pady=5)
        
        # Placeholder for summary info
        self.summary_info = ttk.Label(
            overview_content,
            text="No analysis results available.",
            padding=10
        )
        self.summary_info.pack(anchor=tk.W)
        
        # Confidence score section
        confidence_frame = ttk.Frame(container)
        confidence_frame.pack(fill=tk.X, pady=10)
        
        confidence_label = ttk.Label(
            confidence_frame,
            text="Confidence Score",
            style="Subheader.TLabel"
        )
        confidence_label.pack(anchor=tk.W)
        
        # Confidence meter
        meter_frame = ttk.Frame(confidence_frame, padding=5)
        meter_frame.pack(fill=tk.X)
        
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_meter = ttk.Progressbar(
            meter_frame,
            variable=self.confidence_var,
            style="Tron.Horizontal.TProgressbar",
            length=400,
            mode='determinate',
            maximum=100
        )
        self.confidence_meter.pack(fill=tk.X)
        
        # Confidence label
        self.confidence_text = tk.StringVar(value="No results")
        confidence_result = ttk.Label(
            meter_frame,
            textvariable=self.confidence_text,
            padding=(0, 5)
        )
        confidence_result.pack()
        
        # File list section
        files_frame = ttk.Frame(container)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        files_label = ttk.Label(
            files_frame,
            text="Analyzed Files",
            style="Subheader.TLabel"
        )
        files_label.pack(anchor=tk.W)
        
        # Scrollable list
        list_scroll = ttk.Scrollbar(files_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_list = tk.Listbox(
            files_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            selectbackground=TronTheme.NEON_CYAN,
            selectforeground=TronTheme.DEEP_BLACK,
            height=10,
            yscrollcommand=list_scroll.set
        )
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.results_list.yview)
        
        # Bind selection event to update result details
        self.results_list.bind('<<ListboxSelect>>', self.show_file_details)
    
    def setup_detailed_tab(self):
        """Set up the detailed analysis tab in results."""
        # Container frame with padding
        container = ttk.Frame(self.detailed_tab, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Results header
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        analysis_label = ttk.Label(
            header_frame,
            text="Detailed Analysis",
            style="Subheader.TLabel"
        )
        analysis_label.pack(anchor=tk.W)
        
        # Create scrollable text area for detailed results
        text_frame = ttk.Frame(container)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_scroll = ttk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.details_text = tk.Text(
            text_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            insertbackground=TronTheme.NEON_CYAN,
            borderwidth=1,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            wrap=tk.WORD,
            yscrollcommand=text_scroll.set
        )
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.details_text.yview)
        
        # Configure text tags for styling
        self.details_text.tag_config("header", 
                                    foreground=TronTheme.NEON_CYAN, 
                                    font=("Helvetica", 12, "bold"))
        self.details_text.tag_config("subheader", 
                                   foreground=TronTheme.SKY_BLUE, 
                                   font=("Helvetica", 10, "bold"))
        self.details_text.tag_config("highlight", 
                                   foreground=TronTheme.SUNRAY)
        self.details_text.tag_config("fake", 
                                   foreground=TronTheme.MAGENTA, 
                                   font=("Helvetica", 10, "bold"))
        self.details_text.tag_config("real", 
                                   foreground=TronTheme.NEON_CYAN, 
                                   font=("Helvetica", 10, "bold"))
        
        # Insert placeholder text
        self.details_text.insert(tk.END, "No detailed analysis available.\n\n")
        self.details_text.insert(tk.END, "Run analysis on files to see detailed results.")
        self.details_text.config(state=tk.DISABLED)  # Make read-only
    
    def setup_viz_tab(self):
        """Set up the visualizations tab in results."""
        # Container frame with padding
        container = ttk.Frame(self.viz_tab, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Visualization header
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        viz_label = ttk.Label(
            header_frame,
            text="Result Visualizations",
            style="Subheader.TLabel"
        )
        viz_label.pack(anchor=tk.W)
        
        # Visualization canvas with border
        viz_canvas_frame = ttk.Frame(
            container,
            borderwidth=1,
            relief=tk.SOLID
        )
        viz_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.viz_canvas = tk.Canvas(
            viz_canvas_frame,
            background=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Visualization type selector
        selector_frame = ttk.Frame(container)
        selector_frame.pack(fill=tk.X, pady=10)
        
        selector_label = ttk.Label(
            selector_frame,
            text="Visualization Type:"
        )
        selector_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.viz_type = tk.StringVar(value="heatmap")
        
        heatmap_radio = ttk.Radiobutton(
            selector_frame,
            text="Heatmap",
            variable=self.viz_type,
            value="heatmap",
            command=self.update_visualization
        )
        heatmap_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        graph_radio = ttk.Radiobutton(
            selector_frame,
            text="Confidence Graph",
            variable=self.viz_type,
            value="graph",
            command=self.update_visualization
        )
        graph_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        regions_radio = ttk.Radiobutton(
            selector_frame,
            text="Region Analysis",
            variable=self.viz_type,
            value="regions",
            command=self.update_visualization
        )
        regions_radio.pack(side=tk.LEFT)
    
    def setup_report_tab(self):
        """Set up the report tab in results."""
        # Container frame with padding
        container = ttk.Frame(self.report_tab, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Report header
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        report_label = ttk.Label(
            header_frame,
            text="Analysis Report",
            style="Subheader.TLabel"
        )
        report_label.pack(anchor=tk.W)
        
        # Create scrollable text area for report
        text_frame = ttk.Frame(container)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_scroll = ttk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.report_text = tk.Text(
            text_frame,
            background=TronTheme.DEEP_BLACK,
            foreground="white",
            insertbackground=TronTheme.NEON_CYAN,
            borderwidth=1,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            wrap=tk.WORD,
            yscrollcommand=text_scroll.set
        )
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.report_text.yview)
        
        # Configure text tags for styling
        self.report_text.tag_config("header", 
                                   foreground=TronTheme.NEON_CYAN, 
                                   font=("Helvetica", 12, "bold"))
        self.report_text.tag_config("subheader", 
                                  foreground=TronTheme.SKY_BLUE, 
                                  font=("Helvetica", 10, "bold"))
        self.report_text.tag_config("highlight", 
                                  foreground=TronTheme.SUNRAY)
        
        # Insert placeholder text
        self.report_text.insert(tk.END, "No report available.\n\n")
        self.report_text.insert(tk.END, "Run analysis on files to generate reports.")
        self.report_text.config(state=tk.DISABLED)  # Make read-only
    
    def create_status_bar(self):
        """Create the status bar at the bottom of the application."""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        # Status text
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            padding=(5, 2)
        )
        status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            style="Tron.Horizontal.TProgressbar",
            length=200,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
    
    def setup_event_queue(self):
        """Set up the event queue for threaded operations."""
        self.event_queue = queue.Queue()
        self.root.after(100, self.process_event_queue)
    
    def process_event_queue(self):
        """Process events from the event queue."""
        try:
            # Handle up to 10 events per cycle
            for _ in range(10):
                event = self.event_queue.get_nowait()
                
                if event['type'] == 'status':
                    self.status_var.set(event['message'])
                
                elif event['type'] == 'progress':
                    self.progress_var.set(event['value'])
                
                elif event['type'] == 'result':
                    self.handle_result_event(event)
                
                elif event['type'] == 'error':
                    self.handle_error_event(event)
                
                self.event_queue.task_done()
        
        except queue.Empty:
            pass
        
        # Reschedule
        self.root.after(100, self.process_event_queue)
    
    def handle_result_event(self, event):
        """Handle result events from analysis."""
        # Store result in our results dictionary
        result_id = event['result_id']
        self.current_results[result_id] = event['result']
        
        # Update results list
        filename = event['result']['filename']
        self.results_list.insert(tk.END, filename)
        
        # Update status
        self.status_var.set(f"Analysis complete for {filename}")
        
        # Switch to results tab
        self.tabs.select(self.results_tab)
        
        # Update summary with latest result
        self.update_summary()
    
    def handle_error_event(self, event):
        """Handle error events."""
        messagebox.showerror("Error", event['message'])
        self.status_var.set(f"Error: {event['message']}")
    
    def select_files(self, media_type):
        """Open file dialog to select files for analysis."""
        filetypes = []
        
        if media_type == 'image':
            filetypes = [
                ('Image Files', '*.jpg *.jpeg *.png *.gif *.bmp'),
                ('All Files', '*.*')
            ]
        elif media_type == 'audio':
            filetypes = [
                ('Audio Files', '*.wav *.mp3 *.flac'),
                ('All Files', '*.*')
            ]
        elif media_type == 'video':
            filetypes = [
                ('Video Files', '*.mp4 *.avi *.mov'),
                ('All Files', '*.*')
            ]
        
        files = filedialog.askopenfilenames(
            title=f"Select {media_type.capitalize()} Files",
            filetypes=filetypes
        )
        
        if not files:
            return
        
        # Clear existing files if adding new ones
        if media_type == 'image':
            self.clear_files('image')
        elif media_type == 'audio':
            self.clear_files('audio')
        elif media_type == 'video':
            self.clear_files('video')
        
        # Validate files and add to list
        for file_path in files:
            is_valid, message, detected_type = validate_file(file_path)
            
            if is_valid and detected_type == media_type:
                self.add_file(file_path, media_type)
            else:
                messagebox.showwarning(
                    "Invalid File",
                    f"File {os.path.basename(file_path)} is not a valid {media_type} file: {message}"
                )
    
    def add_file(self, file_path, media_type):
        """Add a file to the appropriate list."""
        # Add to internal tracking
        self.current_files[media_type].append(file_path)
        
        # Add to UI list
        filename = os.path.basename(file_path)
        
        if media_type == 'image':
            self.image_list.insert(tk.END, filename)
        elif media_type == 'audio':
            self.audio_list.insert(tk.END, filename)
        elif media_type == 'video':
            self.video_list.insert(tk.END, filename)
        
        # Update status
        self.status_var.set(f"Added {filename}")
    
    def clear_files(self, media_type):
        """Clear all files from the specified list."""
        self.current_files[media_type] = []
        
        if media_type == 'image':
            self.image_list.delete(0, tk.END)
            self.image_preview.delete("all")
        elif media_type == 'audio':
            self.audio_list.delete(0, tk.END)
            self.audio_viz.delete("all")
        elif media_type == 'video':
            self.video_list.delete(0, tk.END)
            self.video_preview.delete("all")
        
        # Update status
        self.status_var.set(f"Cleared {media_type} files")
    
    def update_image_preview(self, event):
        """Update the image preview when a file is selected."""
        # Get selected file
        selection = self.image_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.current_files['image']):
            return
        
        file_path = self.current_files['image'][index]
        
        # Load and display image
        try:
            from PIL import Image, ImageTk
            
            # Load image and resize to fit canvas
            image = Image.open(file_path)
            
            # Get canvas dimensions
            canvas_width = self.image_preview.winfo_width()
            canvas_height = self.image_preview.winfo_height()
            
            # No resize needed if canvas not yet drawn
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 300
            
            # Calculate resize ratio preserving aspect ratio
            img_width, img_height = image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear previous image
            self.image_preview.delete("all")
            
            # Store reference to prevent garbage collection
            self.current_photo = photo
            
            # Display image
            self.image_preview.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            
            # Update status
            self.status_var.set(f"Viewing {os.path.basename(file_path)}")
            
        except Exception as e:
            self.image_preview.delete("all")
            self.image_preview.create_text(
                200, 150,
                text=f"Error loading image:\n{str(e)}",
                fill=TronTheme.MAGENTA,
                anchor=tk.CENTER
            )
            logger.error(f"Error loading image preview: {e}")
    
    def update_audio_visualization(self, event):
        """Update audio visualization when a file is selected."""
        # Placeholder for audio visualization
        # In a real implementation, this would use libraries like librosa and matplotlib
        # to create waveform and spectrogram visualizations
        
        # Get selected file
        selection = self.audio_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.current_files['audio']):
            return
        
        file_path = self.current_files['audio'][index]
        filename = os.path.basename(file_path)
        
        # Clear canvas
        self.audio_viz.delete("all")
        
        # Create placeholder visualization
        canvas_width = self.audio_viz.winfo_width()
        canvas_height = self.audio_viz.winfo_height()
        
        # No visualization needed if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # Draw waveform placeholder
        wave_height = canvas_height // 2
        wave_center = wave_height // 2
        
        # Create simulated waveform
        for i in range(0, canvas_width, 3):
            # Generate fake waveform heights
            import math
            import random
            
            # Hash the filename to get consistent pattern
            seed = sum(ord(c) for c in filename)
            random.seed(seed + i)
            
            # Generate amplitude using sine wave with noise
            base_amp = math.sin(i * 0.05) * 30
            noise = random.uniform(-10, 10)
            amp = base_amp + noise
            
            # Draw line segment
            self.audio_viz.create_line(
                i, wave_center - amp,
                i + 2, wave_center - (base_amp + random.uniform(-10, 10)),
                fill=TronTheme.NEON_CYAN,
                width=1
            )
        
        # Draw spectrogram placeholder
        spec_top = wave_height
        spec_height = wave_height
        
        # Create gradient background for spectrogram
        for y in range(spec_height):
            # Create color gradient from blue to cyan
            ratio = y / spec_height
            r = int(0)
            g = int(255 * (1 - ratio))
            b = int(255)
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.audio_viz.create_line(
                0, spec_top + y,
                canvas_width, spec_top + y,
                fill=color
            )
        
        # Add some "frequency" bars
        for i in range(0, canvas_width, 20):
            # Generate random height for frequency bar
            random.seed(seed + i + 1000)
            height = random.randint(10, spec_height - 10)
            
            self.audio_viz.create_rectangle(
                i, spec_top + spec_height - height,
                i + 10, spec_top + spec_height,
                fill=TronTheme.MAGENTA,
                outline=""
            )
        
        # Add labels
        self.audio_viz.create_text(
            10, 10,
            text="Waveform",
            fill=TronTheme.NEON_CYAN,
            anchor=tk.NW
        )
        
        self.audio_viz.create_text(
            10, spec_top + 10,
            text="Spectrogram",
            fill=TronTheme.NEON_CYAN,
            anchor=tk.NW
        )
        
        # Update status
        self.status_var.set(f"Visualizing {filename}")
    
    def update_video_preview(self, event):
        """Update video preview when a file is selected."""
        # Placeholder for video preview functionality
        # In a real implementation, this would use libraries like cv2 or moviepy
        # to extract and display video frames
        
        # Get selected file
        selection = self.video_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.current_files['video']):
            return
        
        file_path = self.current_files['video'][index]
        filename = os.path.basename(file_path)
        
        # Clear canvas
        self.video_preview.delete("all")
        
        # Create placeholder preview
        canvas_width = self.video_preview.winfo_width()
        canvas_height = self.video_preview.winfo_height()
        
        # No preview needed if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # Draw placeholder frame
        self.video_preview.create_rectangle(
            10, 10,
            canvas_width - 10, canvas_height - 10,
            outline=TronTheme.NEON_CYAN,
            width=2
        )
        
        # Draw placeholder text
        self.video_preview.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=f"Video Preview\n{filename}",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 14, "bold"),
            anchor=tk.CENTER
        )
        
        # Draw play button overlay
        button_size = 60
        self.video_preview.create_oval(
            (canvas_width - button_size) // 2,
            (canvas_height - button_size) // 2,
            (canvas_width + button_size) // 2,
            (canvas_height + button_size) // 2,
            fill=TronTheme.DEEP_BLACK,
            outline=TronTheme.NEON_CYAN,
            width=2
        )
        
        # Play triangle
        triangle_points = [
            (canvas_width // 2) - 10, (canvas_height // 2) - 15,
            (canvas_width // 2) - 10, (canvas_height // 2) + 15,
            (canvas_width // 2) + 20, (canvas_height // 2)
        ]
        self.video_preview.create_polygon(
            *triangle_points,
            fill=TronTheme.NEON_CYAN,
            outline=""
        )
        
        # Update status
        self.status_var.set(f"Selected {filename}")
    
    def play_selected_audio(self):
        """Play the selected audio file."""
        # Placeholder for audio playback functionality
        # In a real implementation, this would use libraries like pygame.mixer
        
        # Get selected file
        selection = self.audio_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select an audio file to play")
            return
        
        index = selection[0]
        if index >= len(self.current_files['audio']):
            return
        
        file_path = self.current_files['audio'][index]
        filename = os.path.basename(file_path)
        
        # Show message since we don't have actual playback
        messagebox.showinfo("Audio Playback", f"Playing {filename}\n\n(Audio playback not implemented in this version)")
        
        # Update status
        self.status_var.set(f"Playing {filename}")
    
    def play_selected_video(self):
        """Play the selected video file."""
        # Placeholder for video playback functionality
        # In a real implementation, this would use libraries like cv2 or moviepy
        
        # Get selected file
        selection = self.video_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a video file to play")
            return
        
        index = selection[0]
        if index >= len(self.current_files['video']):
            return
        
        file_path = self.current_files['video'][index]
        filename = os.path.basename(file_path)
        
        # Show message since we don't have actual playback
        messagebox.showinfo("Video Playback", f"Playing {filename}\n\n(Video playback not implemented in this version)")
        
        # Update status
        self.status_var.set(f"Playing {filename}")
    
    def stop_video(self):
        """Stop video playback."""
        # Placeholder for stop functionality
        messagebox.showinfo("Video Playback", "Video playback stopped")
        self.status_var.set("Video playback stopped")
    
    def start_analysis(self, media_type):
        """Start analysis of the selected files."""
        files = self.current_files[media_type]
        
        if not files:
            messagebox.showinfo("Info", f"No {media_type} files selected")
            return
        
        # Update status
        self.status_var.set(f"Starting {media_type} analysis...")
        self.progress_var.set(0)
        
        # Start analysis in a separate thread
        threading.Thread(
            target=self.run_analysis,
            args=(files, media_type),
            daemon=True
        ).start()
    
    def run_analysis(self, files, media_type):
        """Run analysis in a background thread."""
        total_files = len(files)
        
        for i, file_path in enumerate(files):
            try:
                # Report progress
                progress = (i / total_files) * 100
                self.event_queue.put({
                    'type': 'progress',
                    'value': progress
                })
                
                filename = os.path.basename(file_path)
                self.event_queue.put({
                    'type': 'status',
                    'message': f"Analyzing {filename} ({i+1}/{total_files})"
                })
                
                # In a real implementation, this would call the actual detector
                # For now, we'll simulate processing with a delay
                time.sleep(1)
                
                # Create a simulated result
                import random
                is_fake = random.random() < 0.5
                confidence = random.uniform(0.6, 0.95) if is_fake else random.uniform(0.7, 0.98)
                
                result = {
                    'filename': filename,
                    'file_path': file_path,
                    'media_type': media_type,
                    'is_fake': is_fake,
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'regions': self.generate_fake_regions(media_type),
                    'details': self.generate_fake_details(media_type, is_fake)
                }
                
                # Send result to main thread
                self.event_queue.put({
                    'type': 'result',
                    'result_id': file_path,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                self.event_queue.put({
                    'type': 'error',
                    'message': f"Error analyzing {os.path.basename(file_path)}: {str(e)}"
                })
        
        # Analysis complete
        self.event_queue.put({
            'type': 'progress',
            'value': 100
        })
        
        self.event_queue.put({
            'type': 'status',
            'message': f"Analysis complete for {total_files} files"
        })
    
    def generate_fake_regions(self, media_type):
        """Generate fake region data for simulated results."""
        import random
        
        if media_type == 'image':
            # For images, generate rectangles
            regions = []
            for i in range(random.randint(1, 3)):
                regions.append({
                    'x': random.uniform(0.1, 0.8),
                    'y': random.uniform(0.1, 0.8),
                    'width': random.uniform(0.1, 0.3),
                    'height': random.uniform(0.1, 0.3),
                    'confidence': random.uniform(0.6, 0.95),
                    'type': random.choice(['face', 'texture', 'artifact'])
                })
            return regions
        
        elif media_type == 'audio':
            # For audio, generate time segments
            regions = []
            for i in range(random.randint(1, 4)):
                start = random.uniform(0, 50)
                duration = random.uniform(1, 10)
                regions.append({
                    'start_time': start,
                    'end_time': start + duration,
                    'confidence': random.uniform(0.6, 0.95),
                    'type': random.choice(['voice', 'splice', 'artifact'])
                })
            return regions
        
        elif media_type == 'video':
            # For video, generate frame ranges
            regions = []
            for i in range(random.randint(1, 3)):
                start = random.randint(0, 300)
                duration = random.randint(10, 60)
                regions.append({
                    'start_frame': start,
                    'end_frame': start + duration,
                    'confidence': random.uniform(0.6, 0.95),
                    'type': random.choice(['face', 'motion', 'sync'])
                })
            return regions
        
        return []
    
    def generate_fake_details(self, media_type, is_fake):
        """Generate fake detailed analysis for simulated results."""
        import random
        
        if is_fake:
            indicators = [
                "Inconsistent compression artifacts detected",
                "Unnatural facial features identified",
                "Temporal inconsistencies present",
                "Audio-visual synchronization issues",
                "Unusual edge patterns detected",
                "Neural network confidence score indicates manipulation",
                "Spectral analysis reveals artifacts"
            ]
            
            # Select random indicators based on media type
            if media_type == 'image':
                selected = random.sample(indicators[:3] + [indicators[4], indicators[5]], 3)
            elif media_type == 'audio':
                selected = random.sample([indicators[2], indicators[5], indicators[6]], 2)
            else:  # video
                selected = random.sample([indicators[1], indicators[2], indicators[3], indicators[5]], 3)
                
            details = "The analysis indicates this content is likely manipulated:\n\n"
            for indicator in selected:
                details += f"• {indicator}\n"
                
            details += f"\nThe detection confidence is {random.uniform(70, 95):.1f}%."
                
        else:
            authentic_indicators = [
                "Consistent compression patterns throughout",
                "Natural feature distribution",
                "Temporal consistency verified",
                "Audio-visual synchronization confirmed",
                "Expected edge pattern distribution",
                "Neural network confidence score supports authenticity",
                "Clean spectral signature"
            ]
            
            # Select random indicators based on media type
            if media_type == 'image':
                selected = random.sample(authentic_indicators[:2] + [authentic_indicators[4], authentic_indicators[5]], 3)
            elif media_type == 'audio':
                selected = random.sample([authentic_indicators[2], authentic_indicators[5], authentic_indicators[6]], 2)
            else:  # video
                selected = random.sample([authentic_indicators[1], authentic_indicators[2], authentic_indicators[3], authentic_indicators[5]], 3)
                
            details = "The analysis indicates this content is likely authentic:\n\n"
            for indicator in selected:
                details += f"• {indicator}\n"
                
            details += f"\nThe authenticity confidence is {random.uniform(75, 98):.1f}%."
        
        return details
    
    def update_summary(self):
        """Update the summary tab with result information."""
        # Count results by type
        total_count = len(self.current_results)
        if total_count == 0:
            return
        
        fake_count = sum(1 for result in self.current_results.values() if result['is_fake'])
        real_count = total_count - fake_count
        
        # Calculate overall confidence
        avg_confidence = sum(result['confidence'] for result in self.current_results.values()) / total_count
        
        # Update summary info
        summary_text = f"Analysis Results Summary:\n"
        summary_text += f"Total files analyzed: {total_count}\n"
        summary_text += f"Detected as likely fake: {fake_count}\n"
        summary_text += f"Detected as likely real: {real_count}\n"
        summary_text += f"Average confidence: {avg_confidence:.1%}"
        
        self.summary_info.config(text=summary_text)
        
        # Update confidence meter
        self.confidence_var.set(avg_confidence * 100)
        
        if fake_count > real_count:
            status = "Likely manipulated content detected"
            self.confidence_text.set(f"Manipulation Confidence: {avg_confidence:.1%}")
        else:
            status = "Content appears mostly authentic"
            self.confidence_text.set(f"Authenticity Confidence: {avg_confidence:.1%}")
        
        # Update status
        self.status_var.set(status)
        
        # Update detailed tab with most recent result
        latest_result = list(self.current_results.values())[-1]
        self.update_details(latest_result)
        
        # Update visualization
        self.update_visualization()
        
        # Update report
        self.generate_report()
    
    def show_file_details(self, event):
        """Show details for the selected file in results list."""
        selection = self.results_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= self.results_list.size():
            return
        
        filename = self.results_list.get(index)
        
        # Find the result for this filename
        for result in self.current_results.values():
            if result['filename'] == filename:
                self.update_details(result)
                self.update_visualization()
                break
    
    def update_details(self, result):
        """Update the detailed analysis tab with result information."""
        # Enable text widget for editing
        self.details_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.details_text.delete(1.0, tk.END)
        
        # Add header
        self.details_text.insert(tk.END, f"Detailed Analysis for {result['filename']}\n\n", "header")
        
        # Add result classification
        if result['is_fake']:
            self.details_text.insert(tk.END, "Classification: ", "subheader")
            self.details_text.insert(tk.END, "LIKELY FAKE / MANIPULATED\n\n", "fake")
        else:
            self.details_text.insert(tk.END, "Classification: ", "subheader")
            self.details_text.insert(tk.END, "LIKELY AUTHENTIC\n\n", "real")
        
        # Add confidence score
        self.details_text.insert(tk.END, f"Confidence: {result['confidence']:.1%}\n\n")
        
        # Add detailed analysis
        self.details_text.insert(tk.END, "Analysis Details:\n", "subheader")
        self.details_text.insert(tk.END, f"{result['details']}\n\n")
        
        # Add region information if available
        if result['regions']:
            self.details_text.insert(tk.END, "Detected Regions:\n", "subheader")
            for i, region in enumerate(result['regions']):
                if result['media_type'] == 'image':
                    self.details_text.insert(tk.END, f"{i+1}. Type: {region['type']}, ")
                    self.details_text.insert(tk.END, f"Position: ({region['x']:.2f}, {region['y']:.2f}), ")
                    self.details_text.insert(tk.END, f"Size: {region['width']:.2f}x{region['height']:.2f}, ")
                    self.details_text.insert(tk.END, f"Confidence: {region['confidence']:.1%}\n")
                elif result['media_type'] == 'audio':
                    self.details_text.insert(tk.END, f"{i+1}. Type: {region['type']}, ")
                    self.details_text.insert(tk.END, f"Time Range: {region['start_time']:.2f}s - {region['end_time']:.2f}s, ")
                    self.details_text.insert(tk.END, f"Confidence: {region['confidence']:.1%}\n")
                elif result['media_type'] == 'video':
                    self.details_text.insert(tk.END, f"{i+1}. Type: {region['type']}, ")
                    self.details_text.insert(tk.END, f"Frame Range: {region['start_frame']} - {region['end_frame']}, ")
                    self.details_text.insert(tk.END, f"Confidence: {region['confidence']:.1%}\n")
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        self.details_text.insert(tk.END, f"\nAnalysis completed: {timestamp}\n")
        
        # Make read-only again
        self.details_text.config(state=tk.DISABLED)
    
    def update_visualization(self):
        """Update the visualization tab based on the selected type."""
        # Get selected visualization type
        viz_type = self.viz_type.get()
        
        # Clear canvas
        self.viz_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.viz_canvas.winfo_width()
        canvas_height = self.viz_canvas.winfo_height()
        
        # No visualization needed if canvas not yet drawn
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        # Check if we have results
        if not self.current_results:
            self.viz_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="No analysis results available",
                fill=TronTheme.NEON_CYAN,
                font=("Helvetica", 12),
                anchor=tk.CENTER
            )
            return
        
        # Get selected result (from results list if something is selected, otherwise latest)
        selected_result = None
        selection = self.results_list.curselection()
        if selection:
            filename = self.results_list.get(selection[0])
            for result in self.current_results.values():
                if result['filename'] == filename:
                    selected_result = result
                    break
        
        if not selected_result:
            # Use the latest result
            selected_result = list(self.current_results.values())[-1]
        
        # Create different visualizations based on type
        if viz_type == "heatmap":
            self.create_heatmap_visualization(selected_result, canvas_width, canvas_height)
        elif viz_type == "graph":
            self.create_confidence_graph(canvas_width, canvas_height)
        elif viz_type == "regions":
            self.create_region_visualization(selected_result, canvas_width, canvas_height)
    
    def create_heatmap_visualization(self, result, width, height):
        """Create a heatmap visualization for image/video detection."""
        # Draw title
        self.viz_canvas.create_text(
            width // 2,
            20,
            text=f"Heatmap Visualization for {result['filename']}",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 12, "bold"),
            anchor=tk.CENTER
        )
        
        # Draw border
        self.viz_canvas.create_rectangle(
            50, 50,
            width - 50, height - 50,
            outline=TronTheme.NEON_CYAN,
            width=2
        )
        
        # Draw gradient background
        for y in range(50, height - 50):
            # Create color gradient
            ratio = (y - 50) / (height - 100)
            
            # Color goes from dark blue to cyan
            r = int(0)
            g = int(255 * ratio)
            b = int(255)
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.viz_canvas.create_line(
                50, y,
                width - 50, y,
                fill=color
            )
        
        # If we have regions, visualize them
        if result['regions']:
            for region in result['regions']:
                if result['media_type'] == 'image':
                    # Convert region coordinates to canvas coordinates
                    x = 50 + region['x'] * (width - 100)
                    y = 50 + region['y'] * (height - 100)
                    w = region['width'] * (width - 100)
                    h = region['height'] * (height - 100)
                    
                    # Draw region with opacity based on confidence
                    alpha = int(region['confidence'] * 255)
                    color = TronTheme.MAGENTA if result['is_fake'] else TronTheme.NEON_CYAN
                    
                    # Draw rectangle
                    self.viz_canvas.create_rectangle(
                        x, y,
                        x + w, y + h,
                        outline=color,
                        width=2
                    )
                    
                    # Draw label
                    self.viz_canvas.create_text(
                        x + w // 2,
                        y + h // 2,
                        text=f"{region['type']}\n{region['confidence']:.1%}",
                        fill="white",
                        font=("Helvetica", 10, "bold"),
                        anchor=tk.CENTER
                    )
        
        # Draw legend
        legend_y = height - 30
        
        # Fake indicator
        self.viz_canvas.create_rectangle(
            width - 150, legend_y,
            width - 130, legend_y + 15,
            fill=TronTheme.MAGENTA,
            outline=""
        )
        self.viz_canvas.create_text(
            width - 125, legend_y + 7,
            text="Manipulated",
            fill="white",
            anchor=tk.W
        )
        
        # Real indicator
        self.viz_canvas.create_rectangle(
            width - 300, legend_y,
            width - 280, legend_y + 15,
            fill=TronTheme.NEON_CYAN,
            outline=""
        )
        self.viz_canvas.create_text(
            width - 275, legend_y + 7,
            text="Authentic",
            fill="white",
            anchor=tk.W
        )
    
    def create_confidence_graph(self, width, height):
        """Create a confidence graph visualization."""
        # Draw title
        self.viz_canvas.create_text(
            width // 2,
            20,
            text="Confidence Scores for All Files",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 12, "bold"),
            anchor=tk.CENTER
        )
        
        # Draw axes
        self.viz_canvas.create_line(
            50, height - 50,
            width - 50, height - 50,
            fill=TronTheme.NEON_CYAN,
            width=2
        )
        self.viz_canvas.create_line(
            50, 50,
            50, height - 50,
            fill=TronTheme.NEON_CYAN,
            width=2
        )
        
        # Draw axis labels
        self.viz_canvas.create_text(
            width // 2,
            height - 20,
            text="Files",
            fill=TronTheme.NEON_CYAN,
            anchor=tk.CENTER
        )
        self.viz_canvas.create_text(
            20,
            height // 2,
            text="Confidence",
            fill=TronTheme.NEON_CYAN,
            angle=90,
            anchor=tk.CENTER
        )
        
        # Draw y-axis tick marks
        for i in range(0, 101, 20):
            y = height - 50 - (i / 100) * (height - 100)
            self.viz_canvas.create_line(
                45, y,
                55, y,
                fill=TronTheme.NEON_CYAN,
                width=1
            )
            self.viz_canvas.create_text(
                40, y,
                text=f"{i}%",
                fill=TronTheme.NEON_CYAN,
                anchor=tk.E
            )
        
        # Plot confidence scores
        if self.current_results:
            results = list(self.current_results.values())
            bar_width = min(40, (width - 100) / len(results))
            
            for i, result in enumerate(results):
                # Calculate bar position
                x = 50 + (i + 0.5) * ((width - 100) / len(results))
                
                # Calculate bar height
                bar_height = result['confidence'] * (height - 100)
                
                # Draw bar
                color = TronTheme.MAGENTA if result['is_fake'] else TronTheme.NEON_CYAN
                self.viz_canvas.create_rectangle(
                    x - bar_width / 2, height - 50 - bar_height,
                    x + bar_width / 2, height - 50,
                    fill=color,
                    outline="white",
                    width=1
                )
                
                # Draw file label (shortened)
                filename = result['filename']
                if len(filename) > 12:
                    filename = filename[:10] + "..."
                
                self.viz_canvas.create_text(
                    x,
                    height - 35,
                    text=filename,
                    fill="white",
                    angle=45,
                    anchor=tk.E
                )
                
                # Draw confidence value
                self.viz_canvas.create_text(
                    x,
                    height - 50 - bar_height - 10,
                    text=f"{result['confidence']:.0%}",
                    fill="white",
                    anchor=tk.S
                )
        else:
            self.viz_canvas.create_text(
                width // 2,
                height // 2,
                text="No data available",
                fill=TronTheme.NEON_CYAN,
                font=("Helvetica", 12),
                anchor=tk.CENTER
            )
    
    def create_region_visualization(self, result, width, height):
        """Create a visualization of detected regions."""
        # Draw title
        self.viz_canvas.create_text(
            width // 2,
            20,
            text=f"Region Analysis for {result['filename']}",
            fill=TronTheme.NEON_CYAN,
            font=("Helvetica", 12, "bold"),
            anchor=tk.CENTER
        )
        
        # Draw based on media type
        if result['media_type'] == 'image':
            # Draw image outline
            self.viz_canvas.create_rectangle(
                50, 50,
                width - 50, height - 150,
                outline=TronTheme.NEON_CYAN,
                width=2
            )
            
            # Draw regions
            if result['regions']:
                for region in result['regions']:
                    # Convert region coordinates to canvas coordinates
                    x = 50 + region['x'] * (width - 100)
                    y = 50 + region['y'] * (height - 200)
                    w = region['width'] * (width - 100)
                    h = region['height'] * (height - 200)
                    
                    # Draw region
                    self.viz_canvas.create_rectangle(
                        x, y,
                        x + w, y + h,
                        outline=TronTheme.MAGENTA,
                        width=2
                    )
            
            # Region list at bottom
            list_y = height - 130
            self.viz_canvas.create_text(
                50,
                list_y,
                text="Detected Regions:",
                fill=TronTheme.NEON_CYAN,
                font=("Helvetica", 10, "bold"),
                anchor=tk.NW
            )
            
            if result['regions']:
                for i, region in enumerate(result['regions']):
                    self.viz_canvas.create_text(
                        50,
                        list_y + 20 + i * 20,
                        text=f"{i+1}. {region['type']} - Confidence: {region['confidence']:.1%}",
                        fill="white",
                        anchor=tk.NW
                    )
            else:
                self.viz_canvas.create_text(
                    50,
                    list_y + 20,
                    text="No specific regions detected",
                    fill="white",
                    anchor=tk.NW
                )
                
        elif result['media_type'] == 'audio':
            # Draw timeline
            timeline_y = height // 2
            self.viz_canvas.create_line(
                50, timeline_y,
                width - 50, timeline_y,
                fill=TronTheme.NEON_CYAN,
                width=2
            )
            
            # Draw time markers
            for i in range(6):
                x = 50 + i * (width - 100) / 5
                self.viz_canvas.create_line(
                    x, timeline_y - 5,
                    x, timeline_y + 5,
                    fill=TronTheme.NEON_CYAN,
                    width=1
                )
                self.viz_canvas.create_text(
                    x, timeline_y + 20,
                    text=f"{i*10}s",
                    fill=TronTheme.NEON_CYAN,
                    anchor=tk.N
                )
            
            # Draw regions on timeline
            if result['regions']:
                max_time = max(region['end_time'] for region in result['regions'])
                time_scale = (width - 100) / max(max_time, 50)  # Scale factor
                
                for region in result['regions']:
                    start_x = 50 + region['start_time'] * time_scale
                    end_x = 50 + region['end_time'] * time_scale
                    
                    # Draw region bar
                    self.viz_canvas.create_rectangle(
                        start_x, timeline_y - 15,
                        end_x, timeline_y + 15,
                        fill=TronTheme.MAGENTA,
                        outline="white",
                        width=1
                    )
                    
                    # Draw label if enough space
                    if end_x - start_x > 40:
                        self.viz_canvas.create_text(
                            (start_x + end_x) / 2,
                            timeline_y,
                            text=region['type'],
                            fill="white",
                            font=("Helvetica", 8),
                            anchor=tk.CENTER
                        )
            
            # Region list at bottom
            list_y = timeline_y + 50
            self.viz_canvas.create_text(
                50,
                list_y,
                text="Detected Audio Segments:",
                fill=TronTheme.NEON_CYAN,
                font=("Helvetica", 10, "bold"),
                anchor=tk.NW
            )
            
            if result['regions']:
                for i, region in enumerate(result['regions']):
                    self.viz_canvas.create_text(
                        50,
                        list_y + 20 + i * 20,
                        text=f"{i+1}. {region['type']} - Time: {region['start_time']:.1f}s to {region['end_time']:.1f}s - Confidence: {region['confidence']:.1%}",
                        fill="white",
                        anchor=tk.NW
                    )
            else:
                self.viz_canvas.create_text(
                    50,
                    list_y + 20,
                    text="No specific segments detected",
                    fill="white",
                    anchor=tk.NW
                )
                
        elif result['media_type'] == 'video':
            # Draw timeline
            timeline_y = height // 2
            self.viz_canvas.create_line(
                50, timeline_y,
                width - 50, timeline_y,
                fill=TronTheme.NEON_CYAN,
                width=2
            )
            
            # Draw frame markers
            for i in range(6):
                x = 50 + i * (width - 100) / 5
                self.viz_canvas.create_line(
                    x, timeline_y - 5,
                    x, timeline_y + 5,
                    fill=TronTheme.NEON_CYAN,
                    width=1
                )
                self.viz_canvas.create_text(
                    x, timeline_y + 20,
                    text=f"Frame {i*60}",
                    fill=TronTheme.NEON_CYAN,
                    anchor=tk.N
                )
            
            # Draw regions on timeline
            if result['regions']:
                max_frame = max(region['end_frame'] for region in result['regions'])
                frame_scale = (width - 100) / max(max_frame, 300)  # Scale factor
                
                for region in result['regions']:
                    start_x = 50 + region['start_frame'] * frame_scale
                    end_x = 50 + region['end_frame'] * frame_scale
                    
                    # Draw region bar
                    self.viz_canvas.create_rectangle(
                        start_x, timeline_y - 15,
                        end_x, timeline_y + 15,
                        fill=TronTheme.MAGENTA,
                        outline="white",
                        width=1
                    )
                    
                    # Draw label if enough space
                    if end_x - start_x > 40:
                        self.viz_canvas.create_text(
                            (start_x + end_x) / 2,
                            timeline_y,
                            text=region['type'],
                            fill="white",
                            font=("Helvetica", 8),
                            anchor=tk.CENTER
                        )
            
            # Region list at bottom
            list_y = timeline_y + 50
            self.viz_canvas.create_text(
                50,
                list_y,
                text="Detected Video Segments:",
                fill=TronTheme.NEON_CYAN,
                font=("Helvetica", 10, "bold"),
                anchor=tk.NW
            )
            
            if result['regions']:
                for i, region in enumerate(result['regions']):
                    self.viz_canvas.create_text(
                        50,
                        list_y + 20 + i * 20,
                        text=f"{i+1}. {region['type']} - Frames: {region['start_frame']} to {region['end_frame']} - Confidence: {region['confidence']:.1%}",
                        fill="white",
                        anchor=tk.NW
                    )
            else:
                self.viz_canvas.create_text(
                    50,
                    list_y + 20,
                    text="No specific segments detected",
                    fill="white",
                    anchor=tk.NW
                )
    
    def generate_report(self):
        """Generate a report based on all results."""
        # Enable text widget for editing
        self.report_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.report_text.delete(1.0, tk.END)
        
        # Add report header
        self.report_text.insert(tk.END, "Deepfake Detection Analysis Report\n\n", "header")
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.report_text.insert(tk.END, f"Report generated: {timestamp}\n\n")
        
        # Add summary
        self.report_text.insert(tk.END, "Executive Summary\n", "subheader")
        
        total_count = len(self.current_results)
        fake_count = sum(1 for result in self.current_results.values() if result['is_fake'])
        real_count = total_count - fake_count
        
        if total_count > 0:
            fake_percent = (fake_count / total_count) * 100
            real_percent = (real_count / total_count) * 100
            
            self.report_text.insert(tk.END, f"A total of {total_count} files were analyzed using our Deepfake Detection Platform. ")
            
            if fake_count > 0:
                self.report_text.insert(tk.END, f"Of these files, {fake_count} ({fake_percent:.1f}%) were classified as likely manipulated. ")
            
            if real_count > 0:
                self.report_text.insert(tk.END, f"{real_count} ({real_percent:.1f}%) were classified as likely authentic. ")
            
            # Add overall assessment
            self.report_text.insert(tk.END, "\n\nOverall Assessment: ")
            
            if fake_count > real_count:
                self.report_text.insert(tk.END, "The majority of analyzed content appears to be manipulated, suggesting significant presence of deepfakes in the dataset.", "highlight")
            elif fake_count == real_count:
                self.report_text.insert(tk.END, "The dataset contains an equal mixture of authentic and manipulated content.", "highlight")
            else:
                self.report_text.insert(tk.END, "The majority of analyzed content appears to be authentic, with only limited presence of potential deepfakes.", "highlight")
        
        # Add detailed file results
        self.report_text.insert(tk.END, "\n\nDetailed Results\n", "subheader")
        
        # Group by media type
        images = [r for r in self.current_results.values() if r['media_type'] == 'image']
        audio = [r for r in self.current_results.values() if r['media_type'] == 'audio']
        videos = [r for r in self.current_results.values() if r['media_type'] == 'video']
        
        # Report on each type
        if images:
            self.report_text.insert(tk.END, "\nImage Analysis:\n")
            for i, result in enumerate(images):
                status = "FAKE/MANIPULATED" if result['is_fake'] else "AUTHENTIC"
                self.report_text.insert(tk.END, f"{i+1}. {result['filename']} - {status} - Confidence: {result['confidence']:.1%}\n")
        
        if audio:
            self.report_text.insert(tk.END, "\nAudio Analysis:\n")
            for i, result in enumerate(audio):
                status = "FAKE/MANIPULATED" if result['is_fake'] else "AUTHENTIC"
                self.report_text.insert(tk.END, f"{i+1}. {result['filename']} - {status} - Confidence: {result['confidence']:.1%}\n")
        
        if videos:
            self.report_text.insert(tk.END, "\nVideo Analysis:\n")
            for i, result in enumerate(videos):
                status = "FAKE/MANIPULATED" if result['is_fake'] else "AUTHENTIC"
                self.report_text.insert(tk.END, f"{i+1}. {result['filename']} - {status} - Confidence: {result['confidence']:.1%}\n")
        
        # Add methodology section
        self.report_text.insert(tk.END, "\n\nMethodology\n", "subheader")
        self.report_text.insert(tk.END, "This analysis was performed using the Deepfake Detection Platform, which employs state-of-the-art detection models:\n\n")
        self.report_text.insert(tk.END, "• Image Detection: Vision Transformer (ViT) model with face detection preprocessing and Error Level Analysis\n")
        self.report_text.insert(tk.END, "• Audio Detection: Wav2Vec2 model with spectrogram analysis\n")
        self.report_text.insert(tk.END, "• Video Detection: GenConViT/TimeSformer hybrid approach with frame and temporal analysis\n\n")
        self.report_text.insert(tk.END, "Each file was analyzed using multiple detection methods and ensemble techniques to provide comprehensive results.")
        
        # Add disclaimer
        self.report_text.insert(tk.END, "\n\nDisclaimer\n", "subheader")
        self.report_text.insert(tk.END, "This report is generated by an automated system and provides analysis based on detection models. While the platform utilizes advanced technology, no detection system is 100% accurate. Results should be verified by human experts for conclusive determination of content authenticity.")
        
        # Make read-only again
        self.report_text.config(state=tk.DISABLED)
    
    def export_pdf(self):
        """Export the report as a PDF."""
        # Placeholder - in real implementation, would use a PDF generation library
        messagebox.showinfo("Export PDF", "PDF export functionality not implemented in this version")
    
    def export_csv(self):
        """Export the results as a CSV file."""
        # Placeholder - in real implementation, would use the CSV module
        messagebox.showinfo("Export CSV", "CSV export functionality not implemented in this version")
    
    def show_settings(self):
        """Show settings dialog."""
        # Placeholder for settings dialog
        messagebox.showinfo("Settings", "Settings dialog not implemented in this version")
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
Deepfake Detection Platform Help

This application allows you to analyze images, audio, and video files for potential deepfakes.

Usage:
1. Select the appropriate tab for your media type (Image, Audio, or Video)
2. Click "Select Files" to choose files for analysis
3. Click "Analyze" to start the detection process
4. View results in the Results tab
5. Export reports as needed

For more detailed information, please refer to the documentation.
        """
        messagebox.showinfo("Help", help_text)

def main():
    """Run the application."""
    root = tk.Tk()
    app = DeepfakeDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()