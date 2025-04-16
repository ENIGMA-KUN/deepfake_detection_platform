"""
Report display component for the Deepfake Detection Platform.

This module provides components for displaying detection reports 
with a Tron Legacy-inspired design, including report templates,
formatting, and export options.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import time
import datetime
import os
import logging
from PIL import Image, ImageTk

# Import utilities from parent modules
from app.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger('report_display')

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

class ReportDisplay:
    """
    Component for displaying detection reports.
    
    This component provides a rich text display for viewing and interacting
    with detection reports.
    """
    
    def __init__(self, parent):
        """
        Initialize the report display.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Create header with title and export buttons
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_var = tk.StringVar(value="Detection Report")
        title_label = ttk.Label(
            header_frame,
            textvariable=self.title_var,
            style="Subheader.TLabel"
        )
        title_label.pack(side=tk.LEFT)
        
        # Export buttons
        export_frame = ttk.Frame(header_frame)
        export_frame.pack(side=tk.RIGHT)
        
        export_pdf_btn = ttk.Button(
            export_frame,
            text="Export PDF",
            command=self.export_pdf
        )
        export_pdf_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        export_html_btn = ttk.Button(
            export_frame,
            text="Export HTML",
            command=self.export_html
        )
        export_html_btn.pack(side=tk.LEFT)
        
        # Timestamp frame
        timestamp_frame = ttk.Frame(self.frame)
        timestamp_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.timestamp_var = tk.StringVar(value="Report generated: N/A")
        timestamp_label = ttk.Label(
            timestamp_frame,
            textvariable=self.timestamp_var,
            font=("Helvetica", 9)
        )
        timestamp_label.pack(side=tk.LEFT)
        
        # Create report text area with scrollbar
        text_frame = ttk.Frame(self.frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.report_text = tk.Text(
            text_frame,
            bg=TronTheme.DEEP_BLACK,
            fg="white",
            insertbackground=TronTheme.NEON_CYAN,
            bd=1,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1,
            font=("Helvetica", 10),
            wrap=tk.WORD,
            padx=10,
            pady=10,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar_y.config(command=self.report_text.yview)
        scrollbar_x.config(command=self.report_text.xview)
        
        # Configure text tags for styling
        self.configure_tags()
        
        # Initialize with empty report
        self.show_empty_report()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def configure_tags(self):
        """Configure text tags for report styling."""
        # Header tags
        self.report_text.tag_configure(
            "title",
            foreground=TronTheme.NEON_CYAN,
            font=("Helvetica", 16, "bold"),
            justify=tk.CENTER,
            spacing1=10,
            spacing3=10
        )
        
        self.report_text.tag_configure(
            "header1",
            foreground=TronTheme.NEON_CYAN,
            font=("Helvetica", 14, "bold"),
            spacing1=10,
            spacing3=5
        )
        
        self.report_text.tag_configure(
            "header2",
            foreground=TronTheme.SKY_BLUE,
            font=("Helvetica", 12, "bold"),
            spacing1=8,
            spacing3=4
        )
        
        self.report_text.tag_configure(
            "header3",
            foreground=TronTheme.POWDER_BLUE,
            font=("Helvetica", 11, "bold"),
            spacing1=6,
            spacing3=3
        )
        
        # Content styles
        self.report_text.tag_configure(
            "normal",
            foreground="white",
            font=("Helvetica", 10),
            spacing1=3,
            spacing3=3
        )
        
        self.report_text.tag_configure(
            "highlight",
            foreground=TronTheme.SKY_BLUE,
            font=("Helvetica", 10)
        )
        
        self.report_text.tag_configure(
            "warning",
            foreground=TronTheme.SUNRAY,
            font=("Helvetica", 10)
        )
        
        self.report_text.tag_configure(
            "danger",
            foreground=TronTheme.MAGENTA,
            font=("Helvetica", 10)
        )
        
        self.report_text.tag_configure(
            "success",
            foreground=TronTheme.NEON_CYAN,
            font=("Helvetica", 10)
        )
        
        # List item styles
        self.report_text.tag_configure(
            "bullet",
            lmargin1=20,
            lmargin2=40
        )
        
        self.report_text.tag_configure(
            "numbered",
            lmargin1=20,
            lmargin2=40
        )
        
        # Table styles
        self.report_text.tag_configure(
            "table_header",
            foreground=TronTheme.NEON_CYAN,
            font=("Helvetica", 10, "bold"),
            background=TronTheme.DEEP_BLACK,
            relief=tk.SOLID,
            borderwidth=1
        )
        
        self.report_text.tag_configure(
            "table_cell",
            foreground="white",
            background=TronTheme.DEEP_BLACK,
            relief=tk.SOLID,
            borderwidth=1
        )
        
        # Section styles
        self.report_text.tag_configure(
            "section",
            background="#101010",
            relief=tk.SOLID,
            borderwidth=1,
            lmargin1=10,
            lmargin2=10,
            rmargin=10
        )
    
    def show_empty_report(self):
        """Show empty report placeholder."""
        # Enable editing
        self.report_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.report_text.delete(1.0, tk.END)
        
        # Add empty report message
        self.report_text.insert(tk.END, "No Report Available\n\n", "title")
        self.report_text.insert(tk.END, "Run detection on media files to generate a detailed report. The report will include:\n\n", "normal")
        
        self.report_text.insert(tk.END, "• Analysis results and confidence scores\n", "bullet")
        self.report_text.insert(tk.END, "• Detected manipulation regions and types\n", "bullet")
        self.report_text.insert(tk.END, "• Technical details about detection methods\n", "bullet")
        self.report_text.insert(tk.END, "• Summary of findings and recommendations\n", "bullet")
        
        # Disable editing
        self.report_text.config(state=tk.DISABLED)
        
        # Reset title and timestamp
        self.title_var.set("Detection Report")
        self.timestamp_var.set("Report generated: N/A")
    
    def set_report_title(self, title):
        """
        Set the report title.
        
        Args:
            title: Report title
        """
        self.title_var.set(title)
    
    def update_timestamp(self):
        """Update the report timestamp to current time."""
        # Get current time
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Update timestamp text
        self.timestamp_var.set(f"Report generated: {timestamp}")
        
        return timestamp
    
    def display_report(self, results, title=None):
        """
        Display a detection report.
        
        Args:
            results: List of detection result dictionaries
            title: Report title (optional)
        """
        if not results:
            self.show_empty_report()
            return
        
        # Update title if provided
        if title:
            self.set_report_title(title)
        
        # Update timestamp
        timestamp = self.update_timestamp()
        
        # Enable editing
        self.report_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.report_text.delete(1.0, tk.END)
        
        # Add report title
        self.report_text.insert(tk.END, "Deepfake Detection Analysis Report\n", "title")
        
        # Add timestamp
        self.report_text.insert(tk.END, f"Report generated: {timestamp}\n\n", "normal")
        
        # Add executive summary
        self.add_executive_summary(results)
        
        # Add detailed results
        self.add_detailed_results(results)
        
        # Add methodology section
        self.add_methodology_section()
        
        # Add disclaimer
        self.add_disclaimer()
        
        # Disable editing
        self.report_text.config(state=tk.DISABLED)
    
    def add_executive_summary(self, results):
        """
        Add executive summary section to report.
        
        Args:
            results: List of detection result dictionaries
        """
        self.report_text.insert(tk.END, "Executive Summary\n", "header1")
        
        # Count results by type
        total_count = len(results)
        fake_count = sum(1 for result in results if result.get('is_fake', False))
        real_count = total_count - fake_count
        
        # Calculate percentages
        fake_percent = (fake_count / total_count) * 100 if total_count > 0 else 0
        real_percent = (real_count / total_count) * 100 if total_count > 0 else 0
        
        # Calculate average confidence
        avg_confidence = sum(result.get('confidence', 0) for result in results) / total_count if total_count > 0 else 0
        
        # Add summary text
        summary_text = f"A total of {total_count} media files were analyzed using the Deepfake Detection Platform. "
        
        if fake_count > 0:
            summary_text += f"Of these files, {fake_count} ({fake_percent:.1f}%) were classified as likely manipulated. "
        
        if real_count > 0:
            summary_text += f"{real_count} ({real_percent:.1f}%) were classified as likely authentic. "
        
        summary_text += f"The average confidence score across all analyses was {avg_confidence:.1%}."
        
        self.report_text.insert(tk.END, summary_text + "\n\n", "normal")
        
        # Add overall assessment
        self.report_text.insert(tk.END, "Overall Assessment: ", "highlight")
        
        if fake_count > real_count:
            assessment = "The majority of analyzed content appears to be manipulated, suggesting significant presence of deepfakes in the dataset."
            self.report_text.insert(tk.END, assessment + "\n\n", "danger")
        elif fake_count == real_count:
            assessment = "The dataset contains an equal mixture of authentic and manipulated content."
            self.report_text.insert(tk.END, assessment + "\n\n", "warning")
        else:
            assessment = "The majority of analyzed content appears to be authentic, with only limited presence of potential deepfakes."
            self.report_text.insert(tk.END, assessment + "\n\n", "success")
    
    def add_detailed_results(self, results):
        """
        Add detailed results section to report.
        
        Args:
            results: List of detection result dictionaries
        """
        self.report_text.insert(tk.END, "Detailed Results\n", "header1")
        
        # Group by media type
        images = [r for r in results if r.get('media_type', '') == 'image']
        audio = [r for r in results if r.get('media_type', '') == 'audio']
        videos = [r for r in results if r.get('media_type', '') == 'video']
        other = [r for r in results if r.get('media_type', '') not in ['image', 'audio', 'video']]
        
        # Add each media type section
        if images:
            self.add_media_type_section("Image Analysis", images)
        
        if audio:
            self.add_media_type_section("Audio Analysis", audio)
        
        if videos:
            self.add_media_type_section("Video Analysis", videos)
        
        if other:
            self.add_media_type_section("Other Media Analysis", other)
    
    def add_media_type_section(self, section_title, results):
        """
        Add a section for a specific media type.
        
        Args:
            section_title: Section title
            results: List of results for this media type
        """
        self.report_text.insert(tk.END, f"{section_title}\n", "header2")
        
        # Add table header
        self.report_text.insert(tk.END, "File Name", "table_header")
        self.report_text.insert(tk.END, " | ", "table_header")
        self.report_text.insert(tk.END, "Classification", "table_header")
        self.report_text.insert(tk.END, " | ", "table_header")
        self.report_text.insert(tk.END, "Confidence", "table_header")
        self.report_text.insert(tk.END, " | ", "table_header")
        self.report_text.insert(tk.END, "Detected Regions", "table_header")
        self.report_text.insert(tk.END, "\n", "normal")
        
        # Add table rows
        for result in results:
            # Get values
            filename = result.get('filename', 'Unknown')
            is_fake = result.get('is_fake', False)
            confidence = result.get('confidence', 0.0)
            regions = result.get('regions', [])
            
            # Format classification
            classification = "FAKE/MANIPULATED" if is_fake else "AUTHENTIC"
            class_tag = "danger" if is_fake else "success"
            
            # Format confidence
            conf_str = f"{confidence:.1%}"
            
            # Format regions
            if regions:
                region_types = set(region.get('type', 'unknown') for region in regions)
                regions_str = f"{len(regions)} ({', '.join(region_types)})"
            else:
                regions_str = "None"
            
            # Add table row
            self.report_text.insert(tk.END, filename, "table_cell")
            self.report_text.insert(tk.END, " | ", "table_cell")
            self.report_text.insert(tk.END, classification, class_tag)
            self.report_text.insert(tk.END, " | ", "table_cell")
            self.report_text.insert(tk.END, conf_str, "table_cell")
            self.report_text.insert(tk.END, " | ", "table_cell")
            self.report_text.insert(tk.END, regions_str, "table_cell")
            self.report_text.insert(tk.END, "\n", "normal")
        
        # Add extra space after table
        self.report_text.insert(tk.END, "\n", "normal")
        
        # Add detailed entries
        for i, result in enumerate(results):
            self.add_detail_entry(i + 1, result)
        
        # Add extra space after section
        self.report_text.insert(tk.END, "\n", "normal")
    
    def add_detail_entry(self, index, result):
        """
        Add a detailed entry for a result.
        
        Args:
            index: Entry index (for numbering)
            result: Result dictionary
        """
        # Get values
        filename = result.get('filename', 'Unknown')
        is_fake = result.get('is_fake', False)
        confidence = result.get('confidence', 0.0)
        regions = result.get('regions', [])
        details = result.get('details', 'No detailed analysis available.')
        
        # Add entry header
        self.report_text.insert(tk.END, f"{index}. {filename}\n", "header3")
        
        # Start section
        self.report_text.insert(tk.END, "Classification: ", "highlight")
        
        if is_fake:
            self.report_text.insert(tk.END, "FAKE/MANIPULATED", "danger")
        else:
            self.report_text.insert(tk.END, "AUTHENTIC", "success")
        
        self.report_text.insert(tk.END, f"  (Confidence: {confidence:.1%})\n", "normal")
        
        # Add details
        self.report_text.insert(tk.END, f"{details}\n", "normal")
        
        # Add regions if available
        if regions:
            self.report_text.insert(tk.END, "\nDetected Regions:\n", "highlight")
            
            for i, region in enumerate(regions):
                region_type = region.get('type', 'unknown')
                region_conf = region.get('confidence', 0.0)
                
                region_desc = f"  • {region_type.capitalize()} - Confidence: {region_conf:.1%}"
                
                # Add region-specific details based on media type
                media_type = result.get('media_type', '')
                
                if media_type == 'image':
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    width = region.get('width', 0)
                    height = region.get('height', 0)
                    region_desc += f" - Position: ({x:.2f}, {y:.2f}), Size: {width:.2f}x{height:.2f}"
                
                elif media_type == 'audio':
                    start_time = region.get('start_time', 0)
                    end_time = region.get('end_time', 0)
                    region_desc += f" - Time Range: {start_time:.2f}s - {end_time:.2f}s"
                
                elif media_type == 'video':
                    start_frame = region.get('start_frame', 0)
                    end_frame = region.get('end_frame', 0)
                    region_desc += f" - Frame Range: {start_frame} - {end_frame}"
                
                self.report_text.insert(tk.END, f"{region_desc}\n", "normal")
        
        # Add separator
        self.report_text.insert(tk.END, "\n", "normal")
    
    def add_methodology_section(self):
        """Add methodology section to report."""
        self.report_text.insert(tk.END, "Methodology\n", "header1")
        
        methodology_text = (
            "This analysis was performed using the Deepfake Detection Platform, which employs "
            "state-of-the-art detection models for different media types:\n\n"
        )
        
        self.report_text.insert(tk.END, methodology_text, "normal")
        
        # Add model descriptions
        self.report_text.insert(tk.END, "  • Image Detection: ", "bullet")
        self.report_text.insert(tk.END, "Vision Transformer (ViT) model with face detection preprocessing and Error Level Analysis\n", "normal")
        
        self.report_text.insert(tk.END, "  • Audio Detection: ", "bullet")
        self.report_text.insert(tk.END, "Wav2Vec2 model with spectrogram analysis\n", "normal")
        
        self.report_text.insert(tk.END, "  • Video Detection: ", "bullet")
        self.report_text.insert(tk.END, "GenConViT/TimeSformer hybrid approach with frame and temporal analysis\n\n", "normal")
        
        additional_text = (
            "Each file was analyzed using multiple detection methods and ensemble techniques to provide comprehensive results. "
            "The confidence scores represent the system's assessment of the likelihood that the media has been manipulated. "
            "Higher scores indicate greater confidence in the classification.\n\n"
        )
        
        self.report_text.insert(tk.END, additional_text, "normal")
    
    def add_disclaimer(self):
        """Add disclaimer section to report."""
        self.report_text.insert(tk.END, "Disclaimer\n", "header1")
        
        disclaimer_text = (
            "This report is generated by an automated system and provides analysis based on detection models. "
            "While the platform utilizes advanced technology, no detection system is 100% accurate. "
            "Results should be verified by human experts for conclusive determination of content authenticity. "
            "The system may produce false positives or false negatives in certain cases.\n\n"
            
            "The Deepfake Detection Platform is intended as a tool to assist in the identification of potentially "
            "manipulated media, but should not be used as the sole basis for making important decisions or drawing "
            "definitive conclusions about the authenticity of media content."
        )
        
        self.report_text.insert(tk.END, disclaimer_text, "normal")
    
    def export_pdf(self):
        """Export the report as a PDF file."""
        try:
            # Get save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Report as PDF"
            )
            
            if not file_path:
                return
            
            # Show message (in a real implementation, this would generate a PDF)
            import tkinter.messagebox as messagebox
            messagebox.showinfo(
                "Export PDF",
                "PDF export functionality would generate a PDF file here.\n\n"
                f"File path: {file_path}"
            )
            
            logger.info(f"Report exported as PDF: {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            
            import tkinter.messagebox as messagebox
            messagebox.showerror(
                "Export Error",
                f"Error exporting report as PDF: {str(e)}"
            )
    
    def export_html(self):
        """Export the report as an HTML file."""
        try:
            # Get save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
                title="Export Report as HTML"
            )
            
            if not file_path:
                return
            
            # Show message (in a real implementation, this would generate HTML)
            import tkinter.messagebox as messagebox
            messagebox.showinfo(
                "Export HTML",
                "HTML export functionality would generate an HTML file here.\n\n"
                f"File path: {file_path}"
            )
            
            logger.info(f"Report exported as HTML: {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            
            import tkinter.messagebox as messagebox
            messagebox.showerror(
                "Export Error",
                f"Error exporting report as HTML: {str(e)}"
            )