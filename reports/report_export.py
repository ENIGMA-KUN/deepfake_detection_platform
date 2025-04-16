"""
Report export functionality for the Deepfake Detection Platform.
Provides PDF and CSV export capabilities for detection reports.
"""

import os
import csv
import json
from typing import Dict, Any, List, Optional, Union
import datetime
from pathlib import Path

# For PDF generation we'll use WeasyPrint
from weasyprint import HTML, CSS

from detectors.detection_result import DetectionResult


class ReportExporter:
    """
    Handles exporting reports to different formats including PDF and CSV.
    """
    
    def __init__(self, output_dir: str = "reports/output"):
        """
        Initialize the report exporter.
        
        Args:
            output_dir: Directory to save exported reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_to_pdf(self, html_path: str, output_path: Optional[str] = None) -> str:
        """
        Export an HTML report to PDF format.
        
        Args:
            html_path: Path to the HTML report file
            output_path: Path to save the PDF file (optional)
            
        Returns:
            Path to the generated PDF file
        """
        # If output path is not provided, use the same name as HTML but with .pdf extension
        if output_path is None:
            output_path = os.path.splitext(html_path)[0] + '.pdf'
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate PDF from HTML
        html = HTML(filename=html_path)
        
        # Define CSS for the PDF rendering
        css = CSS(string='''
            @page {
                margin: 1cm;
                @bottom-center {
                    content: "Page " counter(page) " of " counter(pages);
                }
            }
            body {
                font-family: Arial, sans-serif;
            }
            .tron-grid {
                display: none; /* Hide the background grid in PDF */
            }
            /* Preserve the Tron theme colors but make them print-friendly */
            h1, h2, h3 {
                color: #0080a0;
            }
            .confidence-high {
                color: #006400;
            }
            .confidence-medium {
                color: #ff8c00;
            }
            .confidence-low {
                color: #8b0000;
            }
        ''')
        
        # Write the PDF to the output path
        html.write_pdf(output_path, stylesheets=[css])
        
        return output_path
    
    def export_to_csv(self, result: Union[DetectionResult, List[DetectionResult]], 
                      output_path: Optional[str] = None) -> str:
        """
        Export detection results to CSV format.
        
        Args:
            result: Detection result or list of results
            output_path: Path to save the CSV file (optional)
            
        Returns:
            Path to the generated CSV file
        """
        # Convert single result to list for consistent handling
        results = [result] if isinstance(result, DetectionResult) else result
        
        # If output path is not provided, generate a default one
        if output_path is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.output_dir, f'detection_results_{timestamp}.csv')
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine the media type to use appropriate CSV structure
        if not results:
            # Empty results, create a generic CSV
            self._export_empty_csv(output_path)
        elif all(r.media_type == 'image' for r in results):
            self._export_image_results_to_csv(results, output_path)
        elif all(r.media_type == 'audio' for r in results):
            self._export_audio_results_to_csv(results, output_path)
        elif all(r.media_type == 'video' for r in results):
            self._export_video_results_to_csv(results, output_path)
        else:
            # Mixed media types, export basic information only
            self._export_mixed_results_to_csv(results, output_path)
        
        return output_path
    
    def _export_empty_csv(self, output_path: str) -> None:
        """
        Export an empty CSV file with header columns.
        
        Args:
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Media Type', 'Is Deepfake', 'Confidence Score', 
                             'Detector Used', 'Detection Time'])
    
    def _export_mixed_results_to_csv(self, results: List[DetectionResult], output_path: str) -> None:
        """
        Export mixed media type results to CSV with basic information.
        
        Args:
            results: List of detection results
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Media Type', 'Is Deepfake', 'Confidence Score', 
                             'Detector Used', 'Detection Time'])
            
            for result in results:
                writer.writerow([
                    os.path.basename(result.file_path),
                    result.media_type,
                    'Yes' if result.is_deepfake else 'No',
                    f"{result.confidence_score * 100:.2f}%",
                    result.detector_name,
                    datetime.datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                ])
    
    def _export_image_results_to_csv(self, results: List[DetectionResult], output_path: str) -> None:
        """
        Export image detection results to CSV.
        
        Args:
            results: List of image detection results
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Is Deepfake', 'Confidence Score', 'Detector Used', 
                             'Detection Time', 'Face Count', 'Manipulation Regions'])
            
            for result in results:
                face_count = sum(1 for region in result.regions_of_interest 
                                if region.get('type') == 'face')
                manipulation_regions = sum(1 for region in result.regions_of_interest 
                                         if region.get('type') == 'manipulation')
                
                writer.writerow([
                    os.path.basename(result.file_path),
                    'Yes' if result.is_deepfake else 'No',
                    f"{result.confidence_score * 100:.2f}%",
                    result.detector_name,
                    datetime.datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    face_count,
                    manipulation_regions
                ])
    
    def _export_audio_results_to_csv(self, results: List[DetectionResult], output_path: str) -> None:
        """
        Export audio detection results to CSV.
        
        Args:
            results: List of audio detection results
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Is Deepfake', 'Confidence Score', 'Detector Used', 
                             'Detection Time', 'Duration (s)', 'Sample Rate', 'Temporal Anomalies'])
            
            for result in results:
                duration = result.metadata.get('duration', 0)
                sample_rate = result.metadata.get('sample_rate', 0)
                temporal_anomalies = len(result.analysis_details.get('temporal_anomalies', []))
                
                writer.writerow([
                    os.path.basename(result.file_path),
                    'Yes' if result.is_deepfake else 'No',
                    f"{result.confidence_score * 100:.2f}%",
                    result.detector_name,
                    datetime.datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    f"{duration:.2f}",
                    sample_rate,
                    temporal_anomalies
                ])
    
    def _export_video_results_to_csv(self, results: List[DetectionResult], output_path: str) -> None:
        """
        Export video detection results to CSV.
        
        Args:
            results: List of video detection results
            output_path: Path to save the CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'Is Deepfake', 'Confidence Score', 'Detector Used', 
                             'Detection Time', 'Duration (s)', 'Frame Count', 'FPS',
                             'Manipulated Frames', 'Audio-Video Sync Issues'])
            
            for result in results:
                duration = result.metadata.get('duration', 0)
                frame_count = result.metadata.get('frame_count', 0)
                fps = result.metadata.get('fps', 0)
                
                # Count manipulated frames
                frame_analysis = result.analysis_details.get('frame_analysis', [])
                manipulated_frames = sum(1 for frame in frame_analysis 
                                       if frame.get('confidence', 0) > 0.5)
                
                # Check for audio-video sync issues
                sync_analysis = result.analysis_details.get('sync_analysis', {})
                has_sync_issues = sync_analysis.get('has_issues', False)
                
                writer.writerow([
                    os.path.basename(result.file_path),
                    'Yes' if result.is_deepfake else 'No',
                    f"{result.confidence_score * 100:.2f}%",
                    result.detector_name,
                    datetime.datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    f"{duration:.2f}",
                    frame_count,
                    fps,
                    manipulated_frames,
                    'Yes' if has_sync_issues else 'No'
                ])