#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Result handler module for the Deepfake Detection Platform.
Manages the storage, retrieval, and formatting of detection results.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from detectors.detection_result import DetectionResult

# Configure logger
logger = logging.getLogger(__name__)


class ResultFormat:
    """Supported result formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"


class ResultHandler:
    """Handler for detection results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the result handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get result directory from config
        self.result_dir = config.get("reports", {}).get("save_directory", "./reports/output")
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Get report template directory from config
        self.template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports", "templates"
        )
        
        # Get supported formats
        self.supported_formats = config.get("reports", {}).get(
            "export_formats", [ResultFormat.JSON, ResultFormat.HTML]
        )
    
    def save_result(
        self,
        result: DetectionResult,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Save a detection result in the specified formats.
        
        Args:
            result: Detection result to save
            formats: List of formats to save (or None for all supported)
            
        Returns:
            Dictionary mapping format to output path
        """
        # Default to all supported formats
        if formats is None:
            formats = self.supported_formats
        
        # Ensure result ID is set
        if not hasattr(result, 'id') or not result.id:
            result.id = str(uuid.uuid4())
        
        # Create output paths
        output_paths = {}
        
        # Save in each format
        for fmt in formats:
            if fmt == ResultFormat.JSON:
                path = self._save_json(result)
                if path:
                    output_paths[ResultFormat.JSON] = path
            
            elif fmt == ResultFormat.HTML:
                path = self._save_html(result)
                if path:
                    output_paths[ResultFormat.HTML] = path
            
            elif fmt == ResultFormat.PDF:
                path = self._save_pdf(result)
                if path:
                    output_paths[ResultFormat.PDF] = path
            
            elif fmt == ResultFormat.CSV:
                path = self._save_csv(result)
                if path:
                    output_paths[ResultFormat.CSV] = path
            
            else:
                logger.warning(f"Unsupported format: {fmt}")
        
        return output_paths
    
    def load_result(self, result_id: str) -> Optional[DetectionResult]:
        """Load a detection result from storage.
        
        Args:
            result_id: ID of the result to load
            
        Returns:
            Loaded DetectionResult if found, None otherwise
        """
        # Try to load from JSON (primary storage format)
        json_path = os.path.join(self.result_dir, f"{result_id}.json")
        
        if not os.path.exists(json_path):
            logger.warning(f"Result not found: {result_id}")
            return None
        
        try:
            return DetectionResult.load_from_file(json_path)
        except Exception as e:
            logger.error(f"Error loading result {result_id}: {e}")
            return None
    
    def delete_result(self, result_id: str) -> bool:
        """Delete a detection result and associated files.
        
        Args:
            result_id: ID of the result to delete
            
        Returns:
            True if deleted, False otherwise
        """
        deleted = False
        
        # Delete each format
        for fmt in [ResultFormat.JSON, ResultFormat.HTML, ResultFormat.PDF, ResultFormat.CSV]:
            ext = fmt.lower()
            path = os.path.join(self.result_dir, f"{result_id}.{ext}")
            
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted = True
                except Exception as e:
                    logger.error(f"Error deleting {path}: {e}")
        
        return deleted
    
    def list_results(
        self,
        max_age: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """List available result IDs.
        
        Args:
            max_age: Maximum age in seconds (or None for no limit)
            limit: Maximum number of results to return (or None for no limit)
            
        Returns:
            List of result IDs
        """
        results = []
        
        # List all JSON files (our primary storage format)
        for filename in os.listdir(self.result_dir):
            if filename.endswith(".json"):
                # Extract result ID from filename
                result_id = os.path.splitext(filename)[0]
                
                # Check age if specified
                if max_age is not None:
                    path = os.path.join(self.result_dir, filename)
                    mtime = os.path.getmtime(path)
                    age = time.time() - mtime
                    
                    if age > max_age:
                        continue
                
                results.append(result_id)
        
        # Sort by modification time (newest first)
        results.sort(
            key=lambda rid: os.path.getmtime(os.path.join(self.result_dir, f"{rid}.json")),
            reverse=True
        )
        
        # Apply limit if specified
        if limit is not None:
            results = results[:limit]
        
        return results
    
    def get_result_info(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a result without loading the full data.
        
        Args:
            result_id: ID of the result
            
        Returns:
            Dictionary with result info if found, None otherwise
        """
        # Check if result exists
        json_path = os.path.join(self.result_dir, f"{result_id}.json")
        
        if not os.path.exists(json_path):
            return None
        
        try:
            # Get file info
            stat = os.stat(json_path)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            
            # Load basic info from JSON without parsing the whole file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic info
            info = {
                "id": result_id,
                "filename": data.get("filename", ""),
                "media_type": data.get("media_type", ""),
                "is_deepfake": data.get("is_deepfake", False),
                "confidence_score": data.get("confidence_score", 0.0),
                "timestamp": data.get("timestamp", ""),
                "status": data.get("status", ""),
                "file_mtime": mtime.isoformat(),
                "file_size": size,
                "available_formats": self._get_available_formats(result_id)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting info for result {result_id}: {e}")
            return None
    
    def _get_available_formats(self, result_id: str) -> List[str]:
        """Get available formats for a result.
        
        Args:
            result_id: ID of the result
            
        Returns:
            List of available format strings
        """
        formats = []
        
        for fmt in [ResultFormat.JSON, ResultFormat.HTML, ResultFormat.PDF, ResultFormat.CSV]:
            ext = fmt.lower()
            path = os.path.join(self.result_dir, f"{result_id}.{ext}")
            
            if os.path.exists(path):
                formats.append(fmt)
        
        return formats
    
    def _save_json(self, result: DetectionResult) -> Optional[str]:
        """Save result as JSON.
        
        Args:
            result: Detection result to save
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            # Create output path
            output_path = os.path.join(self.result_dir, f"{result.id}.json")
            
            # Save result
            filepath = result.save_to_file(self.result_dir)
            
            logger.info(f"Saved result as JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving result as JSON: {e}")
            return None
    
    def _save_html(self, result: DetectionResult) -> Optional[str]:
        """Save result as HTML report.
        
        Args:
            result: Detection result to save
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Get template
            template_file = "detailed_report.html"
            template_path = os.path.join(self.template_dir, template_file)
            
            if not os.path.exists(template_path):
                logger.warning(f"HTML template not found: {template_path}")
                return None
            
            # Create Jinja environment
            env = Environment(loader=FileSystemLoader(self.template_dir))
            template = env.get_template(template_file)
            
            # Create output path
            output_path = os.path.join(self.result_dir, f"{result.id}.html")
            
            # Prepare data for template
            data = result.to_dict()
            data["report_time"] = datetime.now().isoformat()
            
            # Render template
            html = template.render(result=data)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Saved result as HTML: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("jinja2 not available, cannot save as HTML")
            return None
            
        except Exception as e:
            logger.error(f"Error saving result as HTML: {e}")
            return None
    
    def _save_pdf(self, result: DetectionResult) -> Optional[str]:
        """Save result as PDF report.
        
        Args:
            result: Detection result to save
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            import pdfkit
            
            # First save as HTML
            html_path = self._save_html(result)
            
            if html_path is None:
                logger.warning("Could not create HTML for PDF conversion")
                return None
            
            # Create output path
            output_path = os.path.join(self.result_dir, f"{result.id}.pdf")
            
            # Convert HTML to PDF
            pdfkit.from_file(html_path, output_path)
            
            logger.info(f"Saved result as PDF: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("pdfkit not available, cannot save as PDF")
            return None
            
        except Exception as e:
            logger.error(f"Error saving result as PDF: {e}")
            return None
    
    def _save_csv(self, result: DetectionResult) -> Optional[str]:
        """Save result as CSV.
        
        Args:
            result: Detection result to save
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            import csv
            
            # Create output path
            output_path = os.path.join(self.result_dir, f"{result.id}.csv")
            
            # Convert result to flat structure suitable for CSV
            data = self._flatten_result(result)
            
            # Write CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(data.keys())
                
                # Write values
                writer.writerow(data.values())
            
            logger.info(f"Saved result as CSV: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("csv module not available, cannot save as CSV")
            return None
            
        except Exception as e:
            logger.error(f"Error saving result as CSV: {e}")
            return None
    
    def _flatten_result(self, result: DetectionResult) -> Dict[str, Any]:
        """Flatten a result into a dictionary suitable for CSV.
        
        Args:
            result: Detection result to flatten
            
        Returns:
            Flattened dictionary
        """
        # Convert to dictionary
        data = result.to_dict()
        
        # Create flattened dictionary
        flat = {
            "id": data.get("id", ""),
            "media_type": data.get("media_type", ""),
            "filename": data.get("filename", ""),
            "timestamp": data.get("timestamp", ""),
            "is_deepfake": data.get("is_deepfake", False),
            "confidence_score": data.get("confidence_score", 0.0),
            "status": data.get("status", ""),
            "model_used": data.get("model_used", ""),
            "execution_time": data.get("execution_time", 0.0),
            "num_regions": len(data.get("regions", [])),
            "num_time_segments": len(data.get("time_segments", [])),
            "categories": ",".join(data.get("categories", [])),
        }
        
        # Add some metadata if available
        metadata = data.get("metadata", {})
        for key in ["duration", "fps", "width", "height", "sample_rate"]:
            if key in metadata:
                flat[f"metadata_{key}"] = metadata[key]
        
        return flat


class ReportGenerator:
    """Generator for detailed reports and visualizations."""
    
    def __init__(self, config: Dict[str, Any], result_handler: ResultHandler):
        """Initialize the report generator.
        
        Args:
            config: Configuration dictionary
            result_handler: ResultHandler instance
        """
        self.config = config
        self.result_handler = result_handler
        
        # Create reports directory
        self.reports_dir = config.get("reports", {}).get("save_directory", "./reports/output")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Get visualization settings
        self.include_visualization = config.get("reports", {}).get("include_visualization", True)
    
    def generate_report(
        self,
        result: Union[DetectionResult, str],
        output_dir: Optional[str] = None,
        include_visualization: Optional[bool] = None
    ) -> Dict[str, str]:
        """Generate a comprehensive report for a detection result.
        
        Args:
            result: DetectionResult or result ID
            output_dir: Output directory (or None to use default)
            include_visualization: Whether to include visualizations (or None to use default)
            
        Returns:
            Dictionary mapping output types to file paths
        """
        # Load result if string ID provided
        if isinstance(result, str):
            loaded_result = self.result_handler.load_result(result)
            if loaded_result is None:
                logger.error(f"Could not load result: {result}")
                return {}
            result = loaded_result
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.reports_dir, result.id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine whether to include visualizations
        if include_visualization is None:
            include_visualization = self.include_visualization
        
        # Generate files
        output_files = {}
        
        # Save result in all formats
        result_files = self.result_handler.save_result(result)
        output_files.update(result_files)
        
        # Create summary file
        summary_path = os.path.join(output_dir, "summary.txt")
        self._create_summary_file(result, summary_path)
        output_files["summary"] = summary_path
        
        # Generate visualizations if requested
        if include_visualization:
            # Generate appropriate visualizations based on media type
            if result.media_type.value == "image":
                self._generate_image_visualizations(result, output_dir, output_files)
            elif result.media_type.value == "audio":
                self._generate_audio_visualizations(result, output_dir, output_files)
            elif result.media_type.value == "video":
                self._generate_video_visualizations(result, output_dir, output_files)
        
        return output_files
    
    def _create_summary_file(self, result: DetectionResult, output_path: str) -> None:
        """Create a summary text file.
        
        Args:
            result: Detection result
            output_path: Output file path
        """
        try:
            # Convert to dictionary
            data = result.to_dict()
            
            # Create summary lines
            lines = [
                "DEEPFAKE DETECTION SUMMARY",
                "=" * 50,
                f"ID: {data.get('id', '')}",
                f"Filename: {data.get('filename', '')}",
                f"Media Type: {data.get('media_type', '')}",
                f"Timestamp: {data.get('timestamp', '')}",
                f"Result: {'DEEPFAKE DETECTED' if data.get('is_deepfake', False) else 'LIKELY AUTHENTIC'}",
                f"Confidence: {data.get('confidence_score', 0.0):.2f}",
                f"Model: {data.get('model_used', '')}",
                f"Execution Time: {data.get('execution_time', 0.0):.2f} seconds",
                "=" * 50
            ]
            
            # Add categories if available
            categories = data.get("categories", [])
            if categories:
                lines.append("Detected Categories:")
                for category in categories:
                    lines.append(f"- {category}")
                lines.append("=" * 50)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Created summary file: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary file: {e}")
    
    def _generate_image_visualizations(
        self,
        result: DetectionResult,
        output_dir: str,
        output_files: Dict[str, str]
    ) -> None:
        """Generate visualizations for image detection results.
        
        Args:
            result: Detection result
            output_dir: Output directory
            output_files: Dictionary to update with output files
        """
        # Import visualization module
        try:
            from detectors.image_detector.visualization import (create_comparison_grid,
                                                              visualize_detection_result)
            
            # We need the media file path
            media_path = self._get_media_path(result)
            
            if media_path:
                # Create visualizations
                vis_path = os.path.join(output_dir, "visualization.png")
                grid_path = os.path.join(output_dir, "comparison_grid.png")
                
                # Generate standard visualization
                visualize_detection_result(
                    media_path, result, vis_path,
                    show_heatmap=True, show_regions=True
                )
                output_files["visualization"] = vis_path
                
                # Generate comparison grid
                create_comparison_grid(
                    media_path, result, grid_path
                )
                output_files["comparison_grid"] = grid_path
                
        except ImportError as e:
            logger.warning(f"Image visualization module not available: {e}")
            
        except Exception as e:
            logger.error(f"Error generating image visualizations: {e}")
    
    def _generate_audio_visualizations(
        self,
        result: DetectionResult,
        output_dir: str,
        output_files: Dict[str, str]
    ) -> None:
        """Generate visualizations for audio detection results.
        
        Args:
            result: Detection result
            output_dir: Output directory
            output_files: Dictionary to update with output files
        """
        # Import visualization module
        try:
            from detectors.audio_detector.visualization import (create_audio_analysis_grid,
                                                              visualize_audio_detection)
            
            # We need the media file path
            media_path = self._get_media_path(result)
            
            if media_path:
                # Create visualizations
                vis_path = os.path.join(output_dir, "visualization.png")
                grid_path = os.path.join(output_dir, "analysis_grid.png")
                
                # Generate standard visualization
                visualize_audio_detection(
                    media_path, result, vis_path,
                    show_waveform=True, show_spectrogram=True
                )
                output_files["visualization"] = vis_path
                
                # Generate analysis grid
                create_audio_analysis_grid(
                    media_path, result, grid_path
                )
                output_files["analysis_grid"] = grid_path
                
        except ImportError as e:
            logger.warning(f"Audio visualization module not available: {e}")
            
        except Exception as e:
            logger.error(f"Error generating audio visualizations: {e}")
    
    def _generate_video_visualizations(
        self,
        result: DetectionResult,
        output_dir: str,
        output_files: Dict[str, str]
    ) -> None:
        """Generate visualizations for video detection results.
        
        Args:
            result: Detection result
            output_dir: Output directory
            output_files: Dictionary to update with output files
        """
        # Import visualization module
        try:
            from detectors.video_detector.visualization import (create_video_report,
                                                              create_video_summary)
            
            # We need the media file path
            media_path = self._get_media_path(result)
            
            if media_path:
                # Create visualizations
                summary_path = os.path.join(output_dir, "video_summary.png")
                
                # Generate video summary
                create_video_summary(
                    media_path, result, summary_path,
                    num_frames=6, include_timeline=True
                )
                output_files["video_summary"] = summary_path
                
                # Generate video report with frame samples
                report_files = create_video_report(
                    media_path, result, output_dir,
                    create_frame_samples=True, num_sample_frames=6
                )
                
                # Add report files to output
                for key, path in report_files.items():
                    output_files[f"video_report_{key}"] = path
                
        except ImportError as e:
            logger.warning(f"Video visualization module not available: {e}")
            
        except Exception as e:
            logger.error(f"Error generating video visualizations: {e}")
    
    def _get_media_path(self, result: DetectionResult) -> Optional[str]:
        """Try to determine media file path from result.
        
        Args:
            result: Detection result
            
        Returns:
            Media path if found, None otherwise
        """
        # Check metadata for original_path
        if hasattr(result, 'metadata') and result.metadata:
            if 'original_path' in result.metadata:
                path = result.metadata['original_path']
                if os.path.exists(path):
                    return path
        
        # Check common locations based on filename
        if hasattr(result, 'filename') and result.filename:
            # Try temp directory
            temp_dir = self.config.get("system", {}).get("temp_directory", "./temp")
            path = os.path.join(temp_dir, result.filename)
            if os.path.exists(path):
                return path
            
            # Try uploads directory
            upload_dir = os.path.join(os.path.dirname(self.reports_dir), "uploads")
            path = os.path.join(upload_dir, result.filename)
            if os.path.exists(path):
                return path
        
        return None