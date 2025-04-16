#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for video deepfake detection results.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.detection_result import (DetectionResult, Region, TimeSegment,
                                       VideoFeatures)
from detectors.detector_utils import extract_frames

# Configure logger
logger = logging.getLogger(__name__)


def create_video_summary(
    video_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    num_frames: int = 4,
    include_timeline: bool = True,
    output_format: str = 'png',
    dpi: int = 150
) -> Optional[np.ndarray]:
    """Create a summary visualization for video detection results.
    
    Args:
        video_path: Path to the original video file
        result: Detection result to visualize
        output_path: Path to save the visualization (None = don't save)
        num_frames: Number of sample frames to include
        include_timeline: Whether to include timeline visualization
        output_format: Image format for saving
        dpi: DPI for output image
        
    Returns:
        Visualization as numpy array if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create figure with grid layout
        if include_timeline:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(3, num_frames, height_ratios=[3, 3, 1])
        else:
            fig = plt.figure(figsize=(15, 8))
            gs = GridSpec(2, num_frames)
        
        # Extract sample frames
        if result.video_features and result.video_features.frame_anomalies:
            # Prioritize frames with detected anomalies
            anomaly_frames = sorted(result.video_features.frame_anomalies.keys())
            
            if len(anomaly_frames) >= num_frames:
                # Take evenly spaced anomaly frames
                step = len(anomaly_frames) // num_frames
                sample_indices = [anomaly_frames[i * step] for i in range(num_frames)]
            else:
                # Take all anomaly frames and add normal frames
                sample_indices = anomaly_frames.copy()
                
                # Add evenly spaced normal frames
                remaining = num_frames - len(sample_indices)
                if remaining > 0 and total_frames > 0:
                    normal_step = total_frames // (remaining + 1)
                    for i in range(1, remaining + 1):
                        normal_idx = i * normal_step
                        # Check if not too close to anomaly frames
                        if all(abs(normal_idx - idx) > 10 for idx in sample_indices):
                            sample_indices.append(normal_idx)
                    
                    # Sort indices
                    sample_indices.sort()
                    
                    # Ensure we have exactly num_frames
                    if len(sample_indices) > num_frames:
                        sample_indices = sample_indices[:num_frames]
                    elif len(sample_indices) < num_frames:
                        # Duplicate last index if needed
                        sample_indices.extend([sample_indices[-1]] * (num_frames - len(sample_indices)))
        else:
            # Evenly space frames across video
            if total_frames > 0:
                step = total_frames // (num_frames + 1)
                sample_indices = [(i + 1) * step for i in range(num_frames)]
            else:
                sample_indices = [0] * num_frames
        
        # Extract frames
        frames = extract_frames(video_path, frame_indices=sample_indices)
        
        # Create visualization for each frame
        for i, frame_idx in enumerate(sorted(frames.keys())):
            frame = frames[frame_idx]
            
            # Check if this frame has anomalies
            has_anomalies = False
            frame_regions = []
            
            if result.video_features and result.video_features.frame_anomalies:
                if frame_idx in result.video_features.frame_anomalies:
                    has_anomalies = True
                    frame_regions = result.video_features.frame_anomalies[frame_idx]
            
            # Add original frame
            ax_orig = fig.add_subplot(gs[0, i])
            ax_orig.imshow(frame)
            ax_orig.set_title(f"Frame {frame_idx}")
            ax_orig.axis('off')
            
            # Add annotated frame
            ax_annotated = fig.add_subplot(gs[1, i])
            annotated_frame = _annotate_frame(frame, frame_regions)
            ax_annotated.imshow(annotated_frame)
            ax_annotated.set_title("Detected Anomalies" if has_anomalies else "No Anomalies")
            ax_annotated.axis('off')
        
        # Add timeline if requested
        if include_timeline:
            ax_timeline = fig.add_subplot(gs[2, :])
            _create_timeline(ax_timeline, result, duration)
        
        # Add overall title
        if result.is_deepfake:
            title = f"DEEPFAKE DETECTED (Confidence: {result.confidence_score:.2f})"
            title_color = 'red'
        else:
            title = f"LIKELY AUTHENTIC (Confidence: {result.confidence_score:.2f})"
            title_color = 'green'
        
        fig.suptitle(title, fontsize=16, color=title_color)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, format=output_format, dpi=dpi)
            logger.info(f"Video summary saved to {output_path}")
        
        # Get image data from figure
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error creating video summary: {e}")
        return None


def _annotate_frame(
    frame: np.ndarray,
    regions: List[Region],
    outline_width: int = 2
) -> np.ndarray:
    """Annotate frame with detected regions.
    
    Args:
        frame: Original frame as numpy array
        regions: List of detected regions
        outline_width: Width of region outlines
        
    Returns:
        Annotated frame as numpy array
    """
    import cv2
    
    # Create a copy of the frame
    annotated = frame.copy()
    
    # Define colors for different confidence levels
    colors = {
        "high": (255, 0, 0),    # Red (BGR ordering for OpenCV)
        "medium": (0, 165, 255),  # Orange
        "low": (0, 255, 255)    # Yellow
    }
    
    # Draw regions
    height, width = frame.shape[:2]
    
    for region in regions:
        # Determine color based on confidence
        if region.confidence >= 0.7:
            color = colors["high"]
        elif region.confidence >= 0.5:
            color = colors["medium"]
        else:
            color = colors["low"]
        
        # Convert normalized coordinates to pixels
        x = int(region.x * width)
        y = int(region.y * height)
        w = int(region.width * width)
        h = int(region.height * height)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        
        # Draw rectangle outline
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, outline_width)
        
        # Draw confidence label
        label = f"{region.confidence:.2f}"
        cv2.putText(
            annotated, label, (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    
    return annotated


def _create_timeline(
    ax: Any,
    result: DetectionResult,
    duration: float
) -> None:
    """Create timeline visualization of detection results.
    
    Args:
        ax: Matplotlib axis
        result: Detection result
        duration: Video duration in seconds
    """
    import matplotlib.patches as patches
    
    # Set axis limits
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 3)
    
    # Add labels
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Temporal", "Audio-Video Sync", "Overall"])
    
    # Add baseline
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=2.5, color='gray', linestyle='-', alpha=0.3)
    
    # Add temporal segments
    if hasattr(result, 'time_segments') and result.time_segments:
        for segment in result.time_segments:
            # Choose color based on confidence
            if segment.confidence >= 0.7:
                color = 'red'
            elif segment.confidence >= 0.5:
                color = 'orange'
            else:
                color = 'yellow'
            
            # Add rectangle
            rect = patches.Rectangle(
                (segment.start_time, 0.25), 
                segment.end_time - segment.start_time, 0.5,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
    
    # Add sync issues
    if result.video_features and result.video_features.sync_issues:
        for issue in result.video_features.sync_issues:
            # Choose color based on confidence
            if issue.confidence >= 0.7:
                color = 'red'
            elif issue.confidence >= 0.5:
                color = 'orange'
            else:
                color = 'yellow'
            
            # Add rectangle
            rect = patches.Rectangle(
                (issue.start_time, 1.25), 
                issue.end_time - issue.start_time, 0.5,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
    
    # Add overall confidence indicator
    if result.is_deepfake:
        # Red bar with length proportional to confidence
        rect = patches.Rectangle(
            (0, 2.25), 
            duration * result.confidence_score, 0.5,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.7
        )
        ax.add_patch(rect)
    
    # Add grid
    ax.grid(True, alpha=0.3)


def create_video_report(
    video_path: str,
    result: DetectionResult,
    output_dir: str,
    create_frame_samples: bool = True,
    num_sample_frames: int = 6,
    output_format: str = 'png',
    dpi: int = 150
) -> Dict[str, str]:
    """Create a comprehensive report for video detection results.
    
    Args:
        video_path: Path to the original video file
        result: Detection result to visualize
        output_dir: Directory to save report files
        create_frame_samples: Whether to create sample frame images
        num_sample_frames: Number of sample frames to create
        output_format: Image format for saving
        dpi: DPI for output images
        
    Returns:
        Dictionary mapping file types to output paths
    """
    report_files = {}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary visualization
        summary_path = os.path.join(output_dir, "summary." + output_format)
        create_video_summary(
            video_path, result, summary_path, 
            num_frames=4, include_timeline=True,
            output_format=output_format, dpi=dpi
        )
        report_files["summary"] = summary_path
        
        # Create timeline visualization
        timeline_path = os.path.join(output_dir, "timeline." + output_format)
        _create_standalone_timeline(
            result, timeline_path, 
            output_format=output_format, dpi=dpi
        )
        report_files["timeline"] = timeline_path
        
        # Create frame samples if requested
        if create_frame_samples:
            # Create frame samples directory
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Create frame samples
            frame_paths = _create_frame_samples(
                video_path, result, frames_dir, 
                num_frames=num_sample_frames,
                output_format=output_format
            )
            
            report_files["frames"] = frame_paths
        
        # Create text report
        report_path = os.path.join(output_dir, "report.txt")
        _create_text_report(video_path, result, report_path)
        report_files["text"] = report_path
        
        return report_files
        
    except Exception as e:
        logger.error(f"Error creating video report: {e}")
        return report_files


def _create_standalone_timeline(
    result: DetectionResult,
    output_path: Optional[str] = None,
    output_format: str = 'png',
    dpi: int = 100
) -> Optional[np.ndarray]:
    """Create a standalone timeline visualization.
    
    Args:
        result: Detection result to visualize
        output_path: Path to save the visualization (None = don't save)
        output_format: Image format for saving
        dpi: DPI for output image
        
    Returns:
        Visualization as numpy array if successful, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        # Get video duration from metadata or use default
        duration = result.metadata.get("duration", 30.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create timeline
        _create_timeline(ax, result, duration)
        
        # Add title
        if result.is_deepfake:
            title = f"Deepfake Detection Timeline (Confidence: {result.confidence_score:.2f})"
            title_color = 'red'
        else:
            title = f"Video Analysis Timeline (Confidence: {result.confidence_score:.2f})"
            title_color = 'green'
        
        fig.suptitle(title, fontsize=14, color=title_color)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, format=output_format, dpi=dpi)
            logger.info(f"Timeline saved to {output_path}")
        
        # Get image data from figure
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close figure to free memory
        plt.close(fig)
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error creating standalone timeline: {e}")
        return None


def _create_frame_samples(
    video_path: str,
    result: DetectionResult,
    output_dir: str,
    num_frames: int = 6,
    output_format: str = 'png'
) -> Dict[int, str]:
    """Create annotated frame samples from the video.
    
    Args:
        video_path: Path to the original video file
        result: Detection result
        output_dir: Directory to save frame images
        num_frames: Number of sample frames to create
        output_format: Image format for saving
        
    Returns:
        Dictionary mapping frame indices to output paths
    """
    import cv2
    
    frame_paths = {}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract sample frames
        if result.video_features and result.video_features.frame_anomalies:
            # Prioritize frames with detected anomalies
            anomaly_frames = sorted(result.video_features.frame_anomalies.keys())
            
            if len(anomaly_frames) >= num_frames:
                # Take evenly spaced anomaly frames
                step = len(anomaly_frames) // num_frames
                sample_indices = [anomaly_frames[i * step] for i in range(num_frames)]
            else:
                # Take all anomaly frames
                sample_indices = anomaly_frames.copy()
                
                # Get video properties
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Add evenly spaced normal frames
                remaining = num_frames - len(sample_indices)
                if remaining > 0 and total_frames > 0:
                    normal_step = total_frames // (remaining + 1)
                    for i in range(1, remaining + 1):
                        normal_idx = i * normal_step
                        # Check if not too close to anomaly frames
                        if all(abs(normal_idx - idx) > 10 for idx in sample_indices):
                            sample_indices.append(normal_idx)
                    
                    # Sort indices
                    sample_indices.sort()
                    
                    # Ensure we have exactly num_frames
                    if len(sample_indices) > num_frames:
                        sample_indices = sample_indices[:num_frames]
                    elif len(sample_indices) < num_frames:
                        # Duplicate last index if needed
                        sample_indices.extend([sample_indices[-1]] * (num_frames - len(sample_indices)))
        else:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Evenly space frames across video
            if total_frames > 0:
                step = total_frames // (num_frames + 1)
                sample_indices = [(i + 1) * step for i in range(num_frames)]
            else:
                sample_indices = [0] * num_frames
        
        # Extract frames
        frames = extract_frames(video_path, frame_indices=sample_indices)
        
        # Process each frame
        for frame_idx, frame in frames.items():
            # Check if this frame has anomalies
            frame_regions = []
            
            if result.video_features and result.video_features.frame_anomalies:
                if frame_idx in result.video_features.frame_anomalies:
                    frame_regions = result.video_features.frame_anomalies[frame_idx]
            
            # Annotate frame
            annotated_frame = _annotate_frame(frame, frame_regions)
            
            # Convert RGB to BGR for OpenCV
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx}.{output_format}")
            cv2.imwrite(frame_path, annotated_frame_bgr)
            
            frame_paths[frame_idx] = frame_path
        
        return frame_paths
        
    except Exception as e:
        logger.error(f"Error creating frame samples: {e}")
        return frame_paths


def _create_text_report(
    video_path: str,
    result: DetectionResult,
    output_path: str
) -> None:
    """Create a text report for video detection results.
    
    Args:
        video_path: Path to the original video file
        result: Detection result
        output_path: Path to save the text report
    """
    try:
        # Get video properties
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create report text
        report_lines = [
            "DEEPFAKE DETECTION REPORT",
            "=" * 50,
            f"Filename: {os.path.basename(video_path)}",
            f"Resolution: {width}x{height}",
            f"Duration: {duration:.2f} seconds",
            f"FPS: {fps:.2f}",
            f"Total Frames: {total_frames}",
            "=" * 50,
            f"DETECTION RESULT: {'DEEPFAKE DETECTED' if result.is_deepfake else 'LIKELY AUTHENTIC'}",
            f"Confidence Score: {result.confidence_score:.4f}",
            "=" * 50
        ]
        
        # Add detailed analysis
        report_lines.extend([
            "DETAILED ANALYSIS:",
            "-" * 50
        ])
        
        # Add frame anomalies
        if result.video_features and result.video_features.frame_anomalies:
            report_lines.append("Frame Anomalies:")
            
            for frame_idx, regions in sorted(result.video_features.frame_anomalies.items()):
                report_lines.append(f"  Frame {frame_idx}:")
                
                for i, region in enumerate(regions):
                    report_lines.append(
                        f"    Region {i+1}: Confidence={region.confidence:.2f}, "
                        f"Position=({region.x:.2f},{region.y:.2f},{region.width:.2f},{region.height:.2f}), "
                        f"Label={region.label}"
                    )
            
            report_lines.append("-" * 50)
        
        # Add temporal inconsistencies
        if result.time_segments:
            report_lines.append("Temporal Inconsistencies:")
            
            for i, segment in enumerate(result.time_segments):
                report_lines.append(
                    f"  Segment {i+1}: Time={segment.start_time:.2f}s-{segment.end_time:.2f}s, "
                    f"Confidence={segment.confidence:.2f}, "
                    f"Label={segment.label}"
                )
            
            report_lines.append("-" * 50)
        
        # Add sync issues
        if result.video_features and result.video_features.sync_issues:
            report_lines.append("Audio-Video Sync Issues:")
            
            for i, issue in enumerate(result.video_features.sync_issues):
                report_lines.append(
                    f"  Issue {i+1}: Time={issue.start_time:.2f}s-{issue.end_time:.2f}s, "
                    f"Confidence={issue.confidence:.2f}"
                )
            
            report_lines.append("-" * 50)
        
        # Add detected deepfake categories
        if result.categories:
            report_lines.append("Detected Deepfake Categories:")
            
            for category in result.categories:
                report_lines.append(f"  - {category.value}")
            
            report_lines.append("-" * 50)
        
        # Add additional metadata
        report_lines.append("Additional Metadata:")
        
        for key, value in result.metadata.items():
            report_lines.append(f"  {key}: {value}")
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Text report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating text report: {e}")


def create_analysis_video(
    video_path: str,
    result: DetectionResult,
    output_path: str,
    fps: Optional[float] = None,
    sample_rate: int = 1,  # Process every Nth frame
    include_text_overlay: bool = True
) -> Optional[str]:
    """Create an annotated video with detection results.
    
    Args:
        video_path: Path to the original video file
        result: Detection result
        output_path: Path to save the output video
        fps: Output video FPS (None = use original FPS)
        sample_rate: Process every Nth frame (1 = all frames)
        include_text_overlay: Whether to include text overlay with detection info
        
    Returns:
        Path to output video if successful, None otherwise
    """
    try:
        import cv2
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None
        
        # Get video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use original FPS if not specified
        if fps is None:
            fps = orig_fps
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for compatibility
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (orig_width, orig_height)
        )
        
        # Process frame by frame
        frame_idx = 0
        frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every Nth frame
            if frame_idx % sample_rate == 0:
                # Check if this frame has anomalies
                frame_regions = []
                
                if result.video_features and result.video_features.frame_anomalies:
                    for anomaly_frame in result.video_features.frame_anomalies:
                        # Allow for slight frame index mismatches
                        if abs(anomaly_frame - frame_idx) <= 2:
                            frame_regions = result.video_features.frame_anomalies[anomaly_frame]
                            break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Annotate frame
                annotated_frame = _annotate_frame(frame_rgb, frame_regions)
                
                # Add text overlay if requested
                if include_text_overlay:
                    annotated_frame = _add_text_overlay(
                        annotated_frame, result, frame_idx, total_frames, fps
                    )
                
                # Convert back to BGR
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(annotated_frame_bgr)
                frames_processed += 1
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Analysis video saved to {output_path} ({frames_processed} frames processed)")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating analysis video: {e}")
        return None


def _add_text_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    frame_idx: int,
    total_frames: int,
    fps: float
) -> np.ndarray:
    """Add text overlay with detection information.
    
    Args:
        frame: Frame to add overlay to
        result: Detection result
        frame_idx: Current frame index
        total_frames: Total number of frames
        fps: Frames per second
        
    Returns:
        Frame with text overlay
    """
    import cv2
    
    # Create a copy of the frame
    overlay = frame.copy()
    
    # Calculate current time
    current_time = frame_idx / fps if fps > 0 else 0
    
    # Add semi-transparent background at the top
    height, width = frame.shape[:2]
    overlay_height = 120
    
    cv2.rectangle(
        overlay, (0, 0), (width, overlay_height),
        (0, 0, 0), -1  # Filled rectangle
    )
    
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay[:overlay_height], alpha, frame[:overlay_height], 1 - alpha, 0, frame[:overlay_height])
    
    # Add title
    if result.is_deepfake:
        title = f"DEEPFAKE DETECTED (Confidence: {result.confidence_score:.2f})"
        title_color = (0, 0, 255)  # Red
    else:
        title = f"LIKELY AUTHENTIC (Confidence: {result.confidence_score:.2f})"
        title_color = (0, 255, 0)  # Green
    
    cv2.putText(
        frame, title, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2
    )
    
    # Add frame information
    frame_info = f"Frame: {frame_idx}/{total_frames} | Time: {current_time:.2f}s"
    cv2.putText(
        frame, frame_info, (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    # Add detection categories if available
    if result.categories:
        categories = "Detected: " + ", ".join([cat.value for cat in result.categories])
        cv2.putText(
            frame, categories, (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
    
    # Add timeline at the bottom
    timeline_height = 20
    timeline_y = height - timeline_height - 10
    timeline_width = width - 40
    timeline_x = 20
    
    # Draw timeline background
    cv2.rectangle(
        frame, 
        (timeline_x, timeline_y),
        (timeline_x + timeline_width, timeline_y + timeline_height),
        (50, 50, 50), -1
    )
    
    # Draw progress bar
    if total_frames > 0:
        progress_width = int((frame_idx / total_frames) * timeline_width)
        cv2.rectangle(
            frame, 
            (timeline_x, timeline_y),
            (timeline_x + progress_width, timeline_y + timeline_height),
            (0, 165, 255), -1
        )
    
    # Draw temporal anomalies on timeline
    if result.time_segments:
        for segment in result.time_segments:
            # Calculate pixel positions
            if total_frames > 0 and fps > 0:
                start_pos = int((segment.start_time * fps / total_frames) * timeline_width)
                end_pos = int((segment.end_time * fps / total_frames) * timeline_width)
                
                # Choose color based on confidence
                if segment.confidence >= 0.7:
                    color = (0, 0, 255)  # Red
                elif segment.confidence >= 0.5:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Draw on timeline
                cv2.rectangle(
                    frame, 
                    (timeline_x + start_pos, timeline_y - 5),
                    (timeline_x + end_pos, timeline_y),
                    color, -1
                )
    
    return frame