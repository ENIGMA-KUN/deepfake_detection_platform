"""
Video preprocessing module for deepfake detection.
"""
import os
import logging
import numpy as np
import cv2
import av
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from data.preprocessing.image_prep import ImagePreprocessor
from data.preprocessing.audio_prep import AudioPreprocessor

class VideoPreprocessor:
    """
    Preprocessor for video data to prepare for deepfake detection.
    Handles operations like frame extraction, face detection, and audio separation.
    """
    
    def __init__(self, frames_per_second: int = 5,
                 target_frame_size: Tuple[int, int] = (224, 224),
                 face_detection_threshold: float = 0.9,
                 segment_duration: float = 10.0,
                 device: str = None):
        """
        Initialize the video preprocessor.
        
        Args:
            frames_per_second: Number of frames to extract per second
            target_frame_size: Target size for extracted frames (height, width)
            face_detection_threshold: Confidence threshold for face detection
            segment_duration: Duration of video segments in seconds
            device: Device to use for processing ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set parameters
        self.frames_per_second = frames_per_second
        self.target_frame_size = target_frame_size
        self.face_detection_threshold = face_detection_threshold
        self.segment_duration = segment_duration
        
        # Initialize image and audio preprocessors
        self.image_processor = ImagePreprocessor(
            face_detection_threshold=face_detection_threshold,
            target_size=target_frame_size,
            device=self.device
        )
        
        self.audio_processor = AudioPreprocessor(
            target_sample_rate=16000,
            segment_duration=segment_duration
        )
        
        self.logger.info(f"VideoPreprocessor initialized (device: {self.device})")
    
    def preprocess(self, video_path: str, extract_audio: bool = True,
                  detect_faces: bool = True, segment_video: bool = True) -> Dict[str, Any]:
        """
        Preprocess a video file for deepfake detection.
        
        Args:
            video_path: Path to the video file
            extract_audio: Whether to extract and process audio
            detect_faces: Whether to detect faces in frames
            segment_video: Whether to segment the video into fixed-length chunks
            
        Returns:
            Dictionary containing:
            - frames: List of extracted frames
            - frame_times: List of frame timestamps
            - faces: Dict mapping frame indices to detected faces (if detect_faces=True)
            - audio_features: Dict of audio features (if extract_audio=True)
            - segments: List of video segment info (if segment_video=True)
            - metadata: Additional preprocessing metadata
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the video cannot be processed
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        try:
            # Extract frames and metadata
            frames, frame_times, video_info = self._extract_frames(video_path)
            
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")
            
            result = {
                "frames": frames,
                "frame_times": frame_times,
                "metadata": {
                    "video_info": video_info,
                    "frames_extracted": len(frames),
                    "target_frame_size": self.target_frame_size,
                    "filename": os.path.basename(video_path)
                }
            }
            
            # Detect faces if requested
            if detect_faces:
                faces_by_frame = self._detect_faces_in_frames(frames)
                result["faces"] = faces_by_frame
                result["metadata"]["total_faces_detected"] = sum(len(faces) for faces in faces_by_frame.values())
            
            # Extract and process audio if requested
            if extract_audio:
                audio_features = self._extract_audio_features(video_path)
                result["audio_features"] = audio_features
                
                # Add audio-video sync analysis
                if audio_features:
                    result["metadata"]["has_audio"] = True
                    result["av_sync_analysis"] = self._analyze_audio_video_sync(
                        frame_times, audio_features.get("waveform", None)
                    )
                else:
                    result["metadata"]["has_audio"] = False
            
            # Segment video if requested
            if segment_video:
                segments = self._segment_video(frames, frame_times, video_info["fps"])
                result["segments"] = segments
                result["metadata"]["num_segments"] = len(segments)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing video {video_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess video: {str(e)}")
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
        """
        Extract frames from a video at specified FPS.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple containing:
            - List of frames as numpy arrays
            - List of frame timestamps
            - Dictionary with video info (fps, duration, etc.)
        """
        try:
            # Open the video file
            container = av.open(video_path)
            
            # Get video stream
            video_stream = next((s for s in container.streams if s.type == 'video'), None)
            
            if video_stream is None:
                raise ValueError(f"No video stream found in {video_path}")
            
            # Get video info
            fps = float(video_stream.average_rate)
            duration = float(video_stream.duration * video_stream.time_base)
            width = video_stream.width
            height = video_stream.height
            
            video_info = {
                "fps": fps,
                "duration": duration,
                "width": width,
                "height": height,
                "total_frames": video_stream.frames,
                "codec_name": video_stream.codec_context.name
            }
            
            # Calculate frame interval based on desired sampling rate
            frame_interval = int(fps / self.frames_per_second)
            frame_interval = max(1, frame_interval)  # Ensure at least 1
            
            # Extract frames
            frames = []
            frame_times = []
            
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx % frame_interval == 0:
                    # Convert frame to numpy array
                    img = frame.to_image()
                    img = img.resize(self.target_frame_size, Image.LANCZOS)
                    img_np = np.array(img)
                    
                    # Convert from RGB to BGR (for OpenCV compatibility)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    frames.append(img_np)
                    frame_times.append(float(frame.pts * video_stream.time_base))
            
            return frames, frame_times, video_info
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from video: {str(e)}")
            raise ValueError(f"Failed to extract frames from video: {str(e)}")
    
    def _detect_faces_in_frames(self, frames: List[np.ndarray]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Detect faces in a list of video frames.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Dictionary mapping frame indices to lists of detected faces
        """
        faces_by_frame = {}
        
        for i, frame in enumerate(frames):
            # Convert BGR to RGB (for PIL compatibility)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Process with image preprocessor
            try:
                result = self.image_processor.preprocess(pil_image, extract_faces=True)
                
                if "faces" in result and "face_boxes" in result:
                    faces = []
                    for j, (face, bbox) in enumerate(zip(result["faces"], result["face_boxes"])):
                        faces.append({
                            "face_index": j,
                            "bbox": bbox[:4],  # x1, y1, x2, y2
                            "confidence": bbox[4],
                            "face_array": face
                        })
                    
                    faces_by_frame[i] = faces
                else:
                    faces_by_frame[i] = []
                    
            except Exception as e:
                self.logger.warning(f"Error detecting faces in frame {i}: {str(e)}")
                faces_by_frame[i] = []
        
        return faces_by_frame
    
    def _extract_audio_features(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract audio from video and compute features.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of audio features or None if no audio stream
        """
        try:
            # Extract audio to temporary file
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_audio = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.wav")
            
            # Check if video has audio stream
            container = av.open(video_path)
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            
            if audio_stream is None:
                self.logger.info(f"No audio stream found in {video_path}")
                return None
            
            # Extract audio using ffmpeg via PyAV
            output = av.open(temp_audio, 'w')
            output_stream = output.add_stream('pcm_s16le', rate=16000, layout='mono')
            
            for frame in container.decode(audio=0):
                # Resample frame to desired sample rate
                frame.pts = None
                output.mux(output_stream.encode(frame))
            
            # Flush encoder
            output.mux(output_stream.encode(None))
            output.close()
            
            # Process audio with audio preprocessor
            audio_result = self.audio_processor.preprocess(
                temp_audio, 
                generate_spectrogram=True,
                segment_audio=True
            )
            
            # Clean up temp file
            try:
                os.remove(temp_audio)
            except:
                pass
            
            return audio_result
            
        except Exception as e:
            self.logger.warning(f"Error extracting audio from video: {str(e)}")
            return None
    
    def _segment_video(self, frames: List[np.ndarray], frame_times: List[float], 
                      fps: float) -> List[Dict[str, Any]]:
        """
        Segment video into fixed-length chunks.
        
        Args:
            frames: List of video frames
            frame_times: List of frame timestamps
            fps: Frames per second of the video
            
        Returns:
            List of segment info dictionaries
        """
        segments = []
        
        # Calculate frames per segment
        frames_per_segment = int(self.segment_duration * self.frames_per_second)
        
        # Create segments
        for i in range(0, len(frames), frames_per_segment):
            end_idx = min(i + frames_per_segment, len(frames))
            
            segment_frames = frames[i:end_idx]
            segment_times = frame_times[i:end_idx]
            
            segment = {
                "start_frame": i,
                "end_frame": end_idx - 1,
                "start_time": segment_times[0],
                "end_time": segment_times[-1],
                "frame_count": len(segment_frames),
                "frames": segment_frames
            }
            
            segments.append(segment)
        
        return segments
    
    def _analyze_audio_video_sync(self, frame_times: List[float], 
                                 audio_waveform: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze synchronization between audio and video.
        
        Args:
            frame_times: List of frame timestamps
            audio_waveform: Audio waveform or None
            
        Returns:
            Dictionary with A/V sync analysis results
        """
        # This is a placeholder for actual A/V sync analysis
        # In a real implementation, this would analyze the correlation between
        # visual speech movements and audio speech patterns
        
        if audio_waveform is None:
            return {"sync_score": 0, "has_audio": False}
        
        return {
            "sync_score": 1.0,  # Placeholder value
            "has_audio": True,
            "video_duration": frame_times[-1],
            "audio_duration": len(audio_waveform) / 16000  # Assuming 16kHz sample rate
        }
    
    def extract_optical_flow(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Calculate optical flow between consecutive frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Array of optical flow features
        """
        if len(frames) < 2:
            return np.zeros((0, 2, *self.target_frame_size))
        
        flow_features = []
        
        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Calculate optical flow for each pair of consecutive frames
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            flow_features.append(flow)
        
        return np.array(flow_features)

def preprocess_batch(video_paths: List[str], 
                    processor: Optional[VideoPreprocessor] = None,
                    extract_audio: bool = True,
                    detect_faces: bool = True,
                    segment_video: bool = True,
                    batch_size: int = 4) -> List[Dict[str, Any]]:
    """
    Preprocess a batch of video files.
    
    Args:
        video_paths: List of paths to video files
        processor: Optional VideoPreprocessor instance
        extract_audio: Whether to extract and process audio
        detect_faces: Whether to detect faces in frames
        segment_video: Whether to segment videos
        batch_size: Number of files to process at once
        
    Returns:
        List of preprocessing results for each video file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing batch of {len(video_paths)} videos")
    
    # Create processor if not provided
    if processor is None:
        processor = VideoPreprocessor()
    
    results = []
    
    # Process in batches
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i+batch_size]
        
        for video_path in batch:
            try:
                result = processor.preprocess(
                    video_path, 
                    extract_audio=extract_audio,
                    detect_faces=detect_faces,
                    segment_video=segment_video
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error preprocessing {video_path}: {str(e)}")
                # Add empty result with error message
                results.append({
                    "error": str(e),
                    "video_path": video_path
                })
        
        logger.debug(f"Processed {min(i+batch_size, len(video_paths))}/{len(video_paths)} videos")
    
    return results
