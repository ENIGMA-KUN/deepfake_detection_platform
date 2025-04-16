import os
import cv2
import numpy as np
import tempfile
import torch
from typing import Dict, Tuple, List, Optional, Union
import logging
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Class for preprocessing video data for deepfake detection."""
    
    def __init__(self, config=None):
        """
        Initialize the video preprocessor.
        
        Args:
            config (dict, optional): Configuration parameters for preprocessing
        """
        self.config = config or {}
        
        # Target parameters for video preprocessing
        self.target_fps = self.config.get('target_fps', 30)  # Target frames per second
        self.target_height = self.config.get('target_height', 224)  # Target frame height
        self.target_width = self.config.get('target_width', 224)  # Target frame width
        self.max_frames = self.config.get('max_frames', 150)  # Maximum number of frames to process
        self.normalize = self.config.get('normalize', True)  # Whether to normalize frames
        self.extract_audio = self.config.get('extract_audio', False)  # Whether to extract audio from video
        self.face_detection = self.config.get('face_detection', False)  # Whether to detect faces in frames
        
        # Initialize face detector if enabled
        if self.face_detection:
            self._init_face_detector()
        
        # Default normalization values (ImageNet)
        self.mean = self.config.get('mean', [0.485, 0.456, 0.406])
        self.std = self.config.get('std', [0.229, 0.224, 0.225])
        
        # Create transform pipeline for frames
        self._create_transforms()
    
    def _init_face_detector(self):
        """Initialize the face detector model."""
        try:
            # Load OpenCV's face detector
            face_cascade_path = self.config.get(
                'face_cascade_path', 
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Check if face detector was loaded successfully
            if self.face_cascade.empty():
                logger.warning("Face detector could not be loaded. Face detection disabled.")
                self.face_detection = False
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            self.face_detection = False
    
    def _create_transforms(self):
        """Create the transformation pipeline for preprocessing frames."""
        transform_list = []
        
        # Resize transform
        transform_list.append(transforms.Resize((self.target_height, self.target_width)))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.normalize:
            transform_list.append(transforms.Normalize(self.mean, self.std))
        
        self.transform = transforms.Compose(transform_list)
    
    def preprocess(self, video_path: str, return_tensor: bool = True) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
        """
        Preprocess a video file for deepfake detection.
        
        Args:
            video_path (str): Path to the video file
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            preprocessed_frames: Preprocessed video frames as tensor or NumPy array
            meta_info (dict): Additional information about preprocessing
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_duration = total_frames / original_fps if original_fps > 0 else 0
            
            # Store original information
            meta_info = {
                'original_fps': original_fps,
                'original_total_frames': total_frames,
                'original_width': original_width,
                'original_height': original_height,
                'original_duration': original_duration,
                'preprocessing_steps': [],
                'faces_detected': 0,
                'face_frames': []
            }
            meta_info['preprocessing_steps'].append('load_video')
            
            # Calculate frame sampling
            sample_every = max(1, int(original_fps / self.target_fps))
            max_output_frames = min(self.max_frames, total_frames // sample_every)
            
            # Initialize list to store frames
            frames = []
            frame_count = 0
            sampled_count = 0
            
            # Read frames
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample frames based on target FPS
                if frame_count % sample_every == 0 and sampled_count < max_output_frames:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces if enabled
                    if self.face_detection:
                        face_frame, face_info = self._detect_and_extract_face(frame_rgb, frame_count)
                        
                        if face_info['face_detected']:
                            # Use the face frame instead
                            frame_rgb = face_frame
                            meta_info['faces_detected'] += 1
                            meta_info['face_frames'].append(frame_count)
                    
                    # Convert to PIL Image for transformation
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    # Apply transforms
                    if return_tensor:
                        processed_frame = self.transform(pil_frame)
                    else:
                        # Resize without normalization
                        pil_frame = pil_frame.resize((self.target_width, self.target_height))
                        processed_frame = np.array(pil_frame)
                    
                    frames.append(processed_frame)
                    sampled_count += 1
                
                frame_count += 1
            
            # Release video capture
            cap.release()
            
            # Update meta info
            meta_info['sampled_frames'] = sampled_count
            meta_info['preprocessing_steps'].append('sample_frames')
            
            if return_tensor:
                # Stack frames into tensor
                frames_tensor = torch.stack(frames)
                meta_info['preprocessing_steps'].append('to_tensor')
                
                if self.normalize:
                    meta_info['preprocessing_steps'].append('normalize')
                
                return frames_tensor, meta_info
            else:
                # Stack frames into numpy array
                frames_array = np.array(frames)
                return frames_array, meta_info
                
        except Exception as e:
            logger.error(f"Error preprocessing video {video_path}: {e}")
            raise
    
    def _detect_and_extract_face(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Detect and extract face from a video frame.
        
        Args:
            frame (np.ndarray): Input frame
            frame_idx (int): Frame index
            
        Returns:
            np.ndarray: Frame with extracted face or original if no face found
            dict: Information about detected face
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_info = {
            'face_detected': len(faces) > 0,
            'frame_idx': frame_idx,
            'num_faces': len(faces),
            'face_regions': []
        }
        
        # If faces found, extract the largest one
        if len(faces) > 0:
            # Find largest face by area
            largest_face_idx = np.argmax([w * h for (x, y, w, h) in faces])
            x, y, w, h = faces[largest_face_idx]
            
            # Add padding around face (20%)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            # Ensure coordinates are within image bounds
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(frame.shape[1], x + w + padding_x)
            y_end = min(frame.shape[0], y + h + padding_y)
            
            # Extract face region
            face_region = frame[y_start:y_end, x_start:x_end]
            
            # Store face location
            face_info['face_regions'].append({
                'x': x_start,
                'y': y_start,
                'width': x_end - x_start,
                'height': y_end - y_start,
                'is_largest': True
            })
            
            return face_region, face_info
        
        # If no face found, return original frame
        return frame, face_info
    
    def extract_audio(self, video_path: str, audio_output_path: str = None) -> Tuple[str, Dict]:
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to the video file
            audio_output_path (str, optional): Path to save the extracted audio
            
        Returns:
            str: Path to the extracted audio file
            dict: Meta information about the extraction
        """
        try:
            import subprocess
            
            # Create a temporary file if no output path provided
            if audio_output_path is None:
                temp_dir = tempfile.gettempdir()
                audio_output_path = os.path.join(temp_dir, f"extracted_audio_{os.path.basename(video_path)}.wav")
            
            # Extract audio using ffmpeg
            command = [
                'ffmpeg',
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                '-c:a', 'pcm_s16le',  # Uncompressed WAV format
                audio_output_path,
                '-y'  # Overwrite if exists
            ]
            
            # Run the command
            result = subprocess.run(command, capture_output=True)
            
            # Check if successful
            if result.returncode != 0:
                error_message = result.stderr.decode('utf-8')
                logger.error(f"Error extracting audio: {error_message}")
                raise RuntimeError(f"Failed to extract audio: {error_message}")
            
            # Meta information
            meta_info = {
                'source_video': video_path,
                'audio_path': audio_output_path,
                'format': 'wav',
                'extraction_successful': True
            }
            
            return audio_output_path, meta_info
            
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            
            # Meta information for failed extraction
            meta_info = {
                'source_video': video_path,
                'audio_path': None,
                'extraction_successful': False,
                'error': str(e)
            }
            
            return None, meta_info
    
    def compute_motion_metrics(self, frames: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute motion-based metrics from video frames.
        
        Args:
            frames (np.ndarray or torch.Tensor): Video frames
            
        Returns:
            dict: Dictionary of motion metrics
        """
        # Convert tensor to numpy if needed
        if torch.is_tensor(frames):
            # Convert to numpy (CPU tensor to numpy, remove normalization)
            frames_np = frames.cpu().numpy()
            if self.normalize:
                # Reverse normalization
                for i in range(3):
                    frames_np[:, i] = frames_np[:, i] * self.std[i] + self.mean[i]
                frames_np = np.clip(frames_np, 0, 1)
            
            # Rearrange from [frames, channels, height, width] to [frames, height, width, channels]
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
            
            # Convert to uint8 (0-255)
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames
        
        # Compute frame differences
        frame_diffs = []
        for i in range(1, len(frames_np)):
            # Convert to grayscale
            prev_frame = cv2.cvtColor(frames_np[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            frame_diffs.append(np.mean(diff))
        
        # Compute metrics
        if frame_diffs:
            mean_motion = np.mean(frame_diffs)
            std_motion = np.std(frame_diffs)
            max_motion = np.max(frame_diffs)
        else:
            mean_motion = 0
            std_motion = 0
            max_motion = 0
        
        # Compute optical flow (sparse Lucas-Kanade)
        flow_magnitudes = []
        
        if len(frames_np) > 1:
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Get first frame and extract features
            first_frame_gray = cv2.cvtColor(frames_np[0], cv2.COLOR_RGB2GRAY)
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(first_frame_gray, mask=None, **feature_params)
            
            if p0 is not None:
                for i in range(1, len(frames_np)):
                    curr_frame_gray = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
                    
                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(first_frame_gray, curr_frame_gray, p0, None, **lk_params)
                    
                    if p1 is not None:
                        # Select good points
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        
                        # Calculate flow magnitudes
                        if len(good_new) > 0 and len(good_old) > 0:
                            flow_mag = np.mean(np.sqrt(
                                (good_new[:, 0] - good_old[:, 0])**2 + 
                                (good_new[:, 1] - good_old[:, 1])**2
                            ))
                            flow_magnitudes.append(flow_mag)
                        
                        # Update previous frame and points
                        first_frame_gray = curr_frame_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)
        
        # Compute flow metrics
        if flow_magnitudes:
            mean_flow = np.mean(flow_magnitudes)
            std_flow = np.std(flow_magnitudes)
            max_flow = np.max(flow_magnitudes)
        else:
            mean_flow = 0
            std_flow = 0
            max_flow = 0
        
        return {
            'mean_motion': mean_motion,
            'std_motion': std_motion,
            'max_motion': max_motion,
            'mean_flow': mean_flow,
            'std_flow': std_flow,
            'max_flow': max_flow
        }
    
    def detect_scene_changes(self, frames: Union[np.ndarray, torch.Tensor], threshold: float = 30.0) -> List[int]:
        """
        Detect scene changes in video frames.
        
        Args:
            frames (np.ndarray or torch.Tensor): Video frames
            threshold (float): Threshold for scene change detection
            
        Returns:
            list: Indices of frames where scene changes occur
        """
        # Convert tensor to numpy if needed
        if torch.is_tensor(frames):
            # Convert to numpy (CPU tensor to numpy, remove normalization)
            frames_np = frames.cpu().numpy()
            if self.normalize:
                # Reverse normalization
                for i in range(3):
                    frames_np[:, i] = frames_np[:, i] * self.std[i] + self.mean[i]
                frames_np = np.clip(frames_np, 0, 1)
            
            # Rearrange from [frames, channels, height, width] to [frames, height, width, channels]
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
            
            # Convert to uint8 (0-255)
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames
        
        # Initialize detector
        scene_changes = []
        
        # Compute frame differences
        for i in range(1, len(frames_np)):
            # Convert to grayscale
            prev_frame = cv2.cvtColor(frames_np[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            mean_diff = np.mean(diff)
            
            # Check if difference exceeds threshold
            if mean_diff > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    def batch_preprocess(self, video_paths: List[str], return_tensor: bool = True) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[Dict]]:
        """
        Preprocess a batch of video files.
        
        Args:
            video_paths (list): List of paths to video files
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            list: List of preprocessed videos
            list: List of meta info dictionaries
        """
        preprocessed_videos = []
        meta_infos = []
        
        for video_path in video_paths:
            try:
                processed_video, meta_info = self.preprocess(video_path, return_tensor)
                preprocessed_videos.append(processed_video)
                meta_infos.append(meta_info)
            except Exception as e:
                logger.error(f"Error preprocessing video {video_path}: {e}")
                # Skip failed videos
                continue
        
        return preprocessed_videos, meta_infos