import numpy as np
import torch
import cv2
import librosa
import random
from typing import Union, Tuple, List, Dict, Optional, Any
import logging
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

class AugmentationPipeline:
    """Pipeline for applying augmentations to different media types."""
    
    def __init__(self, config=None):
        """
        Initialize the augmentation pipeline.
        
        Args:
            config (dict, optional): Configuration parameters for augmentation
        """
        self.config = config or {}
        
        # Probabilities for different augmentation types
        self.image_aug_prob = self.config.get('image_aug_prob', 0.5)
        self.audio_aug_prob = self.config.get('audio_aug_prob', 0.5)
        self.video_aug_prob = self.config.get('video_aug_prob', 0.5)
        
        # Number of augmentations to apply
        self.num_image_augs = self.config.get('num_image_augs', 2)
        self.num_audio_augs = self.config.get('num_audio_augs', 2)
        self.num_video_augs = self.config.get('num_video_augs', 2)
    
    def augment_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image], 
                      num_augs: int = None) -> Tuple[Union[np.ndarray, torch.Tensor, Image.Image], Dict]:
        """
        Apply augmentation to an image.
        
        Args:
            image: Input image (NumPy array, torch tensor, or PIL Image)
            num_augs (int, optional): Number of augmentations to apply
            
        Returns:
            Augmented image
            dict: Information about applied augmentations
        """
        if num_augs is None:
            num_augs = self.num_image_augs
        
        # Convert to PIL Image if not already
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
            input_type = 'numpy'
        elif isinstance(image, torch.Tensor):
            # Assuming tensor is [C, H, W] format with values in [0, 1]
            if image.dim() == 3:
                # Convert to [H, W, C] for numpy
                np_image = image.permute(1, 2, 0).cpu().numpy()
                
                # Scale to [0, 255] if in [0, 1]
                if np_image.max() <= 1.0:
                    np_image = (np_image * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(np_image)
            else:
                raise ValueError("Expected tensor of shape [C, H, W]")
            
            input_type = 'tensor'
        elif isinstance(image, Image.Image):
            pil_image = image
            input_type = 'pil'
        else:
            raise ValueError("Unsupported image type")
        
        # Available augmentations
        augmentations = [
            self._image_color_jitter,
            self._image_random_rotation,
            self._image_random_crop_resize,
            self._image_random_blur,
            self._image_random_noise,
            self._image_random_flip,
        ]
        
        # Randomly select augmentations
        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
        
        # Apply selected augmentations
        aug_info = {'applied_augmentations': []}
        
        for aug_func in selected_augs:
            # Apply augmentation with probability
            if random.random() < self.image_aug_prob:
                try:
                    pil_image, aug_details = aug_func(pil_image)
                    aug_info['applied_augmentations'].append(aug_details)
                except Exception as e:
                    logger.warning(f"Error applying image augmentation {aug_func.__name__}: {e}")
        
        # Convert back to original format
        if input_type == 'numpy':
            return np.array(pil_image), aug_info
        elif input_type == 'tensor':
            # Convert back to tensor [C, H, W] with original scaling
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            tensor_image = torch.from_numpy(np_image).permute(2, 0, 1)
            return tensor_image, aug_info
        else:  # PIL
            return pil_image, aug_info
    
    def _image_color_jitter(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply random color jitter to image."""
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        
        # Apply enhancements
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        image = ImageEnhance.Color(image).enhance(saturation_factor)
        
        return image, {
            'type': 'color_jitter',
            'brightness_factor': brightness_factor,
            'contrast_factor': contrast_factor,
            'saturation_factor': saturation_factor
        }
    
    def _image_random_rotation(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply random rotation to image."""
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        return image, {
            'type': 'random_rotation',
            'angle': angle
        }
    
    def _image_random_crop_resize(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply random crop and resize to image."""
        width, height = image.size
        
        # Random crop size (between 80% and 100% of original size)
        crop_ratio = random.uniform(0.8, 1.0)
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        
        # Random crop position
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        # Crop and resize back to original size
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((width, height), Image.BILINEAR)
        
        return image, {
            'type': 'random_crop_resize',
            'crop_ratio': crop_ratio,
            'crop_position': (left, top)
        }
    
    def _image_random_blur(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply random blur to image."""
        radius = random.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image, {
            'type': 'random_blur',
            'radius': radius
        }
    
    def _image_random_noise(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Add random noise to image."""
        # Convert to numpy
        np_image = np.array(image).astype(np.float32)
        
        # Add noise
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, np_image.shape)
        np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        image = Image.fromarray(np_image)
        
        return image, {
            'type': 'random_noise',
            'noise_level': noise_level
        }
    
    def _image_random_flip(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply random horizontal flip to image."""
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip_applied = True
        else:
            flip_applied = False
        
        return image, {
            'type': 'random_flip',
            'flipped': flip_applied
        }
    
    def augment_audio(self, audio: np.ndarray, sr: int, num_augs: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply augmentation to audio data.
        
        Args:
            audio (np.ndarray): Input audio data
            sr (int): Sample rate
            num_augs (int, optional): Number of augmentations to apply
            
        Returns:
            np.ndarray: Augmented audio data
            dict: Information about applied augmentations
        """
        if num_augs is None:
            num_augs = self.num_audio_augs
        
        # Available augmentations
        augmentations = [
            self._audio_time_shift,
            self._audio_pitch_shift,
            self._audio_time_stretch,
            self._audio_add_noise,
            self._audio_filter
        ]
        
        # Randomly select augmentations
        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
        
        # Apply selected augmentations
        aug_info = {'applied_augmentations': []}
        
        # Create a copy of the audio data
        augmented_audio = np.copy(audio)
        
        for aug_func in selected_augs:
            # Apply augmentation with probability
            if random.random() < self.audio_aug_prob:
                try:
                    augmented_audio, aug_details = aug_func(augmented_audio, sr)
                    aug_info['applied_augmentations'].append(aug_details)
                except Exception as e:
                    logger.warning(f"Error applying audio augmentation {aug_func.__name__}: {e}")
        
        return augmented_audio, aug_info
    
    def _audio_time_shift(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Apply random time shift to audio."""
        shift_factor = random.uniform(0.1, 0.5)
        shift_amount = int(len(audio) * shift_factor)
        
        # Shift audio
        augmented_audio = np.roll(audio, shift_amount)
        
        # To avoid abrupt changes, apply fade-in/out at the roll point
        fade_len = min(1000, shift_amount)
        if shift_amount > 0:
            # Fade out at the end
            fade_out = np.linspace(1.0, 0.0, fade_len)
            augmented_audio[-fade_len:] *= fade_out
            # Fade in at the beginning
            fade_in = np.linspace(0.0, 1.0, fade_len)
            augmented_audio[:fade_len] *= fade_in
        
        return augmented_audio, {
            'type': 'time_shift',
            'shift_factor': shift_factor,
            'shift_amount': shift_amount
        }
    
    def _audio_pitch_shift(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Apply pitch shift to audio."""
        n_steps = random.uniform(-4.0, 4.0)
        augmented_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
        return augmented_audio, {
            'type': 'pitch_shift',
            'n_steps': n_steps
        }
    
    def _audio_time_stretch(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Apply time stretch to audio."""
        rate = random.uniform(0.8, 1.2)
        augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Ensure the augmented audio has the same length as the original
        if len(augmented_audio) > len(audio):
            augmented_audio = augmented_audio[:len(audio)]
        else:
            augmented_audio = np.pad(augmented_audio, (0, max(0, len(audio) - len(augmented_audio))), 'constant')
        
        return augmented_audio, {
            'type': 'time_stretch',
            'rate': rate
        }
    
    def _audio_add_noise(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Add random noise to audio."""
        noise_level = random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level, len(audio))
        augmented_audio = audio + noise
        
        return augmented_audio, {
            'type': 'add_noise',
            'noise_level': noise_level
        }
    
    def _audio_filter(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Apply random filter to audio."""
        filter_type = random.choice(['lowpass', 'highpass'])
        cutoff_freq = random.uniform(1000, 4000) if filter_type == 'lowpass' else random.uniform(50, 500)
        
        # Apply filter
        if filter_type == 'lowpass':
            # Simple lowpass filter using a moving average
            window_size = int(sr / cutoff_freq)
            window = np.ones(window_size) / window_size
            augmented_audio = np.convolve(audio, window, mode='same')
        else:  # highpass
            # Approximate highpass by subtracting lowpass from original
            window_size = int(sr / cutoff_freq)
            window = np.ones(window_size) / window_size
            lowpass = np.convolve(audio, window, mode='same')
            augmented_audio = audio - lowpass
        
        return augmented_audio, {
            'type': 'filter',
            'filter_type': filter_type,
            'cutoff_freq': cutoff_freq
        }
    
    def augment_video(self, frames: Union[np.ndarray, torch.Tensor], num_augs: int = None) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
        """
        Apply augmentation to video frames.
        
        Args:
            frames: Input video frames (NumPy array or torch tensor)
                   Expected shape: [T, H, W, C] for NumPy or [T, C, H, W] for torch
            num_augs (int, optional): Number of augmentations to apply
            
        Returns:
            Augmented video frames
            dict: Information about applied augmentations
        """
        if num_augs is None:
            num_augs = self.num_video_augs
        
        # Check if frames are torch tensor
        is_tensor = isinstance(frames, torch.Tensor)
        
        # Convert tensor to numpy for processing if needed
        if is_tensor:
            # Assuming tensor is [T, C, H, W] format with values in [0, 1]
            np_frames = frames.permute(0, 2, 3, 1).cpu().numpy()
            
            # Scale to [0, 255] if in [0, 1]
            if np_frames.max() <= 1.0:
                np_frames = (np_frames * 255).astype(np.uint8)
        else:
            np_frames = frames
        
        # Available augmentations
        augmentations = [
            self._video_temporal_crop,
            self._video_frame_dropout,
            self._video_temporal_flip,
            self._video_brightness_contrast,
            self._video_spatial_crop,
            self._video_rotation
        ]
        
        # Randomly select augmentations
        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
        
        # Apply selected augmentations
        aug_info = {'applied_augmentations': []}
        
        # Create a copy of the frames
        augmented_frames = np_frames.copy()
        
        for aug_func in selected_augs:
            # Apply augmentation with probability
            if random.random() < self.video_aug_prob:
                try:
                    augmented_frames, aug_details = aug_func(augmented_frames)
                    aug_info['applied_augmentations'].append(aug_details)
                except Exception as e:
                    logger.warning(f"Error applying video augmentation {aug_func.__name__}: {e}")
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            # Convert back to tensor [T, C, H, W] with original scaling
            augmented_frames = augmented_frames.astype(np.float32) / 255.0
            augmented_frames = torch.from_numpy(augmented_frames).permute(0, 3, 1, 2)
            return augmented_frames, aug_info
        else:
            return augmented_frames, aug_info
    
    def _video_temporal_crop(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply random temporal crop to video frames."""
        num_frames = frames.shape[0]
        
        # Randomly select a crop ratio (between 80% and 100%)
        crop_ratio = random.uniform(0.8, 1.0)
        crop_size = int(num_frames * crop_ratio)
        
        # Randomly select a start point
        start_idx = random.randint(0, num_frames - crop_size)
        
        # Crop frames
        cropped_frames = frames[start_idx:start_idx + crop_size]
        
        # Repeat frames to maintain original length
        if crop_size < num_frames:
            repeat_times = num_frames // crop_size
            remainder = num_frames % crop_size
            
            repeated_frames = np.repeat(cropped_frames, repeat_times, axis=0)
            if remainder > 0:
                remainder_frames = cropped_frames[:remainder]
                repeated_frames = np.concatenate([repeated_frames, remainder_frames], axis=0)
            
            augmented_frames = repeated_frames
        else:
            augmented_frames = cropped_frames
        
        return augmented_frames, {
            'type': 'temporal_crop',
            'crop_ratio': crop_ratio,
            'start_idx': start_idx,
            'crop_size': crop_size
        }
    
    def _video_frame_dropout(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Randomly drop and interpolate frames."""
        num_frames = frames.shape[0]
        
        # Calculate number of frames to drop (up to 20%)
        drop_ratio = random.uniform(0.05, 0.2)
        num_drop = int(num_frames * drop_ratio)
        
        # Ensure at least 1 frame is dropped
        num_drop = max(1, num_drop)
        
        # Randomly select frames to drop
        drop_indices = sorted(random.sample(range(1, num_frames - 1), num_drop))
        
        # Create a mask of frames to keep
        mask = np.ones(num_frames, dtype=bool)
        mask[drop_indices] = False
        
        # Keep frames that weren't dropped
        kept_frames = frames[mask]
        
        # Interpolate to maintain original frame count
        augmented_frames = np.zeros_like(frames)
        augmented_frames[mask] = kept_frames
        
        # For each dropped frame, interpolate from neighbors
        for idx in drop_indices:
            # Find the nearest kept frames before and after
            prev_idx = idx - 1
            while prev_idx >= 0 and prev_idx in drop_indices:
                prev_idx -= 1
            
            next_idx = idx + 1
            while next_idx < num_frames and next_idx in drop_indices:
                next_idx += 1
            
            # If both neighbors are valid, interpolate
            if prev_idx >= 0 and next_idx < num_frames:
                augmented_frames[idx] = (frames[prev_idx] + frames[next_idx]) / 2
            # Otherwise, use the one valid neighbor
            elif prev_idx >= 0:
                augmented_frames[idx] = frames[prev_idx]
            elif next_idx < num_frames:
                augmented_frames[idx] = frames[next_idx]
        
        return augmented_frames, {
            'type': 'frame_dropout',
            'drop_ratio': drop_ratio,
            'drop_indices': drop_indices
        }
    
    def _video_temporal_flip(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Reverse the temporal order of frames."""
        augmented_frames = frames[::-1].copy()
        
        return augmented_frames, {
            'type': 'temporal_flip'
        }
    
    def _video_brightness_contrast(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply random brightness and contrast adjustment to all frames."""
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        # Apply to all frames
        augmented_frames = frames.copy()
        
        for i in range(len(augmented_frames)):
            # Convert to PIL for easier manipulation
            frame_pil = Image.fromarray(augmented_frames[i])
            
            # Apply brightness
            frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness_factor)
            
            # Apply contrast
            frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast_factor)
            
            # Convert back to numpy
            augmented_frames[i] = np.array(frame_pil)
        
        return augmented_frames, {
            'type': 'brightness_contrast',
            'brightness_factor': brightness_factor,
            'contrast_factor': contrast_factor
        }
    
    def _video_spatial_crop(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply same spatial crop to all frames."""
        height, width = frames.shape[1:3]
        
        # Random crop size (between 80% and 100% of original size)
        crop_ratio = random.uniform(0.8, 1.0)
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        # Random crop position
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        # Apply crop to all frames
        cropped_frames = frames[:, top:top+crop_height, left:left+crop_width].copy()
        
        # Resize back to original dimensions
        augmented_frames = np.zeros_like(frames)
        for i in range(len(cropped_frames)):
            # Resize using OpenCV
            augmented_frames[i] = cv2.resize(cropped_frames[i], (width, height))
        
        return augmented_frames, {
            'type': 'spatial_crop',
            'crop_ratio': crop_ratio,
            'crop_position': (top, left)
        }
    
    def _video_rotation(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply same rotation to all frames."""
        angle = random.uniform(-15, 15)
        
        # Apply rotation to all frames
        augmented_frames = np.zeros_like(frames)
        
        for i in range(len(frames)):
            # Get frame dimensions
            height, width = frames[i].shape[:2]
            center = (width / 2, height / 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            augmented_frames[i] = cv2.warpAffine(frames[i], rotation_matrix, (width, height))
        
        return augmented_frames, {
            'type': 'rotation',
            'angle': angle
        }
    
    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation to a batch of data.
        
        Args:
            batch (dict): Batch containing different media types
                Expected keys may include 'images', 'audio', 'video'
            
        Returns:
            dict: Augmented batch
        """
        augmented_batch = {}
        aug_info = {}
        
        # Augment images if present
        if 'images' in batch:
            augmented_images = []
            image_aug_info = []
            
            for img in batch['images']:
                aug_img, info = self.augment_image(img)
                augmented_images.append(aug_img)
                image_aug_info.append(info)
            
            augmented_batch['images'] = augmented_images
            aug_info['images'] = image_aug_info
        
        # Augment audio if present
        if 'audio' in batch and 'sr' in batch:
            augmented_audio = []
            audio_aug_info = []
            
            for audio in batch['audio']:
                aug_audio, info = self.augment_audio(audio, batch['sr'])
                augmented_audio.append(aug_audio)
                audio_aug_info.append(info)
            
            augmented_batch['audio'] = augmented_audio
            augmented_batch['sr'] = batch['sr']  # Keep sample rate the same
            aug_info['audio'] = audio_aug_info
        
        # Augment video if present
        if 'video' in batch:
            augmented_video = []
            video_aug_info = []
            
            for video in batch['video']:
                aug_video, info = self.augment_video(video)
                augmented_video.append(aug_video)
                video_aug_info.append(info)
            
            augmented_batch['video'] = augmented_video
            aug_info['video'] = video_aug_info
        
        # Copy any other keys
        for key in batch:
            if key not in augmented_batch and key not in ['images', 'audio', 'video', 'sr']:
                augmented_batch[key] = batch[key]
        
        # Add augmentation info
        augmented_batch['augmentation_info'] = aug_info
        
        return augmented_batch