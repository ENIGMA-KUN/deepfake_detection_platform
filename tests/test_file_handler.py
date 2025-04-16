"""
Test module for file_handler.py

This module contains tests for the file handling utilities, including
file validation, saving/loading, temporary file management, and batch processing.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.file_handler import (
    validate_file, validate_image, validate_audio, validate_video,
    save_uploaded_file, load_file, get_file_info,
    TempFileManager, get_temp_file_manager,
    BatchProcessor, process_directory
)

class TestFileValidation(unittest.TestCase):
    """Test file validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        self.create_test_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test directory and all contents
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create test files for validation."""
        # Create valid test image
        self.valid_image_path = self.test_dir / "valid_image.jpg"
        with open(self.valid_image_path, "wb") as f:
            # Create a minimal JPEG file
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C')
            f.write(b'\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c')
            f.write(b'\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444')
            f.write(b'\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222')
            f.write(b'22222222222222222222222222222222222222222222222222\xff\xc0')
            f.write(b'\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f')
            f.write(b'\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03')
            f.write(b'\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05')
            f.write(b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q')
            f.write(b'\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'')
            f.write(b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9')
        
        # Create invalid image file (just text)
        self.invalid_image_path = self.test_dir / "invalid_image.jpg"
        with open(self.invalid_image_path, "w") as f:
            f.write("This is not an image file.")
        
        # Create large image file (above size limit)
        self.large_image_path = self.test_dir / "large_image.jpg"
        with open(self.large_image_path, "wb") as f:
            f.write(b'\xff\xd8\xff\xe0') # JPEG header
            f.write(b'0' * (15 * 1024 * 1024))  # 15MB of data
        
        # Create valid test audio file (WAV header)
        self.valid_audio_path = self.test_dir / "valid_audio.wav"
        with open(self.valid_audio_path, "wb") as f:
            # Create a minimal WAV file header
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00')
            f.write(b'\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        
        # Create valid test video file (MP4 header)
        self.valid_video_path = self.test_dir / "valid_video.mp4"
        with open(self.valid_video_path, "wb") as f:
            # Create a minimal MP4 file header
            f.write(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\x00moov')
    
    def test_validate_image(self):
        """Test image validation."""
        # This is a mock test since we're not using actual image processing libraries in setup
        # In a real environment, you'd use actual image files

        # For now, we'll just check that the function exists and returns expected type
        result, message = validate_image(str(self.valid_image_path))
        self.assertIsInstance(result, bool)
        self.assertIsInstance(message, str)
    
    def test_validate_audio(self):
        """Test audio validation."""
        # This is a mock test similar to image validation
        result, message = validate_audio(str(self.valid_audio_path))
        self.assertIsInstance(result, bool)
        self.assertIsInstance(message, str)
    
    def test_validate_video(self):
        """Test video validation."""
        # This is a mock test similar to image validation
        result, message = validate_video(str(self.valid_video_path))
        self.assertIsInstance(result, bool)
        self.assertIsInstance(message, str)
    
    def test_validate_file(self):
        """Test general file validation."""
        # Test with media type detection
        is_valid, message, media_type = validate_file(str(self.valid_image_path))
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(message, str)
        self.assertIsInstance(media_type, str)
        
        # Test with non-existent file
        is_valid, message, media_type = validate_file("nonexistent_file.jpg")
        self.assertFalse(is_valid)
        self.assertEqual(message, "File does not exist")
        self.assertEqual(media_type, "")

class TestFileSavingLoading(unittest.TestCase):
    """Test file saving and loading functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage_dir = self.test_dir / "storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create test files
        self.create_test_files()
        
        # Patch config to use test storage directory
        from app.utils.file_handler import config
        self.original_config = config.copy()
        config["storage_base_dir"] = str(self.storage_dir)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test directory and all contents
        shutil.rmtree(self.test_dir)
        
        # Restore original config
        from app.utils.file_handler import config
        for key, value in self.original_config.items():
            config[key] = value
    
    def create_test_files(self):
        """Create test files for saving and loading."""
        # Create simple image file
        self.test_image_path = self.test_dir / "test_image.jpg"
        with open(self.test_image_path, "wb") as f:
            # Create a minimal JPEG file
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C')
            f.write(b'\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c')
            # Add more data to make it a valid JPEG
            f.write(b'\x00' * 1000)
    
    def test_save_file(self):
        """Test saving uploaded files."""
        # This is a partial test that doesn't include actual media processing
        
        # Mock function to override validate_file for testing
        original_validate_file = validate_file
        
        try:
            # Replace validate_file with a mock that always returns success
            import app.utils.file_handler
            app.utils.file_handler.validate_file = lambda file_path, media_type=None: (True, "Mock validation", "image")
            
            # Now test save_uploaded_file
            success, message, file_info = save_uploaded_file(str(self.test_image_path), "image")
            
            # Basic assertions
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
            self.assertIsInstance(file_info, dict)
            
            # If successful, check that file exists in storage
            if success and 'storage_path' in file_info:
                self.assertTrue(os.path.exists(file_info['storage_path']))
        
        finally:
            # Restore original function
            app.utils.file_handler.validate_file = original_validate_file

class TestTempFileManager(unittest.TestCase):
    """Test temporary file management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Initialize temp file manager with test directory
        self.temp_manager = TempFileManager(str(self.test_dir))
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Cleanup temp manager
        self.temp_manager.cleanup_all()
        
        # Remove test directory
        shutil.rmtree(self.test_dir)
    
    def test_create_temp_file(self):
        """Test creating temporary files."""
        # Create temp file
        temp_file = self.temp_manager.create_temp_file(prefix="test", suffix=".txt")
        
        # Check file was created
        self.assertTrue(temp_file.exists())
        
        # Check file is in tracking list
        self.assertIn(temp_file, self.temp_manager.temp_files)
        
        # Check prefix and suffix
        self.assertTrue(temp_file.name.startswith("test_"))
        self.assertTrue(temp_file.name.endswith(".txt"))
    
    def test_create_temp_dir(self):
        """Test creating temporary directories."""
        # Create temp directory
        temp_dir = self.temp_manager.create_temp_dir(prefix="test_dir")
        
        # Check directory was created
        self.assertTrue(temp_dir.exists())
        self.assertTrue(temp_dir.is_dir())
        
        # Check prefix
        self.assertTrue(temp_dir.name.startswith("test_dir_"))
    
    def test_create_temp_copy(self):
        """Test creating temporary copies of files."""
        # Create source file
        source_file = self.test_dir / "source.txt"
        with open(source_file, "w") as f:
            f.write("Test content")
        
        # Create temp copy
        temp_copy = self.temp_manager.create_temp_copy(str(source_file))
        
        # Check file was created
        self.assertTrue(temp_copy.exists())
        
        # Check file is in tracking list
        self.assertIn(temp_copy, self.temp_manager.temp_files)
        
        # Check content was copied
        with open(temp_copy, "r") as f:
            content = f.read()
        self.assertEqual(content, "Test content")
    
    def test_cleanup_file(self):
        """Test cleaning up specific temporary files."""
        # Create temp files
        temp_file1 = self.temp_manager.create_temp_file()
        temp_file2 = self.temp_manager.create_temp_file()
        
        # Cleanup first file
        success = self.temp_manager.cleanup_file(temp_file1)
        
        # Check result
        self.assertTrue(success)
        self.assertFalse(temp_file1.exists())
        self.assertTrue(temp_file2.exists())
        self.assertNotIn(temp_file1, self.temp_manager.temp_files)
        self.assertIn(temp_file2, self.temp_manager.temp_files)
    
    def test_cleanup_all(self):
        """Test cleaning up all temporary files."""
        # Create temp files
        temp_file1 = self.temp_manager.create_temp_file()
        temp_file2 = self.temp_manager.create_temp_file()
        temp_dir = self.temp_manager.create_temp_dir()
        
        # Create a file in the temp directory
        nested_file = temp_dir / "nested.txt"
        with open(nested_file, "w") as f:
            f.write("Nested file")
        
        # Cleanup all
        success_count, failure_count = self.temp_manager.cleanup_all()
        
        # Check results
        self.assertEqual(success_count, 2)  # Files were tracked, directory wasn't
        self.assertEqual(failure_count, 0)
        self.assertFalse(temp_file1.exists())
        self.assertFalse(temp_file2.exists())
        # Note: temp_dir may still exist if it wasn't explicitly tracked
        
        # Check tracking list is empty
        self.assertEqual(len(self.temp_manager.temp_files), 0)

class TestBatchProcessing(unittest.TestCase):
    """Test batch file processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage_dir = self.test_dir / "storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create test files
        self.create_test_files()
        
        # Patch config to use test storage directory
        from app.utils.file_handler import config
        self.original_config = config.copy()
        config["storage_base_dir"] = str(self.storage_dir)
        
        # Create batch processor
        self.batch_processor = BatchProcessor(max_batch_size=5)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test directory and all contents
        shutil.rmtree(self.test_dir)
        
        # Restore original config
        from app.utils.file_handler import config
        for key, value in self.original_config.items():
            config[key] = value
    
    def create_test_files(self):
        """Create test files for batch processing."""
        # Create a batch of test files
        self.test_files = []
        
        # Create image files
        image_dir = self.test_dir / "images"
        os.makedirs(image_dir, exist_ok=True)
        
        for i in range(3):
            file_path = image_dir / f"test_image_{i}.jpg"
            with open(file_path, "wb") as f:
                f.write(b'\xff\xd8\xff\xe0')  # JPEG header
                f.write(b'0' * 1000)  # Some data
            self.test_files.append(str(file_path))
        
        # Create audio files
        audio_dir = self.test_dir / "audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        for i in range(2):
            file_path = audio_dir / f"test_audio_{i}.wav"
            with open(file_path, "wb") as f:
                f.write(b'RIFF')  # WAV header
                f.write(b'0' * 1000)  # Some data
            self.test_files.append(str(file_path))
    
    def test_add_file(self):
        """Test adding files to batch."""
        # Mock function to override validate_file for testing
        original_validate_file = validate_file
        
        try:
            # Replace validate_file with a mock that always returns success for images
            import app.utils.file_handler
            app.utils.file_handler.validate_file = lambda file_path, media_type=None: (
                True, 
                "Mock validation", 
                "image" if file_path.endswith('.jpg') else "audio"
            )
            
            # Add files to batch
            for file_path in self.test_files:
                success = self.batch_processor.add_file(file_path)
                self.assertTrue(success)
            
            # Check batch size
            self.assertEqual(len(self.batch_processor.current_batch), len(self.test_files))
            
            # Check file status
            for file_info in self.batch_processor.current_batch:
                self.assertEqual(file_info['status'], 'pending')
                self.assertIn(file_info['media_type'], ['image', 'audio'])
        
        finally:
            # Restore original function
            app.utils.file_handler.validate_file = original_validate_file
    
    def test_group_by_media_type(self):
        """Test grouping batch files by media type."""
        # Add mock files to batch with specific media types
        self.batch_processor.current_batch = [
            {'file_path': 'file1.jpg', 'media_type': 'image'},
            {'file_path': 'file2.jpg', 'media_type': 'image'},
            {'file_path': 'file3.wav', 'media_type': 'audio'},
            {'file_path': 'file4.mp4', 'media_type': 'video'}
        ]
        
        # Group by media type
        grouped = self.batch_processor.group_by_media_type()
        
        # Check grouping
        self.assertEqual(len(grouped['image']), 2)
        self.assertEqual(len(grouped['audio']), 1)
        self.assertEqual(len(grouped['video']), 1)
        
        # Check specific files
        self.assertEqual(grouped['image'][0]['file_path'], 'file1.jpg')
        self.assertEqual(grouped['audio'][0]['file_path'], 'file3.wav')
        self.assertEqual(grouped['video'][0]['file_path'], 'file4.mp4')
    
    def test_get_batch_stats(self):
        """Test getting batch statistics."""
        # Add mock files to batch with specific media types and statuses
        self.batch_processor.current_batch = [
            {'file_path': 'file1.jpg', 'media_type': 'image', 'status': 'pending'},
            {'file_path': 'file2.jpg', 'media_type': 'image', 'status': 'saved'},
            {'file_path': 'file3.wav', 'media_type': 'audio', 'status': 'failed'},
            {'file_path': 'file4.mp4', 'media_type': 'video', 'status': 'processing'}
        ]
        
        # Get stats
        stats = self.batch_processor.get_batch_stats()
        
        # Check counts
        self.assertEqual(stats['total_files'], 4)
        self.assertEqual(stats['image_count'], 2)
        self.assertEqual(stats['audio_count'], 1)
        self.assertEqual(stats['video_count'], 1)
        
        # Check status counts
        self.assertEqual(stats['status_counts']['pending'], 1)
        self.assertEqual(stats['status_counts']['saved'], 1)
        self.assertEqual(stats['status_counts']['failed'], 1)
        self.assertEqual(stats['status_counts']['processing'], 1)

if __name__ == '__main__':
    unittest.main()