# tests/test_model_version_manager.py
import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from detectors.model_version_manager import ModelVersionManager

class TestModelVersionManager(unittest.TestCase):
    """Tests for the ModelVersionManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        self.cache_dir = os.path.join(self.temp_dir.name, "cache")
        
        # Create a simple config file
        with open(self.config_path, 'w') as f:
            json.dump({"models": {"test_model": {"repo": "test/model"}}}, f)
        
        self.version_manager = ModelVersionManager(self.config_path, self.cache_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_get_model_version_unknown(self):
        """Test getting version for unknown model."""
        version = self.version_manager.get_model_version("unknown_model")
        self.assertEqual(version, "unknown")
    
    @patch('requests.get')
    def test_check_for_updates_available(self, mock_get):
        """Test checking for updates when updates are available."""
        # Mock response from Hugging Face API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sha": "new_version"}
        mock_get.return_value = mock_response
        
        # Check for updates
        has_updates = self.version_manager.check_for_updates("test_model", "test/model")
        self.assertTrue(has_updates)
    
    @patch('requests.get')
    def test_update_model(self, mock_get):
        """Test updating a model."""
        # Mock response from Hugging Face API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sha": "new_version",
            "lastModified": "2023-01-01"
        }
        mock_get.return_value = mock_response
        
        # Mock model loader
        mock_loader = MagicMock()
        mock_loader.return_value = "model"
        
        # Update the model
        success = self.version_manager.update_model("test_model", "test/model", mock_loader)
        self.assertTrue(success)
        
        # Check that version info was updated
        self.assertEqual(self.version_manager.get_model_version("test_model"), "new_version")