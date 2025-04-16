# detectors/model_version_manager.py
import os
import json
import requests
from pathlib import Path

class ModelVersionManager:
    """Manages model versions and updates for the detection models."""
    
    def __init__(self, config_path, cache_dir):
        """
        Initialize the model version manager.
        
        Args:
            config_path (str): Path to the model configuration file
            cache_dir (str): Path to the model cache directory
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.version_file = self.cache_dir / "version_info.json"
        self._load_version_info()
    
    def _load_version_info(self):
        """Load version information from the version file."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.version_info = json.load(f)
        else:
            self.version_info = {}
            self._save_version_info()
    
    def _save_version_info(self):
        """Save version information to the version file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.version_info, f, indent=2)
    
    def get_model_version(self, model_name):
        """
        Get the current version of a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Current version of the model
        """
        return self.version_info.get(model_name, {}).get('version', 'unknown')
    
    def check_for_updates(self, model_name, model_repo):
        """
        Check if updates are available for a model.
        
        Args:
            model_name (str): Name of the model
            model_repo (str): Hugging Face repository for the model
            
        Returns:
            bool: True if updates are available, False otherwise
        """
        try:
            # Fetch model info from Hugging Face API
            api_url = f"https://huggingface.co/api/models/{model_repo}"
            response = requests.get(api_url)
            if response.status_code == 200:
                model_info = response.json()
                latest_version = model_info.get('sha', None)
                
                if latest_version:
                    current_version = self.get_model_version(model_name)
                    if current_version == 'unknown' or current_version != latest_version:
                        return True
            return False
        except Exception as e:
            print(f"Error checking for updates: {e}")
            return False
    
    def update_model(self, model_name, model_repo, model_loader):
        """
        Update a model to the latest version.
        
        Args:
            model_name (str): Name of the model
            model_repo (str): Hugging Face repository for the model
            model_loader: Function to load the model
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Fetch model info from Hugging Face API
            api_url = f"https://huggingface.co/api/models/{model_repo}"
            response = requests.get(api_url)
            if response.status_code == 200:
                model_info = response.json()
                latest_version = model_info.get('sha', None)
                
                if latest_version:
                    # Load the updated model
                    model = model_loader(model_repo, force_reload=True)
                    
                    # Update version info
                    if model_name not in self.version_info:
                        self.version_info[model_name] = {}
                    
                    self.version_info[model_name]['version'] = latest_version
                    self.version_info[model_name]['last_updated'] = model_info.get('lastModified', '')
                    self._save_version_info()
                    
                    return True
            return False
        except Exception as e:
            print(f"Error updating model: {e}")
            return False