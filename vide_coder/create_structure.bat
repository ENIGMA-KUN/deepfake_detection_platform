@echo off
echo Creating project structure...

:: Top-level files
type nul > README.md
type nul > requirements.txt
type nul > config.yaml
type nul > setup.py

:: Create directories
mkdir app\interface\components
mkdir app\interface\static\css
mkdir app\interface\static\js
mkdir app\interface\static\images
mkdir app\core
mkdir app\utils
mkdir detectors\image_detector
mkdir detectors\audio_detector
mkdir detectors\video_detector
mkdir models\cache
mkdir data\preprocessing
mkdir data\augmentation
mkdir data\samples\images
mkdir data\samples\audio
mkdir data\samples\videos
mkdir notebooks
mkdir reports\templates
mkdir reports\output
mkdir tests\test_data\images\real
mkdir tests\test_data\images\fake
mkdir tests\test_data\audio\real
mkdir tests\test_data\audio\fake
mkdir tests\test_data\videos\real
mkdir tests\test_data\videos\fake
mkdir docs

:: Files in the "app" directory
type nul > app\__init__.py
type nul > app\main.py

:: Files in "app\interface"
type nul > app\interface\__init__.py
type nul > app\interface\app.py
type nul > app\interface\routes.py

:: Files in "app\interface\components"
type nul > app\interface\components\__init__.py
type nul > app\interface\components\header.py
type nul > app\interface\components\footer.py
type nul > app\interface\components\upload_panel.py
type nul > app\interface\components\result_view.py
type nul > app\interface\components\settings.py

:: Files in "app\interface\static"
type nul > app\interface\static\css\main.css
type nul > app\interface\static\css\tron.css
type nul > app\interface\static\js\main.js
type nul > app\interface\static\js\visualizations.js
type nul > app\interface\static\images\logo.png
type nul > app\interface\static\images\favicon.ico

:: Files in "app\core"
type nul > app\core\__init__.py
type nul > app\core\processor.py
type nul > app\core\queue_manager.py
type nul > app\core\result_handler.py
type nul > app\core\workflow.py

:: Files in "app\utils"
type nul > app\utils\__init__.py
type nul > app\utils\file_handler.py
type nul > app\utils\logging_utils.py
type nul > app\utils\visualization.py

:: Files in the "detectors" directory
type nul > detectors\__init__.py
type nul > detectors\base_detector.py
type nul > detectors\detection_result.py
type nul > detectors\detector_factory.py
type nul > detectors\detector_utils.py

:: Files in "detectors\image_detector"
type nul > detectors\image_detector\__init__.py
type nul > detectors\image_detector\vit_detector.py
type nul > detectors\image_detector\ela_detector.py
type nul > detectors\image_detector\face_detector.py
type nul > detectors\image_detector\ensemble.py
type nul > detectors\image_detector\visualization.py

:: Files in "detectors\audio_detector"
type nul > detectors\audio_detector\__init__.py
type nul > detectors\audio_detector\wav2vec_detector.py
type nul > detectors\audio_detector\spectrogram_analyzer.py
type nul > detectors\audio_detector\visualization.py

:: Files in "detectors\video_detector"
type nul > detectors\video_detector\__init__.py
type nul > detectors\video_detector\genconvit.py
type nul > detectors\video_detector\frame_analyzer.py
type nul > detectors\video_detector\temporal_analysis.py
type nul > detectors\video_detector\visualization.py

:: Files in "models"
type nul > models\__init__.py
type nul > models\model_loader.py
type nul > models\model_config.py

:: Files in "models\cache"
type nul > models\cache\.gitignore
type nul > models\cache\README.md

:: Files in "data"
type nul > data\__init__.py

:: Files in "data\preprocessing"
type nul > data\preprocessing\__init__.py
type nul > data\preprocessing\image_prep.py
type nul > data\preprocessing\audio_prep.py
type nul > data\preprocessing\video_prep.py

:: Files in "data\augmentation"
type nul > data\augmentation\__init__.py
type nul > data\augmentation\augmenters.py

:: Files in "data\samples"
type nul > data\samples\.gitignore
type nul > data\samples\README.md

:: Files in "notebooks"
type nul > notebooks\model_evaluation.ipynb
type nul > notebooks\feature_visualization.ipynb
type nul > notebooks\experiment_tracking.ipynb

:: Files in "reports\templates" and "reports\output"
type nul > reports\templates\detailed_report.html
type nul > reports\templates\summary_report.html
type nul > reports\output\.gitignore

:: Files in "tests"
type nul > tests\__init__.py
type nul > tests\test_base_detector.py
type nul > tests\test_detector_factory.py
type nul > tests\test_detection_result.py
type nul > tests\test_image_detector.py
type nul > tests\test_audio_detector.py
type nul > tests\test_video_detector.py
type nul > tests\test_processing.py
type nul > tests\test_file_handler.py

:: Files in "tests\test_data"
type nul > tests\test_data\.gitignore
:: (The nested directories within tests\test_data are already created)

:: Files in "docs"
type nul > docs\usage.md
type nul > docs\detector_details.md
type nul > docs\model_descriptions.md
type nul > docs\api_reference.md
type nul > docs\development_guide.md

echo Project structure created successfully.
pause
