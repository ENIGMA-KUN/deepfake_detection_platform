# Deepfake Detection Platform – Comprehensive Fix & Feature Checklist

> Last Updated: 2025‑04‑22 14:45 PDT

## Legend
- [x] = Completed
- [ ] = Pending / In‑Progress
- [P] = Premium Feature (requires API key or special access)

---

### 1. Repository & Infrastructure
- [x] Create and apply `.gitignore`
- [ ] Automate virtual‑env / requirements installation script
- [ ] CI workflow: lint + unit tests on every push

### 2. Core Detection Modules
- [x] Backward‑compatibility key `is_deepfake` in results (processor.py)
- [x] Fix `ViTImageDetector` AttributeError (`pooler_output` ➜ `logits`)
- [x] Implement `detect` method in `AudioEnsembleDetector`
- [x] Integrate **TimeSformer** detector
- [x] Add missing **VideoSwin** detector
- [x] Integrate **SlowFast** detector
- [x] Integrate **X3D** detector
- [x] Download real model weights (replace mock implementations)
- [x] Ensure automatic model download / cache handling for all detectors
- [x] Unify confidence normalization across media types
- [x] Expose model selection parameters from UI → backend (`processor.detect_media`)
- [x] Add API key configuration for premium model access

### 3. Ensemble & Singularity Modes
- [x] Basic ensemble classes for image, audio, video
- [x] Adaptive weighting stubs based on content characteristics
- [x] Replace mock implementations with real model outputs
- [ ] Finish audio adaptive weighting by frequency bands
- [ ] Finish video adaptive weighting (frame vs. temporal vs. AV‑sync)
- [ ] Implement validation‑driven `calibrate_weights()` routine
- [ ] Cache ensemble results to avoid re‑processing duplicate media

### 4. Analysis Modes
- [x] Standard Analysis (fast, minimal metrics)
- [P][x] Deep Analysis (full model set with premium models)
- [P][x] Temporal Oracle™ Mode (video, includes timeline & cross‑modal sync)
- [x] Wire mode selector from UI to backend logic
- [x] Add UI indicators for premium features requiring API keys

### 5. UI / Visualization
#### General UI
- [x] Remove top nav bar & match Tron theme
- [x] Center platform title
- [x] Fix white button colour contrast
- [x] Add "Premium API Required" indicators for unavailable models

#### Media Tabs
- **Image**
  - [x] Overlay green face bounding boxes
  - [x] Heat‑map manipulation visualization (colormap + opacity slider)
  - [x] Model dropdown (ViT, DeiT, BEIT, Swin, Ensemble)
  - [x] Tag premium models in UI
- **Audio**
  - [x] Spectrogram visualization with highlighted anomaly regions
  - [x] Model dropdown (Wav2Vec2, XLSR‑Mamba, Ensemble)
  - [x] Tag premium models in UI
- **Video**
  - [x] Frame timeline graph (confidence vs. time)
  - [x] Preview top suspicious frames
  - [ ] Fix `dash.html.Svg` error when rendering frame boxes
  - [x] Model dropdown (GenConViT, TimeSformer, etc.)
  - [x] Tag premium models in UI


#### Detailed Analysis Report Panel
- [x] Build three‑column grid:
  1. Detection Metrics
  2. Model Contributions
  3. Export Options
- [x] Show metrics:
  - Singularity Mode (e.g., *Visual Sentinel*)
  - Detection Time (ms)
  - Manipulation Type (e.g., Face Swap)
  - Model Consensus (n/N)
- [x] Export PDF report button
- [x] Download JSON metadata button


### 6. Error Handling & Fallbacks
- [x] Mock MTCNN when `facenet_pytorch` unavailable
- [x] Graceful fallback when optional video models missing (import errors)
- [x] User‑friendly error toast messages in UI
- [x] Add API key validation and error handling
- [x] Add "Premium Feature" notices instead of errors for missing API keys

## Component ID Reference

This section tracks all component IDs and naming patterns used across the codebase to ensure consistency and prevent callback errors.

### 1. Tab IDs
- `tab-home` - Home/welcome tab
- `tab-image` - Image analysis tab
- `tab-audio` - Audio analysis tab
- `tab-video` - Video analysis tab
- `tab-reports` - Reports and history tab
- `tab-settings` - Settings tab

### 2. Upload Components
- `upload-image` - Image upload component
- `upload-audio` - Audio upload component
- `upload-video` - Video upload component
- `upload-output` - Hidden component storing upload path

### 3. Analysis Buttons
- `analyze-image-button` - Image analysis button
- `analyze-audio-button` - Audio analysis button
- `analyze-video-button` - Video analysis button
- `clear-image-button` - Clear image button
- `clear-audio-button` - Clear audio button
- `clear-video-button` - Clear video button

### 4. Model Selection Dropdowns
- `image-model-dropdown` - Image model dropdown
- `audio-model-dropdown` - Audio model dropdown
- `video-model-dropdown` - Video model dropdown
- `model-dropdown-output` - Hidden component storing selected model

### 5. Threshold Sliders
- `image-threshold-slider` - Image confidence threshold slider
- `audio-threshold-slider` - Audio confidence threshold slider
- `video-threshold-slider` - Video confidence threshold slider
- `threshold-output` - Hidden component storing threshold value

### 6. Results Containers
- `image-result-container` - Container for image analysis results
- `audio-result-container` - Container for audio analysis results
- `video-result-container` - Container for video analysis results
- `result-confidence-gauge` - Confidence gauge component
- `result-authentic-label` - Label for "Authentic" result
- `result-deepfake-label` - Label for "Deepfake" result
- `result-timestamp` - Analysis timestamp component
- `result-analysis-time` - Analysis time component

### 7. Visualization Components
- `image-heatmap` - Image attention heatmap
- `audio-spectrogram` - Audio spectrogram visualization
- `video-frame-grid` - Grid of analyzed video frames
- `video-timeline` - Video timeline visualization
- `face-bounding-boxes` - Face bounding box overlay
- `anomaly-regions` - Anomaly region indicators

### 8. Export/Report Components
- `export-report-button` - Export report button
- `download-json-button` - Download JSON results button
- `report-preview` - Report preview component

### 9. State Management
- `app-state-store` - Hidden component for app state
- `analysis-status` - Analysis status indicator
- `error-toast` - Error notification component
- `success-toast` - Success notification component

### 10. Settings Components
- `api-key-input` - API key input field
- `api-key-save-button` - Save API key button
- `theme-toggle` - Theme toggle switch
- `cache-clear-button` - Clear cache button

### Callback Connection Patterns

1. **Input → Processing → Output Pattern**:
   ```python
   @app.callback(
       Output('result-container', 'children'),
       Input('analyze-button', 'n_clicks'),
       State('upload-component', 'contents'),
       prevent_initial_call=True
   )
   ```

2. **Multi-Output Pattern**:
   ```python
   @app.callback(
       Output('result-confidence', 'value'),
       Output('result-label', 'children'),
       Input('app-state-store', 'data')
   )
   ```

3. **Tab Visibility Pattern**:
   ```python
   @app.callback(
       Output('tab-content', 'children'),
       Input('tabs', 'value')
   )
   ```

4. **Error Handling Pattern**:
   ```python
   @app.callback(
       Output('result-container', 'children'),
       Output('error-toast', 'is_open'),
       Output('error-toast', 'children'),
       Input('analyze-button', 'n_clicks'),
       # ...states...
       prevent_initial_call=True
   )
   def process_media(n_clicks, *args):
       if n_clicks is None:
           return dash.no_update, dash.no_update, dash.no_update
       try:
           # Processing logic
           return result, False, ""
       except Exception as e:
           return dash.no_update, True, str(e)
   ```

## Component ID Validation Checklist

Use this checklist when adding or modifying components and callbacks to ensure consistency:

- [ ] All component IDs follow naming convention: `{media-type}-{component-purpose}[-{qualifier}]`
- [ ] All callback Outputs reference existing component IDs
- [ ] All callback Inputs reference existing component IDs
- [ ] All callback States reference existing component IDs
- [ ] Component properties in callbacks match available properties for that component
- [ ] Prevent_initial_call is set appropriately for each callback
- [ ] Error handling is implemented for all user-triggered callbacks
- [ ] Component IDs are unique across the entire application
- [ ] Component store keys match component IDs where applicable

## Known ID Mismatches to Fix

The following component ID mismatches have been identified using the validation tool and need to be fixed:

### Missing Components (Referenced but not defined)
- [ ] `report-export-status` - Add component definition
- [ ] `report-item-rep_001`, `report-item-rep_002`, `report-item-rep_003` - Add dynamic ID generation for report items
- [ ] `timesformer-technical-collapse`, `vit-technical-collapse`, `wav2vec-technical-collapse` - Add technical detail components
- [ ] `toggle-timesformer-details`, `toggle-vit-details`, `toggle-wav2vec-details` - Add toggle controls for technical details

### Inconsistent Naming Patterns
- [ ] Model selection: `dropdown-*-model` vs `*-model-dropdown` - Standardize to `*-model-dropdown`
- [ ] Results containers: `*-results-container` vs `*-result-container` - Standardize to `*-result-container`
- [ ] Analysis buttons: `analyze-img-btn` vs `analyze-image-button` - Standardize to `analyze-*-button`
- [ ] Upload outputs: `output-*-upload` vs `*-upload-output` - Standardize to `*-upload-output`

### Category-Specific Issues
1. **Upload Components**:
   - [ ] `upload-image-data` → `upload-image`
   - [ ] `upload-audio-component` → `upload-audio`
   - [ ] `upload-video-component` → `upload-video`

2. **Result Containers**:
   - [ ] `image-result` → `image-result-container`
   - [ ] `audio-result-div` → `audio-result-container`
   - [ ] `video-result-display` → `video-result-container`

3. **Analysis Controls**:
   - [ ] `analyze-img-btn` → `analyze-image-button`
   - [ ] `audio-analyze-button` → `analyze-audio-button`
   - [ ] `video-analyze-button` → `analyze-video-button`

4. **Model Dropdowns**:
   - [ ] `dropdown-image-model` → `image-model-dropdown`
   - [ ] `dropdown-audio-model` → `audio-model-dropdown`
   - [ ] `dropdown-video-model` → `video-model-dropdown`

5. **Threshold Sliders**:
   - [ ] `image-confidence-threshold-slider` → `image-threshold-slider`
   - [ ] `audio-confidence-threshold-slider` → `audio-threshold-slider`
   - [ ] `video-confidence-threshold-slider` → `video-threshold-slider`

6. **State Stores**:
   - [ ] `uploaded-image-store` → `image-upload-store`
   - [ ] `uploaded-audio-store` → `audio-upload-store`
   - [ ] `uploaded-video-store` → `video-upload-store`

7. **Technical Detail Components**:
   - [ ] Add `*-technical-collapse` components to component_ids.py
   - [ ] Add `toggle-*-details` components to component_ids.py

## Component ID Migration Plan

### Phase 1: Update Central Constants (Immediate)
1. [ ] Add all missing component IDs to `app/interface/constants/component_ids.py`
2. [ ] Complete the legacy mapping in `LEGACY_COMPONENT_IDS` for all identified mismatches
3. [ ] Add technical detail component IDs for models

### Phase 2: Component Definition Updates (Next 24h)
1. [ ] Update component definitions in all UI files to use constant imports
2. [ ] Prioritize update order:
   - Main layout components
   - Tab components
   - Media-specific components

### Phase 3: Callback Updates (Next 48h)
1. [ ] Update all callbacks to reference constants rather than string literals
2. [ ] Test callbacks after each update to ensure functionality is maintained

### Phase 4: Validation & Cleanup (Final)
1. [ ] Run validation tool to confirm all mismatches are resolved
2. [ ] Remove unused component definitions
3. [ ] Document any exceptions or special cases

## Component ID Update Tracking

| Component Type | Total Count | Updated | Remaining |
|--------------|------------|---------|-----------|
| Tab IDs | 6 | 0 | 6 |
| Upload Components | 4 | 0 | 4 |
| Analysis Buttons | 6 | 0 | 6 |
| Model Dropdowns | 4 | 0 | 4 |
| Threshold Sliders | 4 | 0 | 4 |
| Result Containers | 8 | 0 | 8 |
| Visualization | 6 | 0 | 6 |
| Report Components | 20+ | 0 | 20+ |
| State Management | 5 | 0 | 5 |
| Settings Components | 4 | 0 | 4 |
| Total | ~70 | 0 | ~70 |