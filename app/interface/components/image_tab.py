"""
Image tab component for the Deepfake Detection Platform UI.
"""
import os
import base64
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from typing import Dict, Any, List

from app.utils.visualization import VisualizationManager

def create_image_tab(app: dash.Dash, visualizer: VisualizationManager) -> html.Div:
    """
    Create the image analysis tab component.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
        
    Returns:
        Dash HTML Division containing the image tab layout
    """
    # Create image tab layout
    image_tab = html.Div([
        html.H3("Image Deepfake Detection", className="text-glow mb-4 text-center"),
        
        dbc.Row([
            # Upload and preview column
            dbc.Col([
                html.Div([
                    html.H5("Upload Image", className="mb-3"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            html.Div(className="loading-spinner", style={'display': 'none'}, id='image-upload-spinner'),
                            html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-3x mb-3 text-glow"),
                                html.P("Drag and Drop or Click to Select Image")
                            ], id='image-upload-text')
                        ]),
                        className='upload-container',
                        multiple=False
                    ),
                    html.Div(id='upload-image-error', className="text-danger mt-2"),
                    
                    html.Div([
                        html.H5("Image Preview", className="mt-4 mb-3"),
                        html.Div(id='image-preview-container', className="text-center"),
                        
                        html.Div([
                            dbc.Button("Analyze", id="analyze-image-button", 
                                      color="primary", className="mt-3 animate-glow",
                                      disabled=True)
                        ], className="text-center")
                    ])
                ], className="card h-100")
            ], md=6),
            
            # Results column
            dbc.Col([
                html.Div([
                    html.H5("Detection Results", className="mb-3"),
                    
                    # Initial state - no results
                    html.Div([
                        html.P("Upload and analyze an image to see results.", className="text-center py-5"),
                    ], id='no-image-results', style={'display': 'block'}),
                    
                    # Loading state
                    html.Div([
                        html.Div(className="loading-container", children=[
                            html.Div(className="loading-spinner"),
                            html.P("Analyzing Image...", className="loading-text")
                        ])
                    ], id='image-analyzing', style={'display': 'none'}),
                    
                    # Results state
                    html.Div([
                        # Verdict header
                        html.Div([
                            html.H4("Verdict:", className="results-title"),
                            html.Div(id='image-verdict', className="results-verdict")
                        ], className="results-header"),
                        
                        # Confidence gauge
                        html.Div([
                            html.H5("Detection Confidence"),
                            html.Div([
                                html.Div(className="confidence-meter", children=[
                                    html.Div(id='image-confidence-fill', className="confidence-fill"),
                                    html.Div(className="confidence-threshold")
                                ]),
                                html.Div(id='image-confidence-text', className="text-center")
                            ])
                        ], className="mb-4"),
                        
                        # Visualization tabs
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id='image-detection-visualization', className="mt-3 text-center")
                            ], label="Detection View", tab_id="detection-tab"),
                            
                            dbc.Tab([
                                html.Div(id='image-heatmap-visualization', className="mt-3 text-center")
                            ], label="Attention Heatmap", tab_id="heatmap-tab"),
                            
                            dbc.Tab([
                                html.Div(id='image-details-container', className="mt-3")
                            ], label="Detailed Analysis", tab_id="details-tab")
                        ], id="image-result-tabs", active_tab="detection-tab", className="mt-3"),
                        
                        # Export section
                        html.Div([
                            html.H5("Export Results", className="mt-4"),
                            dbc.Button("Save Report", id="save-image-report", 
                                      color="primary", className="me-2"),
                            dbc.Button("Download Visualization", id="download-image-vis", 
                                      color="secondary")
                        ], className="mt-4 pt-3 border-top")
                        
                    ], id='image-results', style={'display': 'none'})
                    
                ], className="card h-100")
            ], md=6)
        ])
    ], id="image-tab-content")
    
    # Register callbacks for the image tab
    register_image_callbacks(app, visualizer)
    
    return image_tab

def register_image_callbacks(app: dash.Dash, visualizer: VisualizationManager):
    """
    Register all callbacks for the image tab.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
    """
    # Callback for image upload
    @app.callback(
        [Output('image-preview-container', 'children'),
         Output('analyze-image-button', 'disabled'),
         Output('upload-image-error', 'children'),
         Output('image-upload-spinner', 'style'),
         Output('image-upload-text', 'style')],
        [Input('upload-image', 'contents')],
        [State('upload-image', 'filename')]
    )
    def update_image_preview(content, filename):
        if content is None:
            return None, True, "", {'display': 'none'}, {'display': 'block'}
        
        try:
            # Show loading state
            spinner_style = {'display': 'block'}
            text_style = {'display': 'none'}
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in valid_extensions:
                return None, True, f"Invalid file type. Please upload an image ({', '.join(valid_extensions)})", {'display': 'none'}, {'display': 'block'}
            
            # Display the uploaded image
            image_div = html.Div([
                html.Img(src=content, style={'max-width': '100%', 'max-height': '300px', 'margin': 'auto'})
            ])
            
            # Store the image content in a hidden div for later use
            app.uploaded_image_content = content
            app.uploaded_image_filename = filename
            
            return image_div, False, "", {'display': 'none'}, {'display': 'block'}
            
        except Exception as e:
            return None, True, f"Error: {str(e)}", {'display': 'none'}, {'display': 'block'}
    
    # Callback for analyze button
    @app.callback(
        [Output('no-image-results', 'style'),
         Output('image-analyzing', 'style'),
         Output('image-results', 'style'),
         Output('image-verdict', 'children'),
         Output('image-verdict', 'className'),
         Output('image-confidence-fill', 'style'),
         Output('image-confidence-text', 'children'),
         Output('image-detection-visualization', 'children'),
         Output('image-heatmap-visualization', 'children'),
         Output('image-details-container', 'children')],
        [Input('analyze-image-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def analyze_image(n_clicks):
        if n_clicks is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None
        
        # Get the processor from the app instance
        processor = app.processor
        
        if not hasattr(app, 'uploaded_image_content') or app.uploaded_image_content is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None
        
        try:
            # Show analyzing state
            no_results_style = {'display': 'none'}
            analyzing_style = {'display': 'block'}
            results_style = {'display': 'none'}
            
            # Decode image
            content_type, content_string = app.uploaded_image_content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create a temporary file with the uploaded content
            temp_dir = app.config['general'].get('temp_dir', './temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, app.uploaded_image_filename)
            
            with open(temp_path, "wb") as f:
                f.write(decoded)
            
            # Process the image with the detector
            result = processor.detect_media(temp_path, 'image')
            
            # Determine verdict info
            is_deepfake = result['is_deepfake']
            confidence = result['confidence']
            
            # Configure verdict elements
            verdict_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC"
            verdict_class = "results-verdict verdict-deepfake" if is_deepfake else "results-verdict verdict-authentic"
            
            # Configure confidence bar
            confidence_bar_style = {'width': f"{confidence * 100}%"}
            confidence_text = f"Confidence: {confidence:.2f} ({confidence * 100:.1f}%)"
            
            # Create visualizations
            # 1. Detection overlay
            img = Image.open(temp_path)
            detection_overlay = visualizer.create_detection_overlay(img, result)
            
            # Convert to base64 for display
            buffer = BytesIO()
            detection_overlay.save(buffer, format="PNG")
            buffer.seek(0)
            detection_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            detection_img = html.Img(
                src=f"data:image/png;base64,{detection_b64}",
                style={'max-width': '100%', 'max-height': '400px'}
            )
            
            # 2. Attention heatmap 
            attention_map = None
            if "details" in result and "face_results" in result["details"]:
                for face in result["details"]["face_results"]:
                    if "attention_map" in face:
                        attention_map = np.array(face["attention_map"])
                        break
                        
            if attention_map is not None:
                heatmap_overlay = visualizer.create_heatmap_overlay(img, attention_map)
                
                buffer = BytesIO()
                heatmap_overlay.save(buffer, format="PNG")
                buffer.seek(0)
                heatmap_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                heatmap_img = html.Img(
                    src=f"data:image/png;base64,{heatmap_b64}",
                    style={'max-width': '100%', 'max-height': '400px'}
                )
            else:
                heatmap_img = html.Div([
                    html.P("Attention heatmap not available for this image.")
                ])
            
            # 3. Detailed analysis
            if "details" in result:
                faces_detected = result["details"].get("faces_detected", 0)
                
                details_div = html.Div([
                    html.H5("Analysis Details"),
                    
                    html.Table([
                        html.Tr([
                            html.Td("Faces Detected:"),
                            html.Td(str(faces_detected))
                        ]),
                        html.Tr([
                            html.Td("Model Used:"),
                            html.Td(result.get("model", "ViT"))
                        ]),
                        html.Tr([
                            html.Td("Confidence Threshold:"),
                            html.Td(f"{result.get('threshold', 0.5):.2f}")
                        ]),
                        html.Tr([
                            html.Td("Analysis Time:"),
                            html.Td(f"{result.get('analysis_time', 0):.3f} seconds")
                        ])
                    ], className="table table-sm")
                ])
                
                # If face information is available, add face details
                if "face_results" in result["details"] and faces_detected > 0:
                    face_details = []
                    
                    for i, face in enumerate(result["details"]["face_results"]):
                        face_conf = face.get("confidence", 0)
                        face_div = html.Div([
                            html.H6(f"Face #{i+1}"),
                            html.Div([
                                html.Div("Confidence: ", style={"display": "inline-block", "marginRight": "10px"}),
                                html.Div(f"{face_conf:.2f}", 
                                        style={
                                            "display": "inline-block", 
                                            "color": visualizer._get_confidence_color(face_conf)
                                        })
                            ]),
                        ], className="mb-3")
                        face_details.append(face_div)
                    
                    details_div.children.append(
                        html.Div([
                            html.H5("Face Analysis", className="mt-4"),
                            html.Div(face_details)
                        ])
                    )
            else:
                details_div = html.Div([
                    html.P("No detailed analysis available for this image.")
                ])
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Show results
            return (no_results_style, 
                   {'display': 'none'}, 
                   {'display': 'block'}, 
                   verdict_text, 
                   verdict_class, 
                   confidence_bar_style, 
                   confidence_text, 
                   detection_img, 
                   heatmap_img, 
                   details_div)
            
        except Exception as e:
            # Show error
            error_div = html.Div([
                html.H5("Error analyzing image:"),
                html.P(str(e))
            ])
            
            return ({'display': 'none'}, 
                   {'display': 'none'}, 
                   {'display': 'block'}, 
                   "ERROR", 
                   "results-verdict", 
                   {}, 
                   "", 
                   error_div, 
                   None, 
                   None)
    
    # Callback for download visualization button
    @app.callback(
        Output('download-image-vis', 'href'),
        [Input('image-result-tabs', 'active_tab')],
        [State('image-detection-visualization', 'children'),
         State('image-heatmap-visualization', 'children')],
        prevent_initial_call=True
    )
    def prepare_image_download(active_tab, detection_vis, heatmap_vis):
        if active_tab == "detection-tab" and detection_vis is not None:
            # Extract the base64 string from the img src
            try:
                src = detection_vis['props']['src']
                return src
            except:
                return ""
                
        elif active_tab == "heatmap-tab" and heatmap_vis is not None:
            try:
                src = heatmap_vis['props']['src']
                return src
            except:
                return ""
                
        return ""
