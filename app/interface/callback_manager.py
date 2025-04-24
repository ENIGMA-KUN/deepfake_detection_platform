"""
Callback manager for the Deepfake Detection Platform UI.
Centralizes and organizes all app callbacks in one place.
"""
import os
import base64
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
import logging

from app.utils.visualization import VisualizationManager, fig_to_base64

def register_callbacks(app):
    """
    Register all callbacks for the Dash application.
    
    Args:
        app: Dash application instance
    """
    # Register navigation callbacks
    register_navigation_callbacks(app)
    
    # Register tab-specific callbacks
    register_image_callbacks(app)
    register_audio_callbacks(app)
    register_video_callbacks(app)
    # Reports callbacks are already registered within the reports_tab component.
    # Avoid duplicate callback outputs by not calling a second registration here.
    # register_reports_callbacks(app)

def register_navigation_callbacks(app):
    """
    Register callbacks for main navigation.
    
    Args:
        app: Dash application instance
    """
    # Callback for tab switching from home screen buttons
    @app.callback(
        Output("tabs", "value"),
        [Input("home-image-button", "n_clicks"),
         Input("home-audio-button", "n_clicks"),
         Input("home-video-button", "n_clicks")],
        [State("tabs", "value")],
        prevent_initial_call=True
    )
    def switch_tab(home_image_clicks, home_audio_clicks, home_video_clicks, current_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_tab
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "home-image-button":
            return "image-tab"
        elif button_id == "home-audio-button":
            return "audio-tab"
        elif button_id == "home-video-button":
            return "video-tab"
            
        return current_tab

def register_image_callbacks(app):
    """
    Register callbacks for the image analysis tab.
    
    Args:
        app: Dash application instance
    """
    # Callback for toggling the about section
    @app.callback(
        [Output("image-about-collapse", "is_open"),
         Output("toggle-image-about", "children")],
        [Input("toggle-image-about", "n_clicks")],
        [State("image-about-collapse", "is_open")]
    )
    def toggle_image_about(n, is_open):
        if n:
            return not is_open, "Hide About Image Detection" if not is_open else "Show About Image Detection"
        return is_open, "Show About Image Detection"
    
    # Callback for toggling technical details
    @app.callback(
        [Output("vit-technical-collapse", "is_open"),
         Output("toggle-vit-details", "children")],
        [Input("toggle-vit-details", "n_clicks")],
        [State("vit-technical-collapse", "is_open")]
    )
    def toggle_vit_details(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"
    
    # Callback for image upload and preview
    @app.callback(
        [Output('output-image-upload', 'children'),
         Output('analyze-image-button', 'disabled')],
        [Input('upload-image', 'contents'),
         Input('upload-image', 'filename')],
        [State('upload-image', 'last_modified')]
    )
    def update_image_preview(content, filename, date):
        logger = logging.getLogger(__name__)
        logger.info(f"update_image_preview triggered | filename={filename} | has_content={content is not None}")
        print(f"update_image_preview | filename={filename} | has_content={content is not None}")
        
        if content is None:
            return None, True
        
        try:
            # Show basic preview (without heavy validation first for debugging)
            preview = html.Div([
                html.Img(src=content, style={'maxWidth': '100%', 'maxHeight': '200px'}, className="d-block mx-auto"),
                html.P(filename, className="text-center mt-2")
            ])
            
            # Cache uploaded image in app ctx
            app.uploaded_image = {
                'content': content,
                'filename': filename,
                'date': date
            }
            
            return preview, False
        
        except Exception as e:
            logger.exception("Error in update_image_preview")
            return html.Div([
                html.P(f"Error processing image: {str(e)}", className="text-danger")
            ]), True
    
    # Debug helper – emit console log for ANY change on upload-image
    @app.callback(Output('uploaded-image-store', 'data'),
                  [Input('upload-image', 'contents'),
                   Input('upload-image', 'filename')],
                  prevent_initial_call=True)
    def _debug_upload_signal(c, f):
        print('DEBUG UPLOAD SIGNAL', bool(c), f)
        return None
    
    # Callback for image analysis
    @app.callback(
        Output('image-results-container', 'children'),
        [Input('analyze-image-button', 'n_clicks')],
        [State('image-model-select', 'value'),
         State('image-confidence-threshold-slider', 'value'),
         State('image-analysis-mode', 'value')],
        prevent_initial_call=True
    )
    def analyze_image(n_clicks, model_name, confidence_threshold, analysis_mode):
        if n_clicks is None or not hasattr(app, 'uploaded_image'):
            return html.P("Upload and analyze an image to see results.")
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Analyze image called: model={model_name}, threshold={confidence_threshold}, mode={analysis_mode}")
            
            # Show loading state first
            loading_div = html.Div([
                html.Div(className="d-flex justify-content-center", children=[
                    dbc.Spinner(size="lg", color="primary", type="border"),
                ]),
                html.P("Analyzing image...", className="text-center mt-3")
            ])
            
            # Get the image data
            image_data = app.uploaded_image
            content = image_data['content']
            filename = image_data['filename']
            
            # Decode the base64 image content
            content_type, content_string = content.split(',', 1)
            decoded_content = base64.b64decode(content_string)
            
            # Save image to a temporary file
            temp_file_path = os.path.join(app._processor.temp_dir, filename)
            with open(temp_file_path, 'wb') as f:
                f.write(decoded_content)
            
            # Configure model parameters based on the selected options
            model_params = {
                'model_name': model_name,
                'confidence_threshold': confidence_threshold
            }
            
            # Apply analysis mode settings
            if analysis_mode == "deep" or analysis_mode == "singularity":
                # For deep analysis, always use the ensemble with all models
                model_params['model_name'] = 'ensemble'
                
                # For singularity mode, enable advanced features
                if analysis_mode == "singularity":
                    model_params['enable_singularity'] = True
            
            logger.info(f"Calling detect_media with params: {model_params}")
            
            # Call the backend processor to analyze the image
            result = app._processor.detect_media(temp_file_path, 'image', model_params)
            logger.info(f"Detection result: {result}")
            
            # Get results
            is_deepfake = result.get('is_deepfake', False)
            confidence = result.get('confidence', 0.5) * 100  # Convert to percentage
            
            # Get details from result metadata
            metadata = result.get('metadata', {})
            details = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) and key not in ['timestamp']:
                    details[key] = value
            
            # Create the visualization if available
            visualization = None
            if 'visualization' in result:
                visualization = html.Img(
                    src=f"data:image/png;base64,{result['visualization']}",
                    style={'max-width': '100%'},
                    className="d-block mx-auto mb-3 mt-3"
                )
            
            # Get model consensus information
            model_consensus = result.get('model_consensus', '0/0')
            logger.info(f"Model consensus: {model_consensus}")
            
            # Create the result div
            result_div = html.Div([
                html.H5("Analysis Complete!", className="text-success text-center mb-3"),
                
                # Verdict
                html.Div([
                    html.H6("Verdict:", className="d-inline me-2"),
                    html.Span("DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC MEDIA", 
                             className=f"badge {'bg-danger' if is_deepfake else 'bg-success'} px-3 py-2")
                ], className="mb-3 text-center"),
                
                # Confidence
                html.Div([
                    html.P(f"Confidence: {confidence:.1f}%", className="text-center mb-2"),
                    dbc.Progress(
                        value=confidence, 
                        className="mb-3", 
                        color="danger" if is_deepfake else "success", 
                        striped=True, 
                        animated=True
                    )
                ]),
                
                # Add visualization if available
                visualization if visualization else None,
                
                # Details
                html.Div([
                    html.H6("Analysis Details:", className="mb-2"),
                    html.Ul([
                        html.Li(f"{key}: {value}") for key, value in details.items()
                    ] if details else [html.Li("No detailed metadata available")]),
                    html.P(f"Used model: {result.get('model', 'unknown')}" + 
                          (f" (ensemble of {result.get('model_count', 0)} models)" if result.get('ensemble', False) else "") + 
                          f", Threshold: {result.get('threshold', 0.5):.2f}"),
                    html.P(f"Analysis time: {result.get('analysis_time', 0):.2f} seconds"),
                    html.P(f"Analysis mode: {analysis_mode.title()}"),
                    html.P(f"Model consensus: {model_consensus}")
                ]),
                
                # Report button
                html.Div([
                    dbc.Button("View Full Report", 
                              id="image-view-report", 
                              color="primary", 
                              className="w-100 mt-3")
                ])
            ])
            
            # Save the result for the report and detailed analysis panel
            if not hasattr(app, 'analysis_results'):
                app.analysis_results = {}
            app.analysis_results['image'] = result
            
            return result_div
            
        except Exception as e:
            import traceback
            logger.error(f"Error analyzing image: {str(e)}")
            logger.error(traceback.format_exc())
            return html.Div([
                html.H5("Error analyzing image", className="text-danger"),
                html.P(str(e), className="text-danger"),
                html.Pre(traceback.format_exc(), className="text-danger small")
            ])

def register_audio_callbacks(app):
    """
    Register callbacks for the audio analysis tab.
    
    Args:
        app: Dash application instance
    """
    # Callback for toggling the about section
    @app.callback(
        [Output("audio-about-collapse", "is_open"),
         Output("toggle-audio-about", "children")],
        [Input("toggle-audio-about", "n_clicks")],
        [State("audio-about-collapse", "is_open")]
    )
    def toggle_audio_about(n, is_open):
        if n:
            return not is_open, "Hide About Audio Detection" if not is_open else "Show About Audio Detection"
        return is_open, "Show About Audio Detection"
    
    # Callback for toggling technical details
    @app.callback(
        [Output("wav2vec-technical-collapse", "is_open"),
         Output("toggle-wav2vec-details", "children")],
        [Input("toggle-wav2vec-details", "n_clicks")],
        [State("wav2vec-technical-collapse", "is_open")]
    )
    def toggle_wav2vec_details(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"
    
    # Callback for audio upload and preview
    @app.callback(
        [Output('output-audio-upload', 'children'),
         Output('analyze-audio-button', 'disabled')],
        [Input('upload-audio', 'contents')],
        [State('upload-audio', 'filename'),
         State('upload-audio', 'last_modified')]
    )
    def update_audio_preview(content, filename, date):
        if content is None:
            return None, True
        
        try:
            # Check file extension
            valid_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
            ext = os.path.splitext(filename)[1].lower() if filename else ''
            
            if ext not in valid_extensions:
                return html.Div([
                    html.P(f"Invalid file type. Please upload an audio file ({', '.join(valid_extensions)})",
                         className="text-danger")
                ]), True
            
            # Display audio player for preview
            audio_div = html.Div([
                html.Audio(src=content, controls=True, className="w-100"),
                html.P(filename, className="text-center mt-2")
            ])
            
            # Store the audio content in the app object
            app.uploaded_audio = {
                'content': content,
                'filename': filename,
                'date': date
            }
            
            return audio_div, False
            
        except Exception as e:
            return html.Div([
                html.P(f"Error processing audio: {str(e)}",
                     className="text-danger")
            ]), True
    
    # Callback for audio analysis
    @app.callback(
        Output('audio-results-container', 'children'),
        [Input('analyze-audio-button', 'n_clicks')],
        [State('audio-model-select', 'value'),
         State('audio-confidence-threshold-slider', 'value')],
        prevent_initial_call=True
    )
    def analyze_audio(n_clicks, model_name, confidence_threshold):
        if n_clicks is None or not hasattr(app, 'uploaded_audio'):
            return html.P("Upload and analyze an audio file to see results.")
        
        try:
            # Show loading state first
            loading_div = html.Div([
                html.Div(className="d-flex justify-content-center", children=[
                    dbc.Spinner(size="lg", color="primary", type="border"),
                ]),
                html.P("Analyzing audio...", className="text-center mt-3")
            ])
            
            # Get the audio data
            audio_data = app.uploaded_audio
            content = audio_data['content']
            filename = audio_data['filename']
            
            # Decode the base64 audio content
            content_type, content_string = content.split(',', 1)
            decoded_content = base64.b64decode(content_string)
            
            # Save audio to a temporary file
            temp_file_path = os.path.join(app._processor.temp_dir, filename)
            with open(temp_file_path, 'wb') as f:
                f.write(decoded_content)
            
            # Process the audio using the MediaProcessor
            model_params = {
                'model_name': model_name,
                'confidence_threshold': confidence_threshold  # Already in decimal format (0.5-0.95)
            }
            
            # Call the backend processor to analyze the audio
            result = app._processor.detect_media(temp_file_path, 'audio', model_params)
            
            # Get results
            is_deepfake = result['is_deepfake']
            confidence = result['confidence'] * 100  # Convert to percentage
            details = result.get('details', {})
            
            # Create the visualization if available (e.g. spectrogram)
            visualization = None
            if 'visualization' in result:
                visualization = html.Img(
                    src=f"data:image/png;base64,{result['visualization']}",
                    style={'max-width': '100%'},
                    className="d-block mx-auto mb-3 mt-3"
                )
            
            # Create the result div
            result_div = html.Div([
                html.H5("Analysis Complete!", className="text-success text-center mb-3"),
                
                # Verdict
                html.Div([
                    html.H6("Verdict:", className="d-inline me-2"),
                    html.Span("DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC AUDIO", 
                             className=f"badge {'bg-danger' if is_deepfake else 'bg-success'} px-3 py-2")
                ], className="mb-3 text-center"),
                
                # Confidence
                html.Div([
                    html.P(f"Confidence: {confidence:.1f}%", className="text-center mb-2"),
                    dbc.Progress(
                        value=confidence, 
                        className="mb-3", 
                        color="danger" if is_deepfake else "success", 
                        striped=True, 
                        animated=True
                    )
                ]),
                
                # Add visualization if available
                visualization if visualization else None,
                
                # Details
                html.Div([
                    html.H6("Analysis Details:", className="mb-2"),
                    html.Ul([
                        html.Li(f"{key}: {value}") for key, value in details.items()
                    ]),
                    html.P(f"Used model: {result['model']}, Threshold: {result['threshold']:.2f}"),
                    html.P(f"Analysis time: {result['analysis_time']:.2f} seconds")
                ]),
                
                # Report button
                html.Div([
                    dbc.Button("View Full Report", 
                              id="audio-view-report", 
                              color="primary", 
                              className="w-100 mt-3")
                ])
            ])
            
            # Save the result for the report
            if not hasattr(app, 'analysis_results'):
                app.analysis_results = {}
            app.analysis_results['audio'] = result
            
            return result_div
            
        except Exception as e:
            import traceback
            return html.Div([
                html.H5("Error analyzing audio"),
                html.P(str(e), className="text-danger")
            ])

def register_video_callbacks(app):
    """
    Register callbacks for the video analysis tab.
    
    Args:
        app: Dash application instance
    """
    # Callback for toggling the about section
    @app.callback(
        [Output("video-about-collapse", "is_open"),
         Output("toggle-video-about", "children")],
        [Input("toggle-video-about", "n_clicks")],
        [State("video-about-collapse", "is_open")]
    )
    def toggle_video_about(n, is_open):
        if n:
            return not is_open, "Hide About Video Detection" if not is_open else "Show About Video Detection"
        return is_open, "Show About Video Detection"
    
    # Callback for toggling technical details
    @app.callback(
        [Output("timesformer-technical-collapse", "is_open"),
         Output("toggle-timesformer-details", "children")],
        [Input("toggle-timesformer-details", "n_clicks")],
        [State("timesformer-technical-collapse", "is_open")]
    )
    def toggle_timesformer_details(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"
    
    # Callback for video upload and preview
    @app.callback(
        [Output('output-video-upload', 'children'),
         Output('analyze-video-button', 'disabled')],
        [Input('upload-video', 'contents')],
        [State('upload-video', 'filename'),
         State('upload-video', 'last_modified')]
    )
    def update_video_preview(content, filename, date):
        if content is None:
            return None, True
        
        try:
            # Check file extension
            valid_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
            ext = os.path.splitext(filename)[1].lower() if filename else ''
            
            if ext not in valid_extensions:
                return html.Div([
                    html.P(f"Invalid file type. Please upload a video file ({', '.join(valid_extensions)})",
                         className="text-danger")
                ]), True
            
            # Display video player for preview
            video_div = html.Div([
                html.Video(src=content, controls=True, className="w-100", style={"max-height": "200px"}),
                html.P(filename, className="text-center mt-2")
            ])
            
            # Store the video content in the app object
            app.uploaded_video = {
                'content': content,
                'filename': filename,
                'date': date
            }
            
            return video_div, False
            
        except Exception as e:
            return html.Div([
                html.P(f"Error processing video: {str(e)}",
                     className="text-danger")
            ]), True
    
    # Callback for video analysis
    @app.callback(
        Output('video-results-container', 'children'),
        [Input('analyze-video-button', 'n_clicks')],
        [State('video-model-select', 'value'),
         State('video-confidence-threshold-slider', 'value')],
        prevent_initial_call=True
    )
    def analyze_video(n_clicks, model_name, confidence_threshold):
        if n_clicks is None or not hasattr(app, 'uploaded_video'):
            return html.P("Upload and analyze a video to see results.")
        
        try:
            # Show loading state first
            loading_div = html.Div([
                html.Div(className="d-flex justify-content-center", children=[
                    html.Div(className="spinner-border text-primary", role="status", children=[
                        html.Span("Loading...", className="visually-hidden")
                    ])
                ]),
                html.P("Analyzing video...", className="text-center mt-2")
            ])
            
            # Get the base64 data from the upload
            content = app.uploaded_video['content']
            filename = app.uploaded_video['filename']
            content_type, content_string = content.split(',', 1)
            decoded_content = base64.b64decode(content_string)
            
            # Save to temp file for processing
            import tempfile
            import os
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, filename)
            
            with open(temp_file_path, 'wb') as f:
                f.write(decoded_content)
            
            # Process the video using the MediaProcessor
            model_params = {
                'model_name': model_name,
                'confidence_threshold': confidence_threshold  # Already in decimal format (0.5-0.95)
            }
            
            # Call the backend processor to analyze the video
            result = app._processor.detect_media(temp_file_path, 'video', model_params)
            
            # Get results
            is_deepfake = result['is_deepfake']
            confidence = result['confidence'] * 100  # Convert to percentage
            details = result.get('details', {})
            
            # Create visualization from result if available
            visualization = None
            if 'visualization' in result:
                visualization = html.Img(src=f"data:image/png;base64,{result['visualization']}", 
                                      className="img-fluid mb-3")
            else:
                visualization = html.P("No visualization available")
            
            # Build result display
            result_div = html.Div([
                # Create analysis result display
                html.Div(className="text-center mb-4 mt-4", children=[
                    html.H4("Analysis Result", className="mb-4"),
                    html.Div(["Deepfake Detected" if is_deepfake else "Authentic Video"], 
                           className=f"h3 {'text-danger' if is_deepfake else 'text-success'} mb-3")
                ]),
                
                # Display visualization if available
                html.Div(className="text-center mb-3", children=[visualization]),
                
                # Display confidence visually
                html.Div(className="px-4 mb-4", children=[
                    html.Div(className="d-flex justify-content-between mb-1", children=[
                        html.Span("Real", className="text-success"),
                        html.Span("Fake", className="text-danger")
                    ]),
                    html.P(f"Confidence: {confidence:.1f}%", className="text-center mb-2"),
                    dbc.Progress(value=confidence, className="mb-3", 
                                color="danger" if is_deepfake else "success",
                                striped=True, animated=True)
                ]),
                
                # Details
                html.Div([
                    dbc.Button("View Full Report", 
                              id="video-view-report", 
                              color="primary", 
                              className="w-100 mt-3")
                ])
            ])
            
            # Save the result for the report
            if not hasattr(app, 'analysis_results'):
                app.analysis_results = {}
            app.analysis_results['video'] = result
            
            return result_div
            
        except Exception as e:
            import traceback
            return html.Div([
                html.H5("Error analyzing video"),
                html.P(str(e), className="text-danger"),
                html.Pre(traceback.format_exc(), className="text-danger small")
            ])

def register_reports_callbacks(app):
    """
    Register callbacks for the reports tab.
    
{{ ... }}
    Args:
        app: Dash application instance
    """
    # Callback to populate report list based on filters
    @app.callback(
        Output('report-list', 'children'),
        [Input('report-search', 'value'),
         Input('report-filter-media', 'value'),
         Input('report-filter-result', 'value'),
         Input('report-sort', 'value'),
         Input('report-date-range', 'value')]
    )
    def update_report_list(search_term, media_type, result_type, sort_by, date_range):
        # For demonstration, we'll generate mock reports
        # In a real implementation, this would fetch reports from a database
        
        # Create sample reports
        sample_reports = [
            {
                "id": "rep_001",
                "title": "test_image.jpg",
                "date": "2023-07-15T14:30:00",
                "media_type": "image",
                "result": "deepfake",
                "confidence": 0.87,
                "thumbnail": "/assets/images/sample_thumb1.png"
            },
            {
                "id": "rep_002",
                "title": "voice_sample.mp3",
                "date": "2023-07-14T10:15:00",
                "media_type": "audio",
                "result": "authentic",
                "confidence": 0.92,
                "thumbnail": "/assets/images/sample_thumb2.png"
            },
            {
                "id": "rep_003",
                "title": "interview_clip.mp4",
                "date": "2023-07-13T16:45:00",
                "media_type": "video",
                "result": "deepfake",
                "confidence": 0.78,
                "thumbnail": "/assets/images/sample_thumb3.png"
            }
        ]
        
        # Apply filters
        filtered_reports = sample_reports
        
        # Filter by media type
        if media_type != "all":
            filtered_reports = [r for r in filtered_reports if r["media_type"] == media_type]
            
        # Filter by result type
        if result_type != "all":
            filtered_reports = [r for r in filtered_reports if r["result"] == result_type]
            
        # Filter by search term
        if search_term:
            search_term = search_term.lower()
            filtered_reports = [r for r in filtered_reports if search_term in r["title"].lower()]
        
        # Sort reports
        if sort_by == "date_desc":
            filtered_reports.sort(key=lambda x: x["date"], reverse=True)
        elif sort_by == "date_asc":
            filtered_reports.sort(key=lambda x: x["date"])
        elif sort_by == "conf_desc":
            filtered_reports.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "conf_asc":
            filtered_reports.sort(key=lambda x: x["confidence"])
        
        # Create report list items
        if not filtered_reports:
            return html.Div("No reports match your filters.", className="text-center py-4")
        
        report_items = []
        for report in filtered_reports:
            # Format date for display
            date_obj = datetime.strptime(report["date"], "%Y-%m-%dT%H:%M:%S")
            formatted_date = date_obj.strftime("%b %d, %Y %H:%M")
            
            # Create report item
            item = dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        # Media type icon or thumbnail
                        dbc.Col([
                            html.Div(className=f"media-icon media-icon-{report['media_type']}")
                        ], width=2),
                        
                        # Report details
                        dbc.Col([
                            html.H6(report["title"], className="mb-1"),
                            html.P([
                                html.Small(f"{formatted_date} · {report['media_type'].capitalize()}"),
                            ], className="mb-1 text-muted"),
                            html.Div([
                                html.Span(
                                    "AUTHENTIC" if report["result"] == "authentic" else "DEEPFAKE", 
                                    className=f"badge {'bg-success' if report['result'] == 'authentic' else 'bg-danger'} me-2"
                                ),
                                html.Small(f"Confidence: {int(report['confidence'] * 100)}%")
                            ])
                        ], width=10)
                    ])
                ]),
                className="mb-2 report-list-item",
                id=f"report-item-{report['id']}",
                style={"cursor": "pointer"}
            )
            
            report_items.append(item)
        
        return html.Div(report_items)
    
    # Callback to display report details when a report is clicked
    @app.callback(
        [Output('report-details-view', 'style'),
         Output('no-report-selected', 'style'),
         Output('report-title', 'children'),
         Output('report-verdict', 'children'),
         Output('report-verdict', 'className'),
         Output('report-date', 'children'),
         Output('report-media-type', 'children'),
         Output('report-confidence', 'children'),
         Output('report-filename', 'children'),
         Output('report-visualization', 'children'),
         Output('report-details', 'children')],
        [Input('report-item-rep_001', 'n_clicks'),
         Input('report-item-rep_002', 'n_clicks'),
         Input('report-item-rep_003', 'n_clicks')],
        prevent_initial_call=True
    )
    def display_report_details(clicks1, clicks2, clicks3):
        ctx = dash.callback_context
        if not ctx.triggered:
            # No report selected yet
            return {'display': 'none'}, {'display': 'block'}, "", "", "", "", "", "", "", "", ""
        
        # Determine which report was clicked
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        report_id = trigger_id.replace('report-item-', '')
        
        # Mock data for report details
        if report_id == "rep_001":
            # Image report
            return (
                {'display': 'block'},  # report-details-view style
                {'display': 'none'},   # no-report-selected style
                "test_image.jpg",      # report title
                "DEEPFAKE DETECTED",   # verdict
                "results-verdict verdict-deepfake",  # verdict class
                "Jul 15, 2023 14:30",  # date
                "Image",               # media type
                "87%",                 # confidence
                "test_image.jpg",      # filename
                html.Img(src="/assets/images/sample_heatmap.png", className="img-fluid"),  # visualization
                # Details
                html.Div([
                    html.P("This image shows clear signs of manipulation in the facial region."),
                    html.H6("Detection Models", className="mt-3"),
                    html.Ul([
                        html.Li("ViT Detector: 91% confidence in manipulation"),
                        html.Li("Face Analysis: Inconsistent lighting detected"),
                        html.Li("Frequency Analysis: Abnormal patterns in DCT domain")
                    ]),
                    html.H6("Manipulated Regions", className="mt-3"),
                    dbc.Progress(value=87, className="mb-2", color="danger"),
                    html.P("Primary manipulation detected in facial features, particularly around the eyes and mouth.")
                ])
            )
        elif report_id == "rep_002":
            # Audio report
            return (
                {'display': 'block'},  # report-details-view style
                {'display': 'none'},   # no-report-selected style
                "voice_sample.mp3",    # report title
                "AUTHENTIC",           # verdict
                "results-verdict verdict-authentic",  # verdict class
                "Jul 14, 2023 10:15",  # date
                "Audio",               # media type
                "92%",                 # confidence
                "voice_sample.mp3",    # filename
                html.Img(src="/assets/images/sample_spectrogram.png", className="img-fluid"),  # visualization
                # Details
                html.Div([
                    html.P("Analysis indicates that this audio file contains natural human speech with no signs of synthetic generation."),
                    html.H6("Detection Models", className="mt-3"),
                    html.Ul([
                        html.Li("Wav2Vec2 Detector: 93% confidence in authenticity"),
                        html.Li("XLSR Analysis: Normal speech patterns detected"),
                        html.Li("Spectrogram Analysis: Natural formant transitions observed")
                    ]),
                    html.H6("Audio Characteristics", className="mt-3"),
                    dbc.Progress(value=92, className="mb-2", color="success"),
                    html.P("Natural variation in pitch, tempo, and vocal resonance consistent with human speech.")
                ])
            )
        else:  # rep_003
            # Video report
            return (
                {'display': 'block'},  # report-details-view style
                {'display': 'none'},   # no-report-selected style
                "interview_clip.mp4",  # report title
                "DEEPFAKE DETECTED",   # verdict
                "results-verdict verdict-deepfake",  # verdict class
                "Jul 13, 2023 16:45",  # date
                "Video",               # media type
                "78%",                 # confidence
                "interview_clip.mp4",  # filename
                html.Img(src="/assets/images/sample_frames.png", className="img-fluid"),  # visualization
                # Details
                html.Div([
                    html.P("This video contains inconsistencies across frames indicating facial manipulation."),
                    html.H6("Detection Models", className="mt-3"),
                    html.Ul([
                        html.Li("TimeSformer: 82% confidence in manipulation"),
                        html.Li("Frame Consistency: Temporal anomalies detected"),
                        html.Li("Facial Landmarks: Unnatural movement patterns observed")
                    ]),
                    html.H6("Temporal Analysis", className="mt-3"),
                    dbc.Progress(value=78, className="mb-2", color="danger"),
                    html.P("Manipulation detected primarily in lip synchronization and eye movement. Strongest signals in frames 45-87.")
                ])
            )
    
    # Callback for export buttons
    @app.callback(
        Output("report-export-status", "children"),
        [Input("export-report-pdf", "n_clicks"),
         Input("download-report-vis", "n_clicks"),
         Input("delete-report", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_report_actions(export_clicks, download_clicks, delete_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "export-report-pdf":
            # In a real implementation, this would generate and download a PDF
            return html.Div("PDF export functionality would be implemented here.", className="mt-2 text-info")
            
        elif button_id == "download-report-vis":
            # In a real implementation, this would download the visualization image
            return html.Div("Visualization download functionality would be implemented here.", className="mt-2 text-info")
            
        elif button_id == "delete-report":
            # In a real implementation, this would delete the report
            return html.Div("Report deletion functionality would be implemented here.", className="mt-2 text-info")
            
        return ""
