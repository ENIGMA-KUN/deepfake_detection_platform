"""
Audio tab component for the Deepfake Detection Platform UI.
"""
import os
import base64
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go

from app.utils.visualization import VisualizationManager, fig_to_base64

def create_audio_tab(app: dash.Dash, visualizer: VisualizationManager) -> html.Div:
    """
    Create the audio analysis tab component.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
        
    Returns:
        Dash HTML Division containing the audio tab layout
    """
    # Create audio tab layout
    audio_tab = html.Div([
        html.H3("Audio Deepfake Detection", className="text-glow mb-4 text-center"),
        
        dbc.Row([
            # Upload and preview column
            dbc.Col([
                html.Div([
                    html.H5("Upload Audio", className="mb-3"),
                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            html.Div(className="loading-spinner", style={'display': 'none'}, id='audio-upload-spinner'),
                            html.Div([
                                html.I(className="fas fa-music fa-3x mb-3 text-glow"),
                                html.P("Drag and Drop or Click to Select Audio File")
                            ], id='audio-upload-text')
                        ]),
                        className='upload-container',
                        multiple=False
                    ),
                    html.Div(id='upload-audio-error', className="text-danger mt-2"),
                    
                    html.Div([
                        html.H5("Audio Preview", className="mt-4 mb-3"),
                        html.Div(id='audio-preview-container', className="text-center"),
                        
                        html.Div([
                            dbc.Button("Analyze", id="analyze-audio-button", 
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
                        html.P("Upload and analyze an audio file to see results.", className="text-center py-5"),
                    ], id='no-audio-results', style={'display': 'block'}),
                    
                    # Loading state
                    html.Div([
                        html.Div(className="loading-container", children=[
                            html.Div(className="loading-spinner"),
                            html.P("Analyzing Audio...", className="loading-text")
                        ])
                    ], id='audio-analyzing', style={'display': 'none'}),
                    
                    # Results state
                    html.Div([
                        # Verdict header
                        html.Div([
                            html.H4("Verdict:", className="results-title"),
                            html.Div(id='audio-verdict', className="results-verdict")
                        ], className="results-header"),
                        
                        # Confidence gauge
                        html.Div([
                            html.H5("Detection Confidence"),
                            html.Div([
                                html.Div(className="confidence-meter", children=[
                                    html.Div(id='audio-confidence-fill', className="confidence-fill"),
                                    html.Div(className="confidence-threshold")
                                ]),
                                html.Div(id='audio-confidence-text', className="text-center")
                            ])
                        ], className="mb-4"),
                        
                        # Visualization tabs
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id='audio-spectrogram-visualization', className="mt-3 text-center")
                            ], label="Spectrogram", tab_id="spectrogram-tab"),
                            
                            dbc.Tab([
                                html.Div(id='audio-temporal-visualization', className="mt-3 text-center")
                            ], label="Temporal Analysis", tab_id="temporal-tab"),
                            
                            dbc.Tab([
                                html.Div(id='audio-details-container', className="mt-3")
                            ], label="Detailed Analysis", tab_id="audio-details-tab")
                        ], id="audio-result-tabs", active_tab="spectrogram-tab", className="mt-3"),
                        
                        # Export section
                        html.Div([
                            html.H5("Export Results", className="mt-4"),
                            dbc.Button("Save Report", id="save-audio-report", 
                                      color="primary", className="me-2"),
                            dbc.Button("Download Visualization", id="download-audio-vis", 
                                      color="secondary")
                        ], className="mt-4 pt-3 border-top")
                        
                    ], id='audio-results', style={'display': 'none'})
                    
                ], className="card h-100")
            ], md=6)
        ])
    ], id="audio-tab-content")
    
    # Register callbacks for the audio tab
    register_audio_callbacks(app, visualizer)
    
    return audio_tab

def register_audio_callbacks(app: dash.Dash, visualizer: VisualizationManager):
    """
    Register all callbacks for the audio tab.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
    """
    # Callback for audio upload
    @app.callback(
        [Output('audio-preview-container', 'children'),
         Output('analyze-audio-button', 'disabled'),
         Output('upload-audio-error', 'children'),
         Output('audio-upload-spinner', 'style'),
         Output('audio-upload-text', 'style')],
        [Input('upload-audio', 'contents')],
        [State('upload-audio', 'filename')]
    )
    def update_audio_preview(content, filename):
        if content is None:
            return None, True, "", {'display': 'none'}, {'display': 'block'}
        
        try:
            # Show loading state
            spinner_style = {'display': 'block'}
            text_style = {'display': 'none'}
            
            # Check file extension
            valid_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in valid_extensions:
                return None, True, f"Invalid file type. Please upload an audio file ({', '.join(valid_extensions)})", {'display': 'none'}, {'display': 'block'}
            
            # Create audio player
            audio_div = html.Div([
                html.Audio(
                    src=content,
                    controls=True,
                    style={'width': '100%'}
                ),
                html.P(filename, className="mt-2 text-center")
            ])
            
            # Store the audio content in a hidden div for later use
            app.uploaded_audio_content = content
            app.uploaded_audio_filename = filename
            
            return audio_div, False, "", {'display': 'none'}, {'display': 'block'}
            
        except Exception as e:
            return None, True, f"Error: {str(e)}", {'display': 'none'}, {'display': 'block'}
    
    # Callback for analyze button
    @app.callback(
        [Output('no-audio-results', 'style'),
         Output('audio-analyzing', 'style'),
         Output('audio-results', 'style'),
         Output('audio-verdict', 'children'),
         Output('audio-verdict', 'className'),
         Output('audio-confidence-fill', 'style'),
         Output('audio-confidence-text', 'children'),
         Output('audio-spectrogram-visualization', 'children'),
         Output('audio-temporal-visualization', 'children'),
         Output('audio-details-container', 'children')],
        [Input('analyze-audio-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def analyze_audio(n_clicks):
        if n_clicks is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None
        
        # Get the processor from the app instance
        processor = app.processor
        
        if not hasattr(app, 'uploaded_audio_content') or app.uploaded_audio_content is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None
        
        try:
            # Show analyzing state
            no_results_style = {'display': 'none'}
            analyzing_style = {'display': 'block'}
            results_style = {'display': 'none'}
            
            # Decode audio
            content_type, content_string = app.uploaded_audio_content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create a temporary file with the uploaded content
            temp_dir = app.config['general'].get('temp_dir', './temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, app.uploaded_audio_filename)
            
            with open(temp_path, "wb") as f:
                f.write(decoded)
            
            # Process the audio with the detector
            result = processor.detect_media(temp_path, 'audio')
            
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
            # 1. Spectrogram
            spectrogram_div = None
            if "details" in result and "spectrogram_features" in result["details"]:
                spectrogram = np.array(result["details"]["spectrogram_features"])
                
                # Create spectrogram visualization
                spec_fig = visualizer.create_audio_spectrogram(
                    spectrogram, 
                    is_deepfake=is_deepfake,
                    confidence=confidence
                )
                
                # Convert to base64 for display
                spec_b64 = fig_to_base64(spec_fig)
                
                spectrogram_div = html.Div([
                    html.Img(
                        src=f"data:image/png;base64,{spec_b64}",
                        style={'max-width': '100%', 'max-height': '400px'}
                    )
                ])
            else:
                spectrogram_div = html.Div([
                    html.P("Spectrogram visualization not available for this audio.")
                ])
            
            # 2. Temporal analysis
            temporal_div = None
            if "details" in result and "temporal_analysis" in result["details"]:
                temporal = result["details"]["temporal_analysis"]
                
                if "segment_scores" in temporal and "segment_times" in temporal:
                    scores = temporal["segment_scores"]
                    times = temporal["segment_times"]
                    
                    # Create temporal visualization
                    temp_fig = visualizer.create_temporal_analysis(
                        scores,
                        times,
                        threshold=result.get('threshold', 0.5)
                    )
                    
                    # Convert to base64 for display
                    temp_b64 = fig_to_base64(temp_fig)
                    
                    temporal_div = html.Div([
                        html.Img(
                            src=f"data:image/png;base64,{temp_b64}",
                            style={'max-width': '100%', 'max-height': '400px'}
                        ),
                        html.P(f"Temporal Inconsistency Index: {temporal.get('inconsistency_index', 0):.3f}", 
                             className="mt-2 text-center",
                             style={'color': visualizer._get_confidence_color(temporal.get('inconsistency_index', 0))})
                    ])
                else:
                    temporal_div = html.Div([
                        html.P("Temporal analysis data not available for this audio.")
                    ])
            else:
                temporal_div = html.Div([
                    html.P("Temporal analysis not available for this audio.")
                ])
            
            # 3. Detailed analysis
            details_div = html.Div([
                html.H5("Analysis Details"),
                
                html.Table([
                    html.Tr([
                        html.Td("Original Sample Rate:"),
                        html.Td(f"{result['details'].get('original_sample_rate', 'Unknown')} Hz")
                    ]),
                    html.Tr([
                        html.Td("Duration:"),
                        html.Td(f"{result['details'].get('duration', 0):.2f} seconds")
                    ]),
                    html.Tr([
                        html.Td("Model Used:"),
                        html.Td(result.get("model", "Wav2Vec2"))
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
            
            # If temporal analysis is available, add more details
            if "details" in result and "temporal_analysis" in result["details"]:
                temporal = result["details"]["temporal_analysis"]
                
                temporal_details = html.Div([
                    html.H5("Temporal Analysis", className="mt-4"),
                    html.P([
                        "Inconsistency Index: ",
                        html.Span(
                            f"{temporal.get('inconsistency_index', 0):.3f}",
                            style={'color': visualizer._get_confidence_color(temporal.get('inconsistency_index', 0))}
                        ),
                        html.Br(),
                        html.Small("Higher values indicate potential manipulation across different segments of audio.")
                    ]),
                    
                    # Add a gauge chart for inconsistency
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Indicator(
                                    mode="gauge+number",
                                    value=temporal.get('inconsistency_index', 0) * 100,
                                    title={'text': "Temporal Inconsistency"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': visualizer._get_confidence_color(temporal.get('inconsistency_index', 0))},
                                        'steps': [
                                            {'range': [0, 30], 'color': visualizer.colors["authentic"]},
                                            {'range': [30, 70], 'color': visualizer.colors["uncertain"]},
                                            {'range': [70, 100], 'color': visualizer.colors["deepfake"]}
                                        ],
                                    },
                                    number={'suffix': '%'}
                                )
                            ],
                            'layout': {
                                'height': 250,
                                'margin': {'t': 25, 'b': 0, 'l': 25, 'r': 25},
                                'paper_bgcolor': visualizer.colors["background"],
                                'font': {'color': visualizer.colors["text"]}
                            }
                        },
                        config={'displayModeBar': False}
                    )
                ])
                
                details_div.children.append(temporal_details)
            
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
                   spectrogram_div, 
                   temporal_div, 
                   details_div)
            
        except Exception as e:
            # Show error
            error_div = html.Div([
                html.H5("Error analyzing audio:"),
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
        Output('download-audio-vis', 'href'),
        [Input('audio-result-tabs', 'active_tab')],
        [State('audio-spectrogram-visualization', 'children'),
         State('audio-temporal-visualization', 'children')],
        prevent_initial_call=True
    )
    def prepare_audio_download(active_tab, spectrogram_vis, temporal_vis):
        if active_tab == "spectrogram-tab" and spectrogram_vis is not None:
            # Extract the base64 string from the img src
            try:
                img_div = spectrogram_vis['props']['children'][0]
                src = img_div['props']['src']
                return src
            except:
                return ""
                
        elif active_tab == "temporal-tab" and temporal_vis is not None:
            try:
                img_div = temporal_vis['props']['children'][0]
                src = img_div['props']['src']
                return src
            except:
                return ""
                
        return ""
