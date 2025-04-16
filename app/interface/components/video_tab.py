"""
Video tab component for the Deepfake Detection Platform UI.
"""
import os
import base64
import json
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List

from app.utils.visualization import VisualizationManager, fig_to_base64

def create_video_tab(app: dash.Dash, visualizer: VisualizationManager) -> html.Div:
    """
    Create the video analysis tab component.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
        
    Returns:
        Dash HTML Division containing the video tab layout
    """
    # Create video tab layout
    video_tab = html.Div([
        html.H3("Video Deepfake Detection", className="text-glow mb-4 text-center"),
        
        dbc.Row([
            # Upload and preview column
            dbc.Col([
                html.Div([
                    html.H5("Upload Video", className="mb-3"),
                    dcc.Upload(
                        id='upload-video',
                        children=html.Div([
                            html.Div(className="loading-spinner", style={'display': 'none'}, id='video-upload-spinner'),
                            html.Div([
                                html.I(className="fas fa-film fa-3x mb-3 text-glow"),
                                html.P("Drag and Drop or Click to Select Video")
                            ], id='video-upload-text')
                        ]),
                        className='upload-container',
                        multiple=False
                    ),
                    html.Div(id='upload-video-error', className="text-danger mt-2"),
                    
                    html.Div([
                        html.H5("Video Preview", className="mt-4 mb-3"),
                        html.Div(id='video-preview-container', className="text-center"),
                        
                        html.Div([
                            dbc.Button("Analyze", id="analyze-video-button", 
                                      color="primary", className="mt-3 animate-glow",
                                      disabled=True)
                        ], className="text-center")
                    ])
                ], className="card h-100")
            ], md=5),
            
            # Results column
            dbc.Col([
                html.Div([
                    html.H5("Detection Results", className="mb-3"),
                    
                    # Initial state - no results
                    html.Div([
                        html.P("Upload and analyze a video to see results.", className="text-center py-5"),
                    ], id='no-video-results', style={'display': 'block'}),
                    
                    # Loading state
                    html.Div([
                        html.Div(className="loading-container", children=[
                            html.Div(className="loading-spinner"),
                            html.P("Analyzing Video...", className="loading-text")
                        ])
                    ], id='video-analyzing', style={'display': 'none'}),
                    
                    # Results state
                    html.Div([
                        # Verdict header
                        html.Div([
                            html.H4("Verdict:", className="results-title"),
                            html.Div(id='video-verdict', className="results-verdict")
                        ], className="results-header"),
                        
                        # Confidence gauge
                        html.Div([
                            html.H5("Detection Confidence"),
                            html.Div([
                                html.Div(className="confidence-meter", children=[
                                    html.Div(id='video-confidence-fill', className="confidence-fill"),
                                    html.Div(className="confidence-threshold")
                                ]),
                                html.Div(id='video-confidence-text', className="text-center")
                            ])
                        ], className="mb-4"),
                        
                        # Visualization tabs
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id='video-frames-visualization', className="mt-3")
                            ], label="Key Frames", tab_id="frames-tab"),
                            
                            dbc.Tab([
                                html.Div(id='video-temporal-visualization', className="mt-3")
                            ], label="Temporal Analysis", tab_id="video-temporal-tab"),
                            
                            dbc.Tab([
                                html.Div(id='video-sync-visualization', className="mt-3")
                            ], label="A/V Sync Analysis", tab_id="sync-tab"),
                            
                            dbc.Tab([
                                html.Div(id='video-details-container', className="mt-3")
                            ], label="Detailed Analysis", tab_id="video-details-tab")
                        ], id="video-result-tabs", active_tab="frames-tab", className="mt-3"),
                        
                        # Export section
                        html.Div([
                            html.H5("Export Results", className="mt-4"),
                            dbc.Button("Save Report", id="save-video-report", 
                                      color="primary", className="me-2"),
                            dbc.Button("Download Visualization", id="download-video-vis", 
                                      color="secondary")
                        ], className="mt-4 pt-3 border-top")
                        
                    ], id='video-results', style={'display': 'none'})
                    
                ], className="card h-100")
            ], md=7)
        ])
    ], id="video-tab-content")
    
    # Register callbacks for the video tab
    register_video_callbacks(app, visualizer)
    
    return video_tab

def register_video_callbacks(app: dash.Dash, visualizer: VisualizationManager):
    """
    Register all callbacks for the video tab.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
    """
    # Callback for video upload
    @app.callback(
        [Output('video-preview-container', 'children'),
         Output('analyze-video-button', 'disabled'),
         Output('upload-video-error', 'children'),
         Output('video-upload-spinner', 'style'),
         Output('video-upload-text', 'style')],
        [Input('upload-video', 'contents')],
        [State('upload-video', 'filename')]
    )
    def update_video_preview(content, filename):
        if content is None:
            return None, True, "", {'display': 'none'}, {'display': 'block'}
        
        try:
            # Show loading state
            spinner_style = {'display': 'block'}
            text_style = {'display': 'none'}
            
            # Check file extension
            valid_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in valid_extensions:
                return None, True, f"Invalid file type. Please upload a video ({', '.join(valid_extensions)})", {'display': 'none'}, {'display': 'block'}
            
            # Create video player
            video_div = html.Div([
                html.Video(
                    src=content,
                    controls=True,
                    style={'max-width': '100%', 'max-height': '300px'}
                ),
                html.P(filename, className="mt-2 text-center")
            ])
            
            # Store the video content in a hidden div for later use
            app.uploaded_video_content = content
            app.uploaded_video_filename = filename
            
            return video_div, False, "", {'display': 'none'}, {'display': 'block'}
            
        except Exception as e:
            return None, True, f"Error: {str(e)}", {'display': 'none'}, {'display': 'block'}
    
    # Callback for analyze button
    @app.callback(
        [Output('no-video-results', 'style'),
         Output('video-analyzing', 'style'),
         Output('video-results', 'style'),
         Output('video-verdict', 'children'),
         Output('video-verdict', 'className'),
         Output('video-confidence-fill', 'style'),
         Output('video-confidence-text', 'children'),
         Output('video-frames-visualization', 'children'),
         Output('video-temporal-visualization', 'children'),
         Output('video-sync-visualization', 'children'),
         Output('video-details-container', 'children')],
        [Input('analyze-video-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def analyze_video(n_clicks):
        if n_clicks is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None, None
        
        # Get the processor from the app instance
        processor = app.processor
        
        if not hasattr(app, 'uploaded_video_content') or app.uploaded_video_content is None:
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, "", "", {}, "", None, None, None, None
        
        try:
            # Show analyzing state
            no_results_style = {'display': 'none'}
            analyzing_style = {'display': 'block'}
            results_style = {'display': 'none'}
            
            # Decode video
            content_type, content_string = app.uploaded_video_content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create a temporary file with the uploaded content
            temp_dir = app.config['general'].get('temp_dir', './temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, app.uploaded_video_filename)
            
            with open(temp_path, "wb") as f:
                f.write(decoded)
            
            # Process the video with the detector
            result = processor.detect_media(temp_path, 'video')
            
            # Determine verdict info
            is_deepfake = result['is_deepfake']
            confidence = result['confidence']
            
            # Configure verdict elements
            verdict_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC"
            verdict_class = "results-verdict verdict-deepfake" if is_deepfake else "results-verdict verdict-authentic"
            
            # Configure confidence bar
            confidence_bar_style = {'width': f"{confidence * 100}%"}
            confidence_text = f"Confidence: {confidence:.2f} ({confidence * 100:.1f}%)"
            
            # Create dashboard visualizations
            dashboard = visualizer.create_video_analysis_dashboard(result)
            
            # 1. Key Frames Visualization
            frames_div = html.Div([
                html.H5("Key Frames Analysis"),
                html.P("These frames show the most significant points of analysis in the video.")
            ])
            
            # Add frame overlays to visualization
            frame_cards = []
            for frame in dashboard.get("frame_overlays", []):
                frame_card = dbc.Card([
                    dbc.CardHeader([
                        f"Frame at {frame['time']:.2f}s",
                        html.Span(
                            f"Score: {frame['score']:.2f}",
                            style={
                                'float': 'right', 
                                'color': visualizer._get_confidence_color(frame['score'])
                            }
                        )
                    ]),
                    dbc.CardBody([
                        html.Img(
                            src=f"data:image/png;base64,{frame['overlay']}",
                            style={'max-width': '100%'}
                        )
                    ])
                ], className="mb-3")
                frame_cards.append(frame_card)
            
            # Add cards to frames visualization
            if frame_cards:
                frames_div.children.extend(frame_cards)
            else:
                frames_div.children.append(
                    html.P("No frame analysis available for this video.")
                )
            
            # 2. Temporal Visualization
            temporal_div = html.Div([
                html.H5("Temporal Analysis")
            ])
            
            if "temporal_plot" in dashboard:
                temporal_div.children.extend([
                    html.P("This plot shows how the deepfake confidence varies over time in the video."),
                    html.Img(
                        src=f"data:image/png;base64,{dashboard['temporal_plot']}",
                        style={'max-width': '100%'}
                    )
                ])
            else:
                temporal_div.children.append(
                    html.P("Temporal analysis not available for this video.")
                )
            
            # 3. A/V Sync Analysis
            sync_div = html.Div([
                html.H5("Audio-Video Synchronization")
            ])
            
            # Get A/V sync data
            av_sync_score = 0.0
            has_audio = False
            
            if "details" in result:
                av_sync_score = result["details"].get("av_sync_score", 0.0)
                if "metadata" in result and "has_audio" in result["metadata"]:
                    has_audio = result["metadata"]["has_audio"]
            
            if has_audio:
                # Create gauge chart for A/V sync
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=av_sync_score * 100,
                    title={'text': "A/V Sync Score", 'font': {'color': visualizer.colors["text"]}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': visualizer.colors["text"]},
                        'bar': {'color': visualizer._get_confidence_color(1 - av_sync_score)},
                        'bgcolor': visualizer.colors["grid"],
                        'borderwidth': 2,
                        'bordercolor': visualizer.colors["text"],
                        'steps': [
                            {'range': [0, 40], 'color': visualizer.colors["deepfake"]},
                            {'range': [40, 70], 'color': visualizer.colors["uncertain"]},
                            {'range': [70, 100], 'color': visualizer.colors["authentic"]}
                        ],
                    },
                    number={'suffix': '%', 'font': {'color': visualizer.colors["text"]}}
                ))
                
                fig.update_layout(
                    paper_bgcolor=visualizer.colors["background"],
                    height=300,
                    margin=dict(t=50, b=0, l=25, r=25)
                )
                
                sync_div.children.extend([
                    html.P("This analysis measures the synchronization between audio and visual elements, which can identify deepfakes where lip movements don't match the audio."),
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ])
            else:
                sync_div.children.append(
                    html.P("No audio track detected in this video. Audio-video sync analysis not available.")
                )
            
            # 4. Detailed Analysis
            stats = dashboard.get("summary_stats", {})
            
            details_div = html.Div([
                html.H5("Analysis Details"),
                
                html.Table([
                    html.Tr([
                        html.Td("Video Duration:"),
                        html.Td(f"{result['details']['video_info'].get('duration', 0):.2f} seconds")
                    ]),
                    html.Tr([
                        html.Td("Original Resolution:"),
                        html.Td(f"{result['details']['video_info'].get('width', 0)} x {result['details']['video_info'].get('height', 0)}")
                    ]),
                    html.Tr([
                        html.Td("Frames Analyzed:"),
                        html.Td(f"{stats.get('total_frames_analyzed', 0)}")
                    ]),
                    html.Tr([
                        html.Td("Model Used:"),
                        html.Td(result.get("model", "GenConViT"))
                    ]),
                    html.Tr([
                        html.Td("Confidence Threshold:"),
                        html.Td(f"{result.get('threshold', 0.5):.2f}")
                    ]),
                    html.Tr([
                        html.Td("Analysis Time:"),
                        html.Td(f"{result.get('analysis_time', 0):.3f} seconds")
                    ])
                ], className="table table-sm"),
                
                html.Div([
                    html.H5("Summary Statistics", className="mt-4"),
                    html.Table([
                        html.Tr([
                            html.Td("Deepfake Frame Percentage:"),
                            html.Td(f"{stats.get('deepfake_frame_percentage', 0):.1f}%")
                        ]),
                        html.Tr([
                            html.Td("Temporal Consistency:"),
                            html.Td(f"{stats.get('temporal_consistency', 0):.3f}")
                        ]),
                        html.Tr([
                            html.Td("A/V Sync Score:"),
                            html.Td(f"{stats.get('av_sync_score', 0):.3f}")
                        ])
                    ], className="table table-sm"),
                    
                    # Donut chart showing frame classification
                    html.Div([
                        dcc.Graph(
                            figure={
                                'data': [
                                    go.Pie(
                                        labels=['Authentic Frames', 'Deepfake Frames'],
                                        values=[
                                            100 - stats.get('deepfake_frame_percentage', 0),
                                            stats.get('deepfake_frame_percentage', 0)
                                        ],
                                        hole=.4,
                                        marker=dict(
                                            colors=[
                                                visualizer.colors["authentic"], 
                                                visualizer.colors["deepfake"]
                                            ]
                                        )
                                    )
                                ],
                                'layout': {
                                    'title': 'Frame Classification',
                                    'paper_bgcolor': visualizer.colors["background"],
                                    'font': {'color': visualizer.colors["text"]},
                                    'height': 300,
                                    'margin': {'t': 30, 'b': 0, 'l': 0, 'r': 0}
                                }
                            },
                            config={'displayModeBar': False}
                        )
                    ], className="mt-3")
                ])
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
                   frames_div, 
                   temporal_div, 
                   sync_div,
                   details_div)
            
        except Exception as e:
            # Show error
            error_div = html.Div([
                html.H5("Error analyzing video:"),
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
                   None,
                   None)
    
    # Callback for download visualization button
    @app.callback(
        Output('download-video-vis', 'href'),
        [Input('video-result-tabs', 'active_tab')],
        [State('video-frames-visualization', 'children'),
         State('video-temporal-visualization', 'children')],
        prevent_initial_call=True
    )
    def prepare_video_download(active_tab, frames_vis, temporal_vis):
        if active_tab == "frames-tab" and frames_vis is not None:
            # Extract the base64 string from the first frame if available
            try:
                for child in frames_vis['props']['children']:
                    if isinstance(child, dict) and child.get('type') == 'Card':
                        img = child['props']['children'][1]['props']['children'][0]
                        src = img['props']['src']
                        return src
                return ""
            except:
                return ""
                
        elif active_tab == "video-temporal-tab" and temporal_vis is not None:
            try:
                img = temporal_vis['props']['children'][1]
                src = img['props']['src']
                return src
            except:
                return ""
                
        return ""
