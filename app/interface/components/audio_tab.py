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
import plotly.graph_objects as go
from typing import Dict, Any, List
import json

from app.utils.visualization import VisualizationManager
from app.utils.premium_utils import get_premium_badge_html
from app.interface.components.detailed_analysis_panel import create_detailed_analysis_panel

def create_audio_tab(app: dash.Dash) -> html.Div:
    """
    Create the audio tab component.
    
    Args:
        app: Dash application instance
        
    Returns:
        Dash HTML Division containing the audio tab layout
    """
    # Create audio tab layout
    audio_tab = html.Div([
        html.H3("Audio Deepfake Detection", className="text-glow mb-4 text-center"),
        
        # About section - collapsible information panel
        dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    "About Audio Detection",
                    id="toggle-audio-about",
                    color="info",
                    outline=True,
                )
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.H5("Audio Deepfake Detection", className="card-title"),
                    html.P([
                        "Our audio detection module uses advanced Wav2Vec and XLSR models to identify AI-generated or manipulated audio. ",
                        "The system analyzes spectral patterns, linguistic inconsistencies, and audio artifacts that may not be audible to human ears."
                    ]),
                    html.H6("Key Features:"),
                    html.Ul([
                        html.Li("Voice cloning detection with high accuracy"),
                        html.Li("Text-to-speech synthesis identification"),
                        html.Li("Audio splicing and manipulation detection"),
                        html.Li("Multi-language support for global content analysis")
                    ]),
                    html.H6("Performance:"),
                    html.P([
                        "Our audio detection achieves over 91% accuracy on benchmark datasets including ",
                        html.Span("ASVspoof", style={"font-style": "italic"}),
                        " and ",
                        html.Span("FakeAudio", style={"font-style": "italic"}),
                        "."
                    ])
                ]),
                id="audio-about-collapse",
                is_open=False,
            ),
        ], className="mb-4 shadow"),
        
        dbc.Row([
            # Upload and preview column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Upload Audio", className="text-primary mb-0")),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-audio',
                            children=html.Div([
                                html.I(className="fas fa-microphone me-2", style={"fontSize": "24px"}),
                                html.Span('Drag and Drop or '),
                                html.A('Select Audio File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '120px',
                                'lineHeight': '120px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'backgroundColor': '#0d1117',
                                'borderColor': '#4CC9F0'
                            },
                            multiple=False
                        ),
                        html.Div(id='output-audio-upload', className="mt-3"),
                        html.Div([
                            dbc.Button(
                                "Analyze Audio", 
                                id="analyze-audio-button", 
                                color="primary", 
                                className="w-100 mt-3",
                                disabled=True
                            )
                        ], className="text-center")
                    ])
                ], className="h-100 shadow")
            ], md=6),
            
            # Settings column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Detection Settings", className="text-primary mb-0")),
                    dbc.CardBody([
                        html.Div([
                            html.Label("Detection Model:", className="fw-bold"),
                            dbc.Select(
                                id='audio-model-select',
                                options=[
                                    {"label": "Wav2Vec2 (Self-supervised)", "value": "wav2vec"},
                                    {"label": html.Span([
                                        "XLSR (Cross-lingual)",
                                        get_premium_badge_html("xlsr", "audio")
                                    ]), "value": "xlsr"},
                                    {"label": html.Span([
                                        "Mamba-Audio (State Space Model)",
                                        get_premium_badge_html("mamba", "audio")
                                    ]), "value": "mamba"},
                                    {"label": "TCN (Temporal CNN)", "value": "tcn"},
                                    {"label": "Ensemble (All Models)", "value": "ensemble"}
                                ],
                                value='ensemble',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Analysis Mode:", className="fw-bold"),
                            dbc.RadioItems(
                                id='audio-analysis-mode',
                                options=[
                                    {"label": "Standard Analysis", "value": "standard"},
                                    {"label": "Deep Analysis (slower)", "value": "deep"},
                                    {"label": "Frequency Sentinel™ Mode", "value": "singularity"}
                                ],
                                value='standard',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Confidence Threshold:", className="fw-bold"),
                            html.Div([
                                dcc.Slider(
                                    id='audio-confidence-threshold-slider',
                                    min=0.5,
                                    max=0.95,
                                    step=0.05,
                                    value=0.7,
                                    marks={
                                        0.5: {'label': '0.5', 'style': {'color': '#77B5FE'}},
                                        0.7: {'label': '0.7', 'style': {'color': '#77B5FE'}},
                                        0.9: {'label': '0.9', 'style': {'color': '#77B5FE'}}
                                    },
                                ),
                                html.Div([
                                    html.Span("Lower: More sensitive, may have false positives", className="small text-muted")
                                ], style={"textAlign": "left", "marginTop": "8px"}),
                                html.Div([
                                    html.Span("Higher: Less sensitive, may miss subtle manipulation", className="small text-muted")
                                ], style={"textAlign": "right", "marginTop": "8px"})
                            ])
                        ])
                    ])
                ], className="h-100 shadow")
            ], md=6)
        ], className="mb-4"),
        
        # Results section (initially hidden, shown after analysis)
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                dbc.CardBody([
                    # Main results container
                    html.Div(id="audio-results-container", className="mb-4"),
                    
                    # Spectrogram visualization and controls
                    html.Div([
                        html.H5("Spectrogram Analysis", className="mb-3"),
                        
                        # Audio player and basic results
                        dbc.Row([
                            # Original audio player column
                            dbc.Col([
                                html.H6("Original Audio", className="text-center mb-2"),
                                html.Div(id="original-audio-player", className="d-flex justify-content-center")
                            ], md=6, className="mb-3"),
                            
                            # Basic detection results column
                            dbc.Col([
                                html.H6("Detection Summary", className="text-center mb-2"),
                                html.Div(id="audio-detection-summary", className="d-flex justify-content-center")
                            ], md=6, className="mb-3")
                        ], className="mb-4"),
                        
                        # Spectrogram visualization
                        dbc.Row([
                            dbc.Col([
                                html.H6("Spectrogram with Anomaly Highlighting", className="text-center mb-2"),
                                dcc.Graph(
                                    id="audio-spectrogram-graph",
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['toImage', 'select2d', 'lasso2d'],
                                        'displaylogo': False
                                    },
                                    style={"height": "350px", "backgroundColor": "#0d1117", "border": "1px solid #1E5F75"}
                                )
                            ], className="mb-3")
                        ]),
                        
                        # Spectrogram controls
                        dbc.Row([
                            # Colorscale selection
                            dbc.Col([
                                html.Label("Spectrogram Colorscale:", className="fw-bold"),
                                dbc.Select(
                                    id='spectrogram-colorscale-select',
                                    options=[
                                        {"label": "Tron (Blue-Cyan)", "value": "tron"},
                                        {"label": "Plasma (Purple-Orange)", "value": "plasma"},
                                        {"label": "Viridis (Blue-Green-Yellow)", "value": "viridis"},
                                        {"label": "Inferno (Red-Yellow)", "value": "inferno"},
                                        {"label": "Jet (Blue-Green-Red-Yellow)", "value": "jet"}
                                    ],
                                    value='tron',
                                    className="mb-3"
                                )
                            ], md=4),
                            
                            # Highlight intensity
                            dbc.Col([
                                html.Label("Anomaly Highlight Intensity:", className="fw-bold"),
                                dcc.Slider(
                                    id='highlight-intensity-slider',
                                    min=0.3,
                                    max=1.0,
                                    step=0.05,
                                    value=0.7,
                                    marks={
                                        0.3: {'label': '30%', 'style': {'color': '#77B5FE'}},
                                        0.7: {'label': '70%', 'style': {'color': '#77B5FE'}},
                                        1.0: {'label': '100%', 'style': {'color': '#77B5FE'}}
                                    },
                                )
                            ], md=4),
                            
                            # Window size slider
                            dbc.Col([
                                html.Label("Time Window Size:", className="fw-bold"),
                                dcc.Slider(
                                    id='time-window-slider',
                                    min=1,
                                    max=5,
                                    step=0.5,
                                    value=2.5,
                                    marks={
                                        1: {'label': '1s', 'style': {'color': '#77B5FE'}},
                                        2.5: {'label': '2.5s', 'style': {'color': '#77B5FE'}},
                                        5: {'label': '5s', 'style': {'color': '#77B5FE'}}
                                    },
                                )
                            ], md=4)
                        ], className="mb-3"),
                        
                        # Explanation text
                        html.Div([
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                "The spectrogram shows time on the X-axis, frequency on the Y-axis, and intensity as color. ",
                                "Highlighted pink/magenta regions indicate potential anomalies or manipulation. ",
                                "These areas have frequency patterns that deviate from natural human speech."
                            ], className="small text-muted")
                        ])
                    ], id="spectrogram-visualization-container", style={"display": "none"}),
                    
                    # Detailed Analysis Report Panel
                    html.Div(
                        id="audio-detailed-analysis-container",
                        style={"display": "none"}
                    ),
                    
                    # Divider before detailed analysis
                    html.Hr(className="my-4")
                ])
            ], className="shadow")
        ], id="audio-results-section")
    ])
    
    # Register callbacks for audio analysis
    _register_audio_callbacks(app)
    
    return audio_tab

def _register_audio_callbacks(app: dash.Dash):
    """
    Register callbacks for the audio tab.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("spectrogram-visualization-container", "style"),
        Input("audio-results-container", "children"),
        prevent_initial_call=True
    )
    def show_spectrogram_container(results_content):
        """Show spectrogram container when results are available"""
        if results_content:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("original-audio-player", "children"),
        Input("audio-results-container", "children"),
        prevent_initial_call=True
    )
    def update_audio_player(results_content):
        """Update the audio player based on results"""
        if not results_content:
            return html.Div()
        
        try:
            # In a production implementation, we would extract the audio file path
            # from the detection results
            audio_player = html.Div([
                html.Audio(
                    id="audio-player",
                    src="",  # This would be populated with actual audio source
                    controls=True,
                    style={"width": "100%"}
                ),
                html.Div([
                    html.Button(
                        html.I(className="fas fa-play"),
                        id="play-button",
                        className="btn btn-sm btn-outline-info me-2"
                    ),
                    html.Button(
                        html.I(className="fas fa-pause"),
                        id="pause-button",
                        className="btn btn-sm btn-outline-secondary"
                    )
                ], className="mt-2 text-center")
            ])
            
            return audio_player
            
        except Exception as e:
            print(f"Error creating audio player: {str(e)}")
            return html.Div([
                html.P("Error loading audio player", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ])
    
    @app.callback(
        Output("audio-detection-summary", "children"),
        Input("audio-results-container", "children"),
        prevent_initial_call=True
    )
    def update_detection_summary(results_content):
        """Update the detection summary based on results"""
        if not results_content:
            return html.Div()
        
        try:
            # In a production implementation, we would extract the actual detection results
            # Here we're creating a placeholder summary card
            summary_card = dbc.Card([
                dbc.CardBody([
                    html.H5("97.8% Manipulated", className="text-danger text-center"),
                    html.Hr(className="my-2"),
                    html.P([
                        html.Span("Manipulation Type: ", className="fw-bold"),
                        "Voice Cloning"
                    ], className="mb-2"),
                    html.P([
                        html.Span("Affected Segments: ", className="fw-bold"),
                        "3 segments detected"
                    ], className="mb-2"),
                    html.P([
                        html.Span("Model Consensus: ", className="fw-bold"),
                        "4/5 models agree"
                    ], className="mb-0")
                ])
            ], style={"width": "100%"})
            
            return summary_card
            
        except Exception as e:
            print(f"Error creating detection summary: {str(e)}")
            return html.Div([
                html.P("Error loading summary", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ])
    
    @app.callback(
        Output("audio-spectrogram-graph", "figure"),
        [Input("audio-results-container", "children"),
         Input("spectrogram-colorscale-select", "value"),
         Input("highlight-intensity-slider", "value"),
         Input("time-window-slider", "value")],
        prevent_initial_call=True
    )
    def update_spectrogram_visualization(results_content, colorscale, highlight_intensity, time_window):
        """Update the spectrogram visualization based on results and user settings"""
        if not results_content:
            return {}
        
        try:
            # Create a sample spectrogram visualization with anomaly highlighting
            # In production, this would use real spectrogram data from the analysis
            
            # Create time and frequency data for the spectrogram
            time = np.linspace(0, 10, 100)  # 10 seconds of audio
            freq = np.linspace(0, 8000, 80)  # Frequencies up to 8kHz
            
            # Create spectrogram data (random for demonstration)
            spectrogram = np.random.rand(len(freq), len(time)) * 0.5
            
            # Add some structure to make it look more like speech
            # Formant frequencies (typical for speech)
            for formant in [500, 1500, 2500]:
                formant_idx = np.argmin(np.abs(freq - formant))
                spectrogram[formant_idx-2:formant_idx+2, :] += 0.3
            
            # Add some temporal structure
            for t in range(0, len(time), 10):
                spectrogram[:, t:t+5] *= 1.5
            
            # Create "anomaly" regions for highlighting
            anomaly_regions = [
                (20, 30),  # Time indices for first anomaly
                (55, 65),  # Time indices for second anomaly
                (80, 90)   # Time indices for third anomaly
            ]
            
            # Map colorscale to the Plotly values
            if colorscale == "tron":
                colorscale_value = [
                    [0, "#041E42"],
                    [0.33, "#0047AB"],
                    [0.66, "#00BFFF"],
                    [1, "#7DF9FF"]
                ]
            else:
                colorscale_value = colorscale
            
            # Create the spectrogram figure
            fig = go.Figure()
            
            # Add the base spectrogram
            fig.add_trace(go.Heatmap(
                z=spectrogram,
                x=time,
                y=freq,
                colorscale=colorscale_value,
                showscale=False
            ))
            
            # Add highlighted anomaly regions
            for start, end in anomaly_regions:
                # Create a separate heatmap for each anomaly region with a different color
                anomaly_data = np.zeros_like(spectrogram)
                anomaly_data[:, start:end] = spectrogram[:, start:end] * highlight_intensity
                
                fig.add_trace(go.Heatmap(
                    z=anomaly_data,
                    x=time,
                    y=freq,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,0,255,0.7)"]],
                    showscale=False,
                    hoverinfo="skip"
                ))
            
            # Add annotations for the anomalies
            for i, (start, end) in enumerate(anomaly_regions):
                midpoint = (time[start] + time[end]) / 2
                fig.add_annotation(
                    x=midpoint,
                    y=7500,  # Near the top of the frequency range
                    text=f"Anomaly {i+1}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#FF1493",
                    arrowwidth=2,
                    font=dict(color="#FFFFFF")
                )
            
            # Update layout
            fig.update_layout(
                title="",
                xaxis=dict(
                    title="Time (seconds)",
                    showgrid=True,
                    gridcolor="rgba(82, 113, 255, 0.2)",
                    range=[0, time_window]  # Show just the first X seconds (adjustable)
                ),
                yaxis=dict(
                    title="Frequency (Hz)",
                    showgrid=True,
                    gridcolor="rgba(82, 113, 255, 0.2)"
                ),
                plot_bgcolor="#0a0f18",
                paper_bgcolor="#0d1117",
                font=dict(color="#FFFFFF"),
                margin=dict(l=50, r=20, t=20, b=50),
                dragmode="pan",
                hovermode="closest"
            )
            
            # Add custom buttons for navigation
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="<<",
                                method="relayout",
                                args=[{"xaxis.range": [max(0, time_window - 5), max(time_window, 5)]}]
                            ),
                            dict(
                                label=">>",
                                method="relayout",
                                args=[{"xaxis.range": [min(max(time) - time_window, 5), min(max(time), 10)]}]
                            ),
                        ],
                        direction="right",
                        pad={"r": 10, "t": 10},
                        x=0.1,
                        y=1.1,
                        xanchor="right",
                        yanchor="top"
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating spectrogram: {str(e)}")
            # Return an empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title="Error loading spectrogram",
                annotations=[
                    dict(
                        text=str(e),
                        showarrow=False,
                        font=dict(color="red")
                    )
                ],
                plot_bgcolor="#0a0f18",
                paper_bgcolor="#0d1117",
                font=dict(color="#FFFFFF")
            )
            return fig
            
    # Add callback for detailed analysis panel
    @app.callback(
        [Output("audio-detailed-analysis-container", "children"),
         Output("audio-detailed-analysis-container", "style")],
        Input("audio-results-container", "children"),
        prevent_initial_call=True
    )
    def update_detailed_analysis_panel(results_content):
        """Show detailed analysis panel when results are available"""
        if not results_content:
            return None, {"display": "none"}
        
        try:
            # In a production environment, we would retrieve the actual detection results
            # For this demo, we'll use placeholder data
            mock_detection_result = {
                "is_deepfake": True,
                "confidence": 0.95,
                "detection_time_ms": 278,
                "manipulation_type": "Voice Cloning",
                "analysis_mode": "Frequency Sentinel™",
                "model_contributions": [
                    {"model": "Wav2Vec2", "confidence": 0.92, "weight": 0.30},
                    {"model": "XLSR", "confidence": 0.98, "weight": 0.25},
                    {"model": "Mamba-Audio", "confidence": 0.95, "weight": 0.25},
                    {"model": "TCN", "confidence": 0.88, "weight": 0.20}
                ],
                "model_results": [
                    {"model": "Wav2Vec2", "is_deepfake": True, "confidence": 0.92},
                    {"model": "XLSR", "is_deepfake": True, "confidence": 0.98},
                    {"model": "Mamba-Audio", "is_deepfake": True, "confidence": 0.95},
                    {"model": "TCN", "is_deepfake": True, "confidence": 0.88}
                ]
            }
            
            # Create the detailed analysis panel
            panel = create_detailed_analysis_panel("audio", mock_detection_result)
            
            return panel, {"display": "block"}
            
        except Exception as e:
            print(f"Error creating detailed analysis panel: {str(e)}")
            return html.Div([
                html.P("Error loading detailed analysis", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ]), {"display": "block"}
            
    # Add other callbacks for audio tab functionality
