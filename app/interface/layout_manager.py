"""
Layout manager for the Deepfake Detection Platform UI.
Centralizes layout creation and component integration.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from app.interface.components.home_tab import create_home_tab
from app.interface.components.image_tab import create_image_tab
from app.interface.components.audio_tab import create_audio_tab
from app.interface.components.video_tab import create_video_tab
from app.interface.components.reports_tab import create_reports_tab
from app.interface.components.about_tab import create_about_tab
from app.utils.visualization import VisualizationManager

def create_app_layout(app, processor, config):
    """
    Create the main application layout integrating all components.
    
    Args:
        app: Dash application instance
        processor: Media processor instance
        config: Application configuration
        
    Returns:
        Dash layout object
    """
    # Create visualization manager
    output_dir = config.get('reports', {}).get('output_dir', None)
    theme = config.get('visualization', {}).get('theme', 'tron_legacy')
    dpi = config.get('visualization', {}).get('dpi', 100)
    visualizer = VisualizationManager(output_dir=output_dir, theme=theme, dpi=dpi)
    
    # Initialize required components
    home_tab = create_home_tab(app)
    image_tab = create_image_tab(app)
    audio_tab = create_audio_tab(app)
    video_tab = create_video_tab(app)
    reports_tab = create_reports_tab(app, visualizer)
    about_tab = create_about_tab(app)
    
    # Define the overall layout
    layout = html.Div([
        # Navigation bar at the top
        dbc.Navbar(
            [
                # Logo/brand centered
                html.Div(
                    html.H2("DEEPFAKE DETECTION PLATFORM", className="navbar-brand mb-0"),
                    className="mx-auto"
                ),
            ],
            color="dark",
            dark=True,
            className="mb-0",
        ),
        
        # Main content container
        html.Div(
            [
                # Tabs navigation
                dcc.Tabs(id="tabs", value="home-tab", className="custom-tabs", children=[
                    # Home tab
                    dcc.Tab(
                        label="HOME",
                        value="home-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=home_tab
                    ),
                    
                    # Image Analysis tab
                    dcc.Tab(
                        label="IMAGE ANALYSIS",
                        value="image-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=image_tab
                    ),
                    
                    # Audio Analysis tab
                    dcc.Tab(
                        label="AUDIO ANALYSIS",
                        value="audio-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=audio_tab
                    ),
                    
                    # Video Analysis tab
                    dcc.Tab(
                        label="VIDEO ANALYSIS",
                        value="video-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=video_tab
                    ),
                    
                    # Reports tab
                    dcc.Tab(
                        label="REPORTS",
                        value="reports-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=reports_tab
                    ),
                    
                    # About tab
                    dcc.Tab(
                        label="ABOUT",
                        value="about-tab",
                        className="custom-tab",
                        selected_className="custom-tab-selected",
                        children=about_tab
                    ),
                ]),
                
                # Footer with version and attribution
                html.Footer(
                    [
                        html.P(f" 2025 Deepfake Detection Platform", className="m-0")
                    ],
                    className="footer"
                )
            ],
            id="main-container",
            className="container-fluid p-0"
        ),

        # Hidden divs for storing app state
        dcc.Store(id='selected-report-store'),
        dcc.Store(id='uploaded-image-store'),
        dcc.Store(id='uploaded-audio-store'),
        dcc.Store(id='uploaded-video-store'),
        dcc.Download(id='download-report')
    ])
    
    return layout
