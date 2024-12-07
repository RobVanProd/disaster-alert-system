"""
Web Dashboard for Disaster Alert System
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

from alerts.alert_manager import AlertManager, AlertType, AlertSeverity
from data.collectors import SeismicDataCollector, WeatherDataCollector

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Disaster Alert System"
)

# Layout components
header = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Disaster Alert System", className="text-white")),
            dbc.Col(
                dbc.Button("Refresh Data", color="light", className="ml-auto"),
                width={"size": 2}
            )
        ])
    ]),
    dark=True,
    color="dark",
    className="mb-4"
)

alert_card = dbc.Card([
    dbc.CardHeader("Active Alerts"),
    dbc.CardBody(id="alert-content")
], className="mb-4")

map_card = dbc.Card([
    dbc.CardHeader("Disaster Map"),
    dbc.CardBody([
        dcc.Graph(id="disaster-map")
    ])
], className="mb-4")

stats_row = dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Earthquake Statistics"),
            dbc.CardBody(id="earthquake-stats")
        ]),
        width=6
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Weather Alerts"),
            dbc.CardBody(id="weather-stats")
        ]),
        width=6
    )
])

# Main layout
app.layout = dbc.Container([
    header,
    alert_card,
    map_card,
    stats_row,
    dcc.Interval(
        id="interval-component",
        interval=30*1000,  # 30 seconds
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    Output("alert-content", "children"),
    Input("interval-component", "n_intervals")
)
def update_alerts(n):
    """Update active alerts"""
    # TODO: Get real alert data
    alerts = [
        {"title": "Strong Earthquake", "severity": "HIGH", "location": "Los Angeles"},
        {"title": "Severe Weather", "severity": "MEDIUM", "location": "Miami"}
    ]
    
    return dbc.ListGroup([
        dbc.ListGroupItem(
            [
                html.H5(alert["title"]),
                html.P(f"Severity: {alert['severity']}"),
                html.P(f"Location: {alert['location']}")
            ],
            color="danger" if alert["severity"] == "HIGH" else "warning"
        )
        for alert in alerts
    ])

@app.callback(
    Output("disaster-map", "figure"),
    Input("interval-component", "n_intervals")
)
def update_map(n):
    """Update disaster map"""
    # TODO: Get real location data
    df = pd.DataFrame({
        'lat': [34.0522, 25.7617],
        'lon': [-118.2437, -80.1918],
        'type': ['Earthquake', 'Weather'],
        'magnitude': [6.2, None],
        'description': ['Strong Earthquake', 'Hurricane Warning']
    })
    
    fig = go.Figure()
    
    # Add earthquake markers
    earthquakes = df[df['type'] == 'Earthquake']
    fig.add_trace(go.Scattergeo(
        lon=earthquakes['lon'],
        lat=earthquakes['lat'],
        text=earthquakes['description'],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        ),
        name='Earthquakes'
    ))
    
    # Add weather markers
    weather = df[df['type'] == 'Weather']
    fig.add_trace(go.Scattergeo(
        lon=weather['lon'],
        lat=weather['lat'],
        text=weather['description'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='triangle-up'
        ),
        name='Weather Events'
    ))
    
    fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(243, 243, 243)',
            oceancolor='rgb(204, 229, 255)',
            showframe=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

@app.callback(
    [Output("earthquake-stats", "children"),
     Output("weather-stats", "children")],
    Input("interval-component", "n_intervals")
)
def update_stats(n):
    """Update statistics"""
    # TODO: Get real statistics
    earthquake_stats = html.Div([
        html.H4("6.2", className="text-danger"),
        html.P("Strongest Recent Magnitude"),
        html.H4("12", className="text-warning"),
        html.P("Events in Last 24 Hours")
    ])
    
    weather_stats = html.Div([
        html.H4("3", className="text-danger"),
        html.P("Active Weather Alerts"),
        html.H4("2", className="text-warning"),
        html.P("Regions Affected")
    ])
    
    return earthquake_stats, weather_stats

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
