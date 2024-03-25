# Import necessary libraries
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Load your dataset
df = pd.read_csv('SP 500 ESG Risk Ratings.csv')

# Ensure there are no NaN values in the 'Sector' column
df = df.dropna(subset=['Sector'])

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Modal for ESG Score Information
info_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("About ESG Scores")),
        dbc.ModalBody(
            "Environmental, Social, and Governance (ESG) scores help measure a company's "
            "sustainability and societal impact. These scores are crucial for investors "
            "looking to support responsible businesses."
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="modal",
    is_open=False,  # Starts closed
)

# Define the layout of the app
app.layout = dbc.Container([
    html.H1('Green Investment Dashboard'),
    html.Div('Visualizing ESG Data for S&P 500 Companies.'),
    dbc.Button("Learn About ESG Scores", id="open-modal", n_clicks=0),
    info_modal,
    dbc.Tabs([
        dbc.Tab(label="Sector Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select Sector:'),
                    dcc.Dropdown(
                        id='sector-dropdown',
                        options=[{'label': sector, 'value': sector} for sector in df['Sector'].unique()],
                        value=df['Sector'].unique().tolist(),
                        multi=True
                    ),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='esg-score-by-sector'), width=12),
            ]),
        ]),
        dbc.Tab(label="More Analysis", children=[
            # Add more content or tabs for additional analyses
            html.P("Additional insights and analyses can be added here.")
        ]),
    ]),
], fluid=True)

# Callback to toggle the modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback to update graph based on dropdown selection
@app.callback(
    Output('esg-score-by-sector', 'figure'),
    [Input('sector-dropdown', 'value')]
)
def update_graph(selected_sectors):
    filtered_df = df[df['Sector'].isin(selected_sectors)]
    fig = px.bar(filtered_df, x='Sector', y='Total ESG Risk score', color='Sector',
                 labels={'Total ESG Risk score': 'Average ESG Score'}, height=400)
    fig.update_layout(transition_duration=500)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
