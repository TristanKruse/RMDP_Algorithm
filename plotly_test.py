import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample Data
df = px.data.iris()

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div(
    [
        dcc.Dropdown(
            id="dropdown",
            options=[{"label": col, "value": col} for col in df.columns if df[col].dtype != "object"],
            value="sepal_width",
        ),
        dcc.Graph(id="graph"),
    ]
)


# Callback for interactivity
@app.callback(Output("graph", "figure"), Input("dropdown", "value"))
def update_graph(selected_feature):
    fig = px.histogram(df, x=selected_feature, title=f"Histogram of {selected_feature}")
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
