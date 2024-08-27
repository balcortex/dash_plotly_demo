# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from statistics import NormalDist
import numpy as np

# Incorporate data
df = pd.read_csv(r"./data/Aug13.csv")

# Initialize the app - incorporate css
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=external_stylesheets)

# App layout
app.layout = [
    html.H1("Energy Insights", style={"textAlign": "center"}),
    html.H3("IEA Extended Dataset", style={"textAlign": "center"}),
    html.Div(
        children=[
            html.P("Select a region: "),
            dcc.Dropdown(
                df.region.unique(),
                id="region-dropdown",
                placeholder="Select a region",
                value="EU",
                style={"width": "200px"},
            ),
        ],
    ),
    html.Div(
        children=[
            html.P("Select an scenario: "),
            dcc.RadioItems(
                options=["STEPS", "APS"],
                value="STEPS",
                id="radio-plot-2",
            ),
        ]
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="five columns",
                children=[dcc.Graph(figure={}, id="plot-1")],
            ),
            html.Div(
                className="five columns",
                children=[
                    dcc.Graph(figure={}, id="plot-2"),
                ],
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="five columns",
                children=[dcc.Graph(figure={}, id="plot-3")],
            ),
            html.Div(
                className="five columns",
                children=[
                    dcc.Graph(figure={}, id="plot-4"),
                    html.P("Prediction interval: "),
                    dcc.RadioItems(
                        options=[0.90, 0.95, 0.99],
                        value=0.95,
                        id="radio-prediction-interval",
                        inline=True,
                    ),
                ],
            ),
        ],
    ),
]


# Add controls to build the interaction
@callback(
    Output(component_id="plot-1", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
)
def update_graph_1(selected_region):
    dff = df[df["region"] == selected_region]
    dff_steps = dff[dff["scenario"] == "STEPS"]
    dff_aps = dff[dff["scenario"] == "APS"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="STEPS",
            x=dff_steps["year"],
            y=dff_steps["total_co2"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name="APS",
            x=dff_aps["year"],
            y=dff_aps["total_co2"],
        )
    )
    fig.update_layout(
        title="Total CO₂ Emissions Over the Years",
        xaxis_title="Year",
        yaxis_title="Total CO₂ (Million Metric Tons)",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


@callback(
    Output(component_id="plot-2", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="radio-plot-2", component_property="value"),
)
def update_graph_2(selected_region, selected_scenario):
    dff = df[df["region"] == selected_region]
    dff = dff[dff["scenario"] == selected_scenario]

    # Reshape the data for scikit-learn
    X = dff["year"].values.reshape(-1, 1)
    y = dff["total_co2"].values

    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Make the prediction
    y_pred = model.predict(X)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Original",
            x=dff["year"],
            y=dff["total_co2"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Model",
            x=dff["year"].values,
            y=y_pred,
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Total CO₂ Emissions Over the Years with Regression Line",
        xaxis_title="Year",
        yaxis_title="Total CO₂ (Million Metric Tons)",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


@callback(
    Output(component_id="plot-3", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="radio-plot-2", component_property="value"),
)
def update_graph_3(selected_region, selected_scenario):
    dff = df[df["region"] == selected_region]
    dff = dff[dff["scenario"] == selected_scenario]

    # Reshape the data for scikit-learn
    X = dff["year"].values.reshape(-1, 1)
    y = dff["total_co2"].values
    x = list(dff["year"].values)

    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Make the prediction
    y_pred = model.predict(X)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Original",
            x=dff["year"],
            y=dff["total_co2"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Model",
            x=dff["year"].values,
            y=y_pred,
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            name="15% Uncert",
            x=x + x[::-1],
            y=list(y_pred * 1.15) + list(y_pred * 0.85)[::-1],
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line_color="rgba(255,255,255,0)",
            # showlegend=False,
        )
    )

    fig.update_layout(
        title="Total CO₂ Emissions Over the Years with Regression Line",
        xaxis_title="Year",
        yaxis_title="Total CO₂ (Million Metric Tons)",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


@callback(
    Output(component_id="plot-4", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="radio-plot-2", component_property="value"),
    Input(component_id="radio-prediction-interval", component_property="value"),
)
def update_graph_4(selected_region, selected_scenario, selected_pi):
    dff = df[df["region"] == selected_region]
    dff = dff[dff["scenario"] == selected_scenario]

    # Reshape the data for scikit-learn
    X = dff["year"].values.reshape(-1, 1)
    y = dff["total_co2"].values
    x = list(dff["year"].values)

    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Make the prediction
    y_pred = model.predict(X)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Original",
            x=dff["year"],
            y=dff["total_co2"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Model",
            x=dff["year"].values,
            y=y_pred,
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    CI = confidence_interval(y, y_pred, selected_pi)
    fig.add_trace(
        go.Scatter(
            name="upper",
            x=dff["year"],
            y=y_pred + CI,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["year"],
            y=y_pred - CI,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name=f"{selected_pi*100:.0f}% PI",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.2)",
        )
    )

    fig.update_layout(
        title="Total CO₂ Emissions Over the Years with Regression Line",
        xaxis_title="Year",
        yaxis_title="Total CO₂ (Million Metric Tons)",
        # hovermode="x",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


def confidence_interval(y, yhat, confidence=0.95):
    # https://machinelearningmastery.com/prediction-intervals-for-machine-learning/

    z = NormalDist().inv_cdf((1 + confidence) / 2.0)
    sum_errs = np.sum((y - yhat) ** 2)
    stdev = np.sqrt(1 / (len(y) - 2) * sum_errs)

    return z * stdev


# Run the app
if __name__ == "__main__":
    app.run(debug=True)


# - - - - - -
# Unused code
# - - - - - -

# @callback(
#     Output(component_id="plot-4", component_property="figure"),
#     Input(component_id="region-dropdown", component_property="value"),
#     Input(component_id="radio-plot-2", component_property="value"),
#     Input(component_id="radio-prediction-interval", component_property="value"),
# )
# def update_graph_4(selected_region, selected_scenario, selected_pi):
#     dff = df[df["region"] == selected_region]
#     dff = dff[dff["scenario"] == selected_scenario]

#     # Reshape the data for scikit-learn
#     X = dff["year"].values.reshape(-1, 1)
#     y = dff["total_co2"].values
#     x = list(dff["year"].values)

#     # Create a linear regression model and fit it to the data
#     model = LinearRegression()
#     model.fit(X, y)

#     # Make the prediction
#     y_pred = model.predict(X)

#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             name="Original",
#             x=dff["year"],
#             y=dff["total_co2"],
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             name="Model",
#             x=dff["year"].values,
#             y=y_pred,
#             mode="lines",
#             line=dict(dash="dash"),
#         )
#     )
#     CI = confidence_interval(y_pred, selected_pi)
#     fig.add_trace(
#         go.Scatter(
#             name="upper",
#             x=dff["year"],
#             y=y_pred + CI,
#             mode="lines",
#             line_color="rgba(0,0,0,0)",
#             showlegend=False,
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=dff["year"],
#             y=y_pred - CI,
#             mode="lines",
#             line_color="rgba(0,0,0,0)",
#             name=f"{selected_pi*100:.0f}% PI",
#             fill="tonexty",
#             fillcolor="rgba(255, 0, 0, 0.2)",
#         )
#     )

#     fig.update_layout(
#         title="Total CO₂ Emissions Over the Years with Regression Line",
#         xaxis_title="Year",
#         yaxis_title="Total CO₂ (Million Metric Tons)",
#     )
#     fig.update(layout={"title": {"x": 0.5}})  # Center the title
#     return fig

# html.Div(
#     className="row",
#     children=[
#         html.Div(
#             className="five columns",
#             children=[
#                 dcc.Graph(figure={}, id="plot-4"),
#                 dcc.RadioItems(
#                     options=[0.90, 0.95, 0.99],
#                     value=0.95,
#                     id="radio-prediction-interval",
#                     inline=True,
#                 ),
#             ],
#         ),
#     ],
# ),


# def confidence_interval(data, confidence=0.95):
#     # https://stackoverflow.com/questions/70076213/how-to-add-95-confidence-interval-for-a-line-chart-in-plotly
#     dist = NormalDist.from_samples(data)
#     z = NormalDist().inv_cdf((1 + confidence) / 2.0)
#     h = dist.stdev * z / ((len(data) - 1) ** 0.5)
#     return h
