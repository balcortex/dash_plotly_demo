# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from statistics import NormalDist
import numpy as np

# Incorporate data
df = pd.read_csv(r"./data/WEO2023_Extended_Data_Regions_Full_Sorted.csv")
df = df.drop(df[df.scenario == "Net Zero Emissions by 2050 Scenario"].index)

# Initialize the app - incorporate css
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=external_stylesheets)

# App layout
app.layout = [
    html.H1("Energy Insights", style={"textAlign": "center"}),
    html.H3("IEA Extended Dataset", style={"textAlign": "center"}),
    html.P("Select a region: "),
    dcc.Dropdown(
        df.region.unique(),
        id="region-dropdown",
        placeholder="Select a region",
        value="North America",
    ),
    html.P("Select an scenario: "),
    dcc.Dropdown(
        df.scenario.unique(),
        id="scenario-dropdown",
        placeholder="Select an scenario",
        value="Stated Policies Scenario",
    ),
    html.P("Select a category: "),
    dcc.Dropdown(
        df["category"].unique(),
        id="category-dropdown",
        placeholder="Select a category",
        value="Energy",
    ),
    html.P("Select a product: "),
    dcc.Dropdown(
        df["product"].unique(),
        id="product-dropdown",
        placeholder="Select a product",
        value="Total",
    ),
    html.P("Select a flow: "),
    dcc.Dropdown(
        df["flow"].unique(),
        id="flow-dropdown",
        placeholder="Select a flow",
        value="Total energy supply",
    ),
    dcc.Graph(figure={}, id="plot-1"),
    dcc.Graph(figure={}, id="plot-2"),
    html.P("Prediction Interval"),
    dcc.RadioItems(
        options=[0.90, 0.95, 0.99],
        value=0.95,
        id="radio-prediction-interval-plot-2",
        inline=True,
    ),
]


@callback(
    Output(component_id="plot-1", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="flow-dropdown", component_property="value"),
    Input(component_id="product-dropdown", component_property="value"),
    Input(component_id="category-dropdown", component_property="value"),
)
def update_graph_1(
    selected_region,
    selected_flow,
    selected_product,
    selected_category,
):
    dff = df[df.flow == selected_flow]
    dff = dff[dff.category == selected_category]
    dff = dff[dff["product"] == selected_product]
    dff = dff[dff["region"] == selected_region]

    dff_steps = dff[dff["scenario"] == "Stated Policies Scenario"]
    dff_aps = dff[dff["scenario"] == "Announced Pledges Scenario"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="STEPS",
            x=dff_steps["year"],
            y=dff_steps["value"],
        )
    )
    fig.add_trace(
        go.Scatter(
            name="APS",
            x=dff_aps["year"],
            y=dff_aps["value"],
        )
    )
    fig.update_layout(
        title=f"{selected_flow}",
        hovermode="x unified",
        # xaxis_title="Year",
        # yaxis_title="Total Energy Supply (PJ)",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


@callback(
    Output(component_id="plot-2", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="scenario-dropdown", component_property="value"),
    Input(component_id="radio-prediction-interval-plot-2", component_property="value"),
    Input(component_id="flow-dropdown", component_property="value"),
    Input(component_id="product-dropdown", component_property="value"),
    Input(component_id="category-dropdown", component_property="value"),
)
def update_graph_2(
    selected_region,
    selected_scenario,
    selected_pi,
    selected_flow,
    selected_product,
    selected_category,
):
    dff = df[df.flow == selected_flow]
    dff = dff[dff.category == selected_category]
    dff = dff[dff["product"] == selected_product]
    dff = dff[dff["region"] == selected_region]
    dff = dff[dff["scenario"] == selected_scenario]

    # Reshape the data for scikit-learn
    X = dff["year"].values.reshape(-1, 1)
    y = dff["value"].values

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
            y=dff["value"],
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
        title=f"{selected_flow}",
        hovermode="x unified",
        # xaxis_title="Year",
        # yaxis_title="Total Energy Supply (PJ)",
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
