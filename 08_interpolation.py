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

# Initialize the app - incorporate css
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=external_stylesheets)

# App layout
app.layout = [
    html.H1("Energy Insights", style={"textAlign": "center"}),
    html.H3("IEA Extended Dataset - Interpolation", style={"textAlign": "center"}),
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
    html.P("Select a flow: "),
    dcc.Dropdown(
        df["flow"].unique(),
        id="flow-dropdown",
        placeholder="Select a flow",
        value="Total energy supply",
    ),
    html.P("Select a product: "),
    dcc.Dropdown(
        df["product"].unique(),
        id="product-dropdown",
        placeholder="Select a product",
        value="Total",
    ),
    dcc.Graph(figure={}, id="plot-1"),
]


@callback(
    Output(component_id="plot-1", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
    Input(component_id="scenario-dropdown", component_property="value"),
    Input(component_id="flow-dropdown", component_property="value"),
    Input(component_id="product-dropdown", component_property="value"),
    Input(component_id="category-dropdown", component_property="value"),
)
def update_graph_1(
    selected_region,
    selected_scenario,
    selected_flow,
    selected_product,
    selected_category,
):
    # Filter the dataframe
    dff = df[df.flow == selected_flow]
    dff = dff[dff.category == selected_category]
    dff = dff[dff.scenario == selected_scenario]
    dff = dff[dff["product"] == selected_product]
    dff = dff[dff.region == selected_region].copy()

    # Transform the year to datetime
    dff["year"] = pd.to_datetime(dff["year"], format="%Y")

    # Store the original order of the columns
    cols = dff.columns

    # Set the year as the index, and then upsample to include the missing years
    dff = dff.set_index("year").resample("1YE").ffill()

    # Reset the index to return the year to its original place in the dataframe
    dff = dff.reset_index()
    dff = dff[cols]

    # Since we filled all the rows while upsampling, we need to drop the duplicated
    # values from the `value` column (replace them by NaN).
    # In this way we can choose the interpolation method later
    dff.loc[dff["value"].duplicated(keep="first"), "value"] = np.nan

    # Restore the year column to only display the year (no more datetime object)
    dff["year"] = pd.to_numeric(dff.year.dt.strftime("%Y"))

    # Different types of interpolation
    dff_linear = dff.copy()
    dff_linear["value"] = dff.value.interpolate(method="linear")

    dff_slinear = dff.copy()
    dff_slinear["value"] = dff.value.interpolate(method="slinear")

    dff_spline = dff.copy()
    dff_spline["value"] = dff.value.interpolate(method="spline", order=3)

    dff_quad = dff.copy()
    dff_quad["value"] = dff.value.interpolate(method="quadratic")

    dff_cubic = dff.copy()
    dff_cubic["value"] = dff.value.interpolate(method="cubic")

    dff_bary = dff.copy()
    dff_bary["value"] = dff.value.interpolate(method="barycentric")

    dff_poly = dff.copy()
    dff_poly["value"] = dff.value.interpolate(method="polynomial", order=2)

    mask = dff.value.isna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Original",
            x=dff["year"][~mask],
            y=dff["value"][~mask],
            marker=dict(size=12),
            mode="markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Linear",
            x=dff_linear["year"],
            y=dff_linear["value"],
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Slinear",
            x=dff_slinear["year"],
            y=dff_slinear["value"],
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Spline",
            x=dff_spline["year"],
            y=dff_spline["value"],
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Quadratic",
            x=dff_quad["year"],
            y=dff_quad["value"],
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Cubic",
            x=dff_cubic["year"],
            y=dff_cubic["value"],
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Barycentric",
            x=dff_bary["year"],
            y=dff_bary["value"],
            mode="lines",
            visible="legendonly",
        )
    )

    fig.update_layout(
        title=f"{selected_flow} - {selected_product}",
        hovermode="x unified",
        yaxis_title=f"{dff.unit.unique()[0]}",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


@callback(
    Output(component_id="flow-dropdown", component_property="options"),
    Output(component_id="flow-dropdown", component_property="value"),
    Input(component_id="category-dropdown", component_property="value"),
)
def update_flow_dropdown(selected_category):
    dff = df[df.category == selected_category]
    options = dff.flow.unique()
    # Update the list of options and select the first one
    return options, options[0]


@callback(
    Output(component_id="product-dropdown", component_property="options"),
    Output(component_id="product-dropdown", component_property="value"),
    Input(component_id="flow-dropdown", component_property="value"),
)
def update_product_dropdown(selected_flow):
    dff = df[df.flow == selected_flow]
    options = dff["product"].unique()
    return options, options[0]


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
