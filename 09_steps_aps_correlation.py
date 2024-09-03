# Import packages

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

LABELS = {"Stated Policies Scenario": "STEPS", "Announced Pledges Scenario": "APS"}

# Incorporate data
df = pd.read_csv(r"./data/WEO2023_Extended_Data_Regions_Full_Sorted.csv")

# Initialize the app - incorporate css
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=external_stylesheets)

# App layout
app.layout = [
    html.H1("Energy Insights", style={"textAlign": "center"}),
    html.H3("IEA Extended Dataset - Correlation Model", style={"textAlign": "center"}),
    html.P("Select a region: "),
    dcc.Dropdown(
        df.region.unique(),
        id="region-dropdown",
        placeholder="Select a region",
        value="North America",
    ),
    dcc.Graph(figure={}, id="plot-1", style={"width": "100vh", "height": "60vh"}),
]


@callback(
    Output(component_id="plot-1", component_property="figure"),
    Input(component_id="region-dropdown", component_property="value"),
)
def update_graph_1(selected_region):
    df_tes_model_adjusted = main(selected_region)

    fig = go.Figure()

    # Plot actual TES values
    for scenario in df_tes_model_adjusted["scenario"].unique():
        scenario_df = df_tes_model_adjusted[
            df_tes_model_adjusted["scenario"] == scenario
        ]
        fig.add_trace(
            go.Scatter(
                name=f"Actual {LABELS[scenario]}",
                x=scenario_df["year"],
                y=scenario_df["TES"],
                mode="lines+markers",
            )
        )
    # Plot predicted TES adjusted values
    for scenario in df_tes_model_adjusted["scenario"].unique():
        scenario_df = df_tes_model_adjusted[
            df_tes_model_adjusted["scenario"] == scenario
        ]
        fig.add_trace(
            go.Scatter(
                name=f"Predicted {LABELS[scenario]}",
                x=scenario_df["year"],
                y=scenario_df["predicted_TES_adjusted"],
                mode="lines+markers",
                line=dict(dash="dash"),
                marker=dict(symbol="x", size=8),
            )
        )

    fig.update_layout(
        title=f"Actual TES vs Predicted TES Adjusted for {selected_region}",
        hovermode="x unified",
        yaxis_title=f"TES (PJ)",
        xaxis_title=f"Year",
    )
    fig.update(layout={"title": {"x": 0.5}})  # Center the title
    return fig


# ---------------------------------------------------
# ------------------ Data code ----------------------
# ---------------------------------------------------


def get_total_energy_supply(df, region_name):
    """
    This function filters the dataset to return the Total Energy Supply (TES)
    for a specific region, ensuring the output contains only the columns:
    region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing the TES data with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by category 'Energy'
    filtered_df = filtered_df[filtered_df["category"] == "Energy"]

    # Step 3: Filter by product 'Total'
    filtered_df = filtered_df[filtered_df["product"] == "Total"]

    # Step 4: Filter by flow 'Total energy supply'
    tes_df = filtered_df[filtered_df["flow"] == "Total energy supply"]

    # Step 5: Select only the necessary columns
    tes_df = tes_df[["region", "scenario", "year", "value"]]

    return tes_df


def get_co2_total(df, region_name):
    """
    This function filters the dataset to return the CO2 total
    for a specific region, ensuring the output contains only the columns:
    region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing the CO2 total data with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by category 'CO2 total'
    filtered_df = filtered_df[filtered_df["category"] == "CO2 total"]

    # Step 3: Filter by product 'Total'
    filtered_df = filtered_df[filtered_df["product"] == "Total"]

    # Step 4: Filter by flow 'Total energy supply'
    co2_df = filtered_df[filtered_df["flow"] == "Total energy supply"]

    # Step 5: Select only the necessary columns
    co2_df = co2_df[["region", "scenario", "year", "value"]]

    return co2_df


def get_gdp_per_capita(df, region_name):
    """
    This function filters the dataset to return the GDP per capita
    for a specific region, ensuring the output contains only the columns:
    region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing the GDP per capita data with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by flow 'GDP per capita'
    gdp_df = filtered_df[filtered_df["flow"] == "GDP per capita"]

    # Step 3: Select only the necessary columns
    gdp_df = gdp_df[["region", "scenario", "year", "value"]]

    return gdp_df


def get_ee1(df, region_name):
    """
    This function calculates ee1 by summing 'Services' and 'Residential'
    for 'Floorspace' in the product column, filtered by the given region,
    ensuring the output contains only the columns: region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing ee1 with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by product 'Floorspace'
    filtered_df = filtered_df[filtered_df["product"] == "Floorspace"]

    # Step 3: Extract 'Services' and 'Residential' from the flow column
    services_df = filtered_df[filtered_df["flow"] == "Services"]
    residential_df = filtered_df[filtered_df["flow"] == "Residential"]

    # Step 4: Summing Services and Residential to calculate ee1
    ee1_df = services_df.copy()
    ee1_df["value"] = services_df["value"] + residential_df["value"].values

    # Step 5: Select only the necessary columns
    ee1_df = ee1_df[["region", "scenario", "year", "value"]]

    return ee1_df


def get_ee2(df, region_name):
    """
    This function calculates ee2 by summing 'Road passenger light duty vehicle'
    and 'Road freight trucks' for 'Activity of stock' in the category column,
    filtered by the given region, ensuring the output contains only the columns:
    region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing ee2 with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by category 'Activity of stock'
    filtered_df = filtered_df[filtered_df["category"] == "Activity of stock"]

    # Step 3: Extract 'Road passenger light duty vehicle' and 'Road freight trucks' from the flow column
    passenger_df = filtered_df[
        filtered_df["flow"] == "Road passenger light duty vehicle"
    ]
    freight_df = filtered_df[filtered_df["flow"] == "Road freight trucks"]

    # Step 4: Summing Road passenger light duty vehicle and Road freight trucks to calculate ee2
    ee2_df = passenger_df.copy()
    ee2_df["value"] = passenger_df["value"] + freight_df["value"].values

    # Step 5: Select only the necessary columns
    ee2_df = ee2_df[["region", "scenario", "year", "value"]]

    return ee2_df


def get_ee3(df, region_name):
    """
    This function calculates ee3 by summing 'Primary chemicals', 'Crude steel',
    'Cement', and 'Aluminium' for 'Industrial material production' in the category column,
    filtered by the given region, ensuring the output contains only the columns:
    region, scenario, year, and value.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame containing ee3 with the relevant columns.
    """
    # Step 1: Filter by region
    filtered_df = df[df["region"] == region_name]

    # Step 2: Filter by category 'Industrial material production'
    filtered_df = filtered_df[
        filtered_df["category"] == "Industrial material production"
    ]

    # Step 3: Extract relevant products
    chemicals_df = filtered_df[filtered_df["product"] == "Primary chemicals"]
    steel_df = filtered_df[filtered_df["product"] == "Crude steel"]
    cement_df = filtered_df[filtered_df["product"] == "Cement"]
    aluminium_df = filtered_df[filtered_df["product"] == "Aluminium"]

    # Step 4: Summing the values to calculate ee3
    ee3_df = chemicals_df.copy()
    ee3_df["value"] = (
        chemicals_df["value"].values
        + steel_df["value"].values
        + cement_df["value"].values
        + aluminium_df["value"].values
    )

    # Step 5: Select only the necessary columns
    ee3_df = ee3_df[["region", "scenario", "year", "value"]]

    return ee3_df


def create_df_ee(df, region_name):
    """
    This function creates a new DataFrame (df_ee) that combines ee1, ee2, and ee3
    into a single DataFrame for the specified region. The resulting DataFrame contains
    the columns: region, scenario, year, ee1, ee2, and ee3.

    Parameters:
    - df: The original DataFrame containing the data.
    - region_name: The name of the region you want to filter by.

    Returns:
    - A DataFrame (df_ee) containing the combined data for ee1, ee2, and ee3.
    """
    # Calculate ee1, ee2, ee3
    ee1_df = get_ee1(df, region_name)
    ee2_df = get_ee2(df, region_name)
    ee3_df = get_ee3(df, region_name)

    # Merge dataframes on common keys (region, scenario, year)
    df_ee = ee1_df.merge(
        ee2_df, on=["region", "scenario", "year"], suffixes=("_ee1", "_ee2")
    )
    df_ee = df_ee.merge(ee3_df, on=["region", "scenario", "year"])

    # Rename columns for clarity
    df_ee.rename(
        columns={"value_ee1": "ee1", "value_ee2": "ee2", "value": "ee3"}, inplace=True
    )

    return df_ee


def calculate_ee_from_df_ee(df_ee):
    """
    This function calculates the Energy Efficiency (EE) using PCA on ee1, ee2, and ee3
    from the DataFrame (df_ee), ensuring the output contains only the columns:
    region, scenario, year, and EE.

    Parameters:
    - df_ee: The DataFrame containing ee1, ee2, and ee3.

    Returns:
    - A DataFrame with the calculated EE for each year and scenario with the relevant columns.
    """
    # Step 1: Standardization
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_ee[["ee1", "ee2", "ee3"]])

    # Step 2: Perform PCA
    pca = PCA(n_components=1)
    df_ee["EE"] = pca.fit_transform(standardized_data)

    # Select only the necessary columns
    df_ee = df_ee[["region", "scenario", "year", "EE"]]

    return df_ee


def create_tes_model_df(tes_df, co2_df, gdp_df, ee_final_df):
    """
    This function merges the TES, CO2 total, GDP per capita, and EE DataFrames
    into a single DataFrame (df_tes_model) for modeling.

    Parameters:
    - tes_df: DataFrame containing TES values.
    - co2_df: DataFrame containing CO2 total values.
    - gdp_df: DataFrame containing GDP per capita values.
    - ee_final_df: DataFrame containing EE values.

    Returns:
    - A DataFrame (df_tes_model) containing all the combined data.
    """
    # Merge dataframes on common keys (region, scenario, year)
    df_tes_model = tes_df.merge(
        co2_df[["region", "scenario", "year", "value"]],
        on=["region", "scenario", "year"],
        suffixes=("", "_CO2"),
    )
    df_tes_model = df_tes_model.merge(
        gdp_df[["region", "scenario", "year", "value"]],
        on=["region", "scenario", "year"],
        suffixes=("", "_GDP"),
    )
    df_tes_model = df_tes_model.merge(
        ee_final_df[["region", "scenario", "year", "EE"]],
        on=["region", "scenario", "year"],
    )

    # Rename columns for clarity
    df_tes_model.rename(
        columns={
            "value": "TES",
            "value_CO2": "CO2_total",
            "value_GDP": "GDP_per_capita",
        },
        inplace=True,
    )

    return df_tes_model


def create_synthetic_average(df_tes_model):
    """
    This function calculates the synthetic average for each year by averaging
    the TES, CO2_total, GDP_per_capita, and EE values across the two scenarios.

    Parameters:
    - df_tes_model: The DataFrame containing TES, CO2_total, GDP_per_capita, and EE data for both scenarios.

    Returns:
    - A new DataFrame containing the average values for each year.
    """
    avg_df = (
        df_tes_model.groupby("year")
        .agg(
            avg_TES=("TES", "mean"),
            avg_CO2_total=("CO2_total", "mean"),
            avg_GDP_per_capita=("GDP_per_capita", "mean"),
            avg_EE=("EE", "mean"),
        )
        .reset_index()
    )

    return avg_df


def standardize_variables(avg_df):
    """
    This function standardizes the variables in the average DataFrame.

    Parameters:
    - avg_df: The DataFrame containing the synthetic average data.

    Returns:
    - A DataFrame with standardized values for CO2_total, GDP_per_capita, and EE.
    """
    scaler = StandardScaler()
    avg_df[["avg_CO2_total", "avg_GDP_per_capita", "avg_EE"]] = scaler.fit_transform(
        avg_df[["avg_CO2_total", "avg_GDP_per_capita", "avg_EE"]]
    )

    return avg_df


def fit_best_tes_model(avg_df, max_degree=4):
    """
    This function fits several regression models (linear and polynomial of different degrees)
    to predict TES as a function of CO2_total, GDP_per_capita, and EE, and selects the best one
    based on Mean Squared Error (MSE).

    Parameters:
    - avg_df: The DataFrame containing the synthetic average data with standardized variables.
    - max_degree: The maximum degree of the polynomial regression to consider.

    Returns:
    - The best fitted regression model, along with its performance metrics (MSE and R-squared).
    """
    X = avg_df[["avg_CO2_total", "avg_GDP_per_capita", "avg_EE"]]
    y = avg_df["avg_TES"]

    best_mse = np.inf
    best_model = None
    best_degree = 0
    best_r2 = 0

    for degree in range(1, max_degree + 1):
        if degree == 1:
            # Linear regression model
            poly = PolynomialFeatures(degree=1, include_bias=False)
        else:
            # Polynomial regression model
            poly = PolynomialFeatures(degree=degree)

        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Degree {degree}: MSE = {mse}, R-squared = {r2}")

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_degree = degree
            best_r2 = r2

    print(f"\nBest Model: Degree {best_degree}")
    print(f"Best Model MSE: {best_mse}")
    print(f"Best Model R-squared: {best_r2}")

    return best_model, best_degree, best_mse, best_r2


def predict_tes_using_best_model(avg_df, best_model, best_degree):
    """
    This function predicts TES using the best fitted model for the synthetic average scenario.

    Parameters:
    - avg_df: The DataFrame containing the synthetic average data with standardized variables.
    - best_model: The best regression model fitted to the data.
    - best_degree: The degree of the polynomial used in the best model.

    Returns:
    - A DataFrame with the predicted TES values.
    """
    X = avg_df[["avg_CO2_total", "avg_GDP_per_capita", "avg_EE"]]
    poly = PolynomialFeatures(degree=best_degree)
    X_poly = poly.fit_transform(X)

    avg_df["predicted_TES"] = best_model.predict(X_poly)

    return avg_df


def objective_function(params, avg_df, df_tes_model):
    """
    Objective function for optimization to find the best multipliers a and b.

    Parameters:
    - params: The multipliers [a, b] being optimized.
    - avg_df: The DataFrame containing the predicted TES for the average scenario.
    - df_tes_model: The original DataFrame with actual TES values for both scenarios.

    Returns:
    - The total mean squared error for the STEPS and APS scenarios.
    """
    a, b = params
    mse_steps = mse_aps = 0

    for scenario, multiplier in [
        ("Stated Policies Scenario", a),
        ("Announced Pledges Scenario", b),
    ]:
        actual_tes = df_tes_model[df_tes_model["scenario"] == scenario]["TES"].values
        predicted_tes = avg_df["predicted_TES"].values * multiplier
        mse = mean_squared_error(actual_tes, predicted_tes)

        if scenario == "Stated Policies Scenario":
            mse_steps = mse
        else:
            mse_aps = mse

    total_mse = mse_steps + mse_aps
    return total_mse


def apply_multipliers_with_difference(avg_df, df_tes_model, a, b):
    """
    Apply the optimized multipliers to predict TES for both scenarios and calculate the percentage difference.

    Parameters:
    - avg_df: The DataFrame containing the predicted TES for the average scenario.
    - df_tes_model: The original DataFrame with actual TES values for both scenarios.
    - a: Optimized multiplier for STEPS.
    - b: Optimized multiplier for APS.

    Returns:
    - A DataFrame with the adjusted TES predictions for both scenarios and the percentage difference.
    """
    df_tes_model["predicted_TES_adjusted"] = 0

    for scenario, multiplier in [
        ("Stated Policies Scenario", a),
        ("Announced Pledges Scenario", b),
    ]:
        df_tes_model.loc[
            df_tes_model["scenario"] == scenario, "predicted_TES_adjusted"
        ] = (avg_df["predicted_TES"].values * multiplier)

    # Calculate percentage difference
    df_tes_model["percentage_difference"] = (
        100
        * (df_tes_model["predicted_TES_adjusted"] - df_tes_model["TES"])
        / df_tes_model["TES"]
    )

    return df_tes_model


def main(region_name):
    tes_df = get_total_energy_supply(df, region_name)
    co2_df = get_co2_total(df, region_name)
    gdp_df = get_gdp_per_capita(df, region_name)
    ee1_df = get_ee1(df, region_name)
    ee2_df = get_ee2(df, region_name)
    ee3_df = get_ee3(df, region_name)
    df_ee = create_df_ee(df, region_name)
    ee_final_df = calculate_ee_from_df_ee(df_ee)
    df_tes_model = create_tes_model_df(tes_df, co2_df, gdp_df, ee_final_df)
    avg_df = create_synthetic_average(df_tes_model)
    avg_df_standardized = standardize_variables(avg_df)
    best_model, best_degree, best_mse, best_r2 = fit_best_tes_model(avg_df_standardized)
    avg_df_with_predictions = predict_tes_using_best_model(
        avg_df_standardized, best_model, best_degree
    )

    # Set the initial guess for the multipliers a and b
    initial_guess = [1.0, 1.0]

    # Perform the optimization
    result = minimize(
        objective_function,
        initial_guess,
        args=(avg_df_with_predictions, df_tes_model),
        method="BFGS",
    )

    # Extract the optimal multipliers
    a_opt, b_opt = result.x

    # Apply the optimized multipliers and calculate percentage difference
    df_tes_model_adjusted = apply_multipliers_with_difference(
        avg_df_with_predictions, df_tes_model, a_opt, b_opt
    )

    return df_tes_model_adjusted


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
