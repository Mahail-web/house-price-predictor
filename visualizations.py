"""
Interactive Plotly visualizations for Exploratory Data Analysis (EDA),
Model Performance Evaluation, and Feature Coefficient Interpretability.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Theme palette
NAVY_PRIMARY = "#0F172A"
SLATE_TEXT = "#334155"
TEAL_ACCENT = "#0D9488"
GOLD_ACCENT = "#D97706"
CORAL_ACCENT = "#E11D48"
GRID_COLOR = "#E2E8F0"
PLOT_BG = "#FFFFFF"


def apply_common_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply consistent styling across all Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=16, color=NAVY_PRIMARY),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=SLATE_TEXT, size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor=NAVY_PRIMARY,
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) CHARTS
# ==============================================================================

def plot_price_distribution(df: pd.DataFrame, target_col: str = "price", currency_symbol: str = "$") -> go.Figure:
    """Plot distribution of house prices with mean and median markers."""
    valid_prices = df[df[target_col] > 0][target_col]
    mean_val = valid_prices.mean()
    median_val = valid_prices.median()

    fig = px.histogram(
        valid_prices,
        x=target_col,
        nbins=60,
        marginal="box",
        color_discrete_sequence=[TEAL_ACCENT],
        labels={target_col: f"House Price ({currency_symbol})"}
    )

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=GOLD_ACCENT,
        annotation_text=f"Mean: {currency_symbol}{mean_val:,.0f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=NAVY_PRIMARY,
        annotation_text=f"Median: {currency_symbol}{median_val:,.0f}",
        annotation_position="bottom right"
    )

    fig = apply_common_layout(fig, f"House Price Distribution ({currency_symbol})")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> go.Figure:
    """Plot interactive correlation heatmap for numerical property features."""
    valid_df = df[df["price"] > 0]
    if numeric_cols is None:
        numeric_cols = [
            "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
            "floors", "waterfront", "view", "condition", "sqft_basement",
            "yr_built", "yr_renovated"
        ]
    cols = [c for c in numeric_cols if c in valid_df.columns]
    corr_matrix = valid_df[cols].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="Blues",
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Pearson r")
    ))

    fig = apply_common_layout(fig, "Feature Correlation Heatmap (Pearson r)", height=500)
    return fig


def plot_living_area_vs_price(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    """Scatter plot of interior living area (sqft_living) vs house price with OLS trendline."""
    valid_df = df[(df["price"] > 0) & (df["sqft_living"] > 0)].copy()

    fig = px.scatter(
        valid_df,
        x="sqft_living",
        y="price",
        color_discrete_sequence=["#38BDF8"],
        opacity=0.6,
        labels={
            "sqft_living": "Interior Living Area (Sq Ft)",
            "price": f"Price ({currency_symbol})"
        },
        hover_data=["city", "bedrooms", "bathrooms", "yr_built"]
    )

    # Compute OLS linear regression line via np.polyfit
    x_vals = valid_df["sqft_living"].values
    y_vals = valid_df["price"].values
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = slope * x_range + intercept

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_line,
        mode="lines",
        line=dict(color=GOLD_ACCENT, width=2.5),
        name=f"OLS Trendline (slope: +{currency_symbol}{slope:,.1f}/sqft)"
    ))

    fig = apply_common_layout(fig, "Living Area (Sq Ft) vs. House Price (with OLS Trendline)")
    return fig


def plot_price_by_city(df: pd.DataFrame, top_n: int = 15, currency_symbol: str = "$") -> go.Figure:
    """Bar chart showing median and mean house price across top cities."""
    valid_df = df[df["price"] > 0]
    city_stats = valid_df.groupby("city")["price"].agg(
        Median_Price="median",
        Mean_Price="mean",
        Count="count"
    ).reset_index()

    # Filter cities with at least 15 transactions for statistical stability
    city_stats = city_stats[city_stats["Count"] >= 15].sort_values(by="Median_Price", ascending=True).tail(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=city_stats["city"],
        x=city_stats["Median_Price"],
        orientation="h",
        name="Median Price",
        marker=dict(color=TEAL_ACCENT)
    ))
    fig.add_trace(go.Bar(
        y=city_stats["city"],
        x=city_stats["Mean_Price"],
        orientation="h",
        name="Mean Price",
        marker=dict(color="#94A3B8")
    ))

    fig.update_layout(
        barmode="group",
        xaxis_title=f"Price ({currency_symbol})",
        yaxis_title="City",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig = apply_common_layout(fig, f"Top {top_n} Cities by Housing Price (Median vs Mean)", height=480)
    return fig


def plot_price_by_waterfront_and_view(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    """Box plot of price stratified by waterfront presence and view score."""
    valid_df = df[df["price"] > 0].copy()
    valid_df["Waterfront_Status"] = valid_df["waterfront"].map({0: "No Waterfront", 1: "Waterfront"})

    fig = px.box(
        valid_df,
        x="view",
        y="price",
        color="Waterfront_Status",
        color_discrete_map={"No Waterfront": "#64748B", "Waterfront": TEAL_ACCENT},
        labels={
            "view": "View Rating (0 = None, 4 = Exceptional)",
            "price": f"Price ({currency_symbol})",
            "Waterfront_Status": "Waterfront"
        }
    )
    fig = apply_common_layout(fig, "House Price Distribution Across View Ratings & Waterfront")
    return fig


def plot_price_by_bedrooms(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    """Box plot showing house price distribution by number of bedrooms."""
    valid_df = df[(df["price"] > 0) & (df["bedrooms"] > 0) & (df["bedrooms"] <= 7)].copy()
    valid_df["bedrooms_str"] = valid_df["bedrooms"].astype(int).astype(str) + " Beds"

    fig = px.box(
        valid_df,
        x="bedrooms_str",
        y="price",
        color_discrete_sequence=[TEAL_ACCENT],
        category_orders={"bedrooms_str": [f"{i} Beds" for i in range(1, 8)]},
        labels={"bedrooms_str": "Bedrooms", "price": f"Price ({currency_symbol})"}
    )
    fig = apply_common_layout(fig, "House Price by Bedroom Count (1 to 7 Bedrooms)")
    return fig


# ==============================================================================
# MODEL EVALUATION CHARTS
# ==============================================================================

def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    currency_symbol: str = "$"
) -> go.Figure:
    """
    Plot actual vs. predicted prices on held-out test data.
    Includes the ideal 45-degree reference line (y = x).
    """
    df_eval = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_test_pred
    })

    min_val = min(df_eval["Actual"].min(), df_eval["Predicted"].min())
    max_val = max(df_eval["Actual"].max(), df_eval["Predicted"].max())

    fig = px.scatter(
        df_eval,
        x="Actual",
        y="Predicted",
        opacity=0.6,
        color_discrete_sequence=[TEAL_ACCENT],
        labels={
            "Actual": f"Actual Price ({currency_symbol})",
            "Predicted": f"Predicted Price ({currency_symbol})"
        }
    )

    # 45-degree reference line (perfect prediction)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=GOLD_ACCENT, dash="dash", width=2),
        name="Ideal 45° Line (y = x)"
    ))

    fig = apply_common_layout(fig, f"Actual vs. Predicted House Prices ({currency_symbol}) - Held-out Test Data")
    return fig


def plot_residuals_vs_predicted(
    y_test_pred: np.ndarray,
    residuals: pd.Series,
    currency_symbol: str = "$"
) -> go.Figure:
    """
    Plot residuals (Actual - Predicted) versus predicted values.
    Includes a reference line at residual = 0 to evaluate homoscedasticity.
    """
    df_res = pd.DataFrame({
        "Predicted": y_test_pred,
        "Residual": residuals.values
    })

    fig = px.scatter(
        df_res,
        x="Predicted",
        y="Residual",
        opacity=0.6,
        color_discrete_sequence=["#6366F1"],
        labels={
            "Predicted": f"Predicted Price ({currency_symbol})",
            "Residual": f"Residual [Actual - Predicted] ({currency_symbol})"
        }
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=CORAL_ACCENT,
        line_width=2,
        annotation_text="Zero Residual Reference",
        annotation_position="bottom right"
    )

    fig = apply_common_layout(fig, f"Residual Plot (Residuals vs. Predicted Price)")
    return fig


def plot_residual_distribution(residuals: pd.Series, currency_symbol: str = "$") -> go.Figure:
    """Plot distribution of residuals (prediction errors)."""
    mean_err = residuals.mean()
    std_err = residuals.std()

    fig = px.histogram(
        residuals,
        nbins=50,
        color_discrete_sequence=[TEAL_ACCENT],
        labels={"value": f"Prediction Error ({currency_symbol})"}
    )

    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=CORAL_ACCENT,
        annotation_text="Zero Error",
        annotation_position="top left"
    )

    fig = apply_common_layout(fig, f"Residual Distribution (Prediction Errors) [Mean: {currency_symbol}{mean_err:,.0f}, Std: {currency_symbol}{std_err:,.0f}]")
    return fig


# ==============================================================================
# FEATURE COEFFICIENT ANALYSIS
# ==============================================================================

def plot_coefficients(
    coef_df: pd.DataFrame,
    top_n: int = 20,
    currency_symbol: str = "$"
) -> go.Figure:
    """
    Horizontal bar chart of learned Multiple Linear Regression coefficients.
    Color codes positive impacts (teal) and negative impacts (coral/amber).
    """
    # Select top coefficients by absolute magnitude
    plot_data = coef_df.head(top_n).copy()
    plot_data = plot_data.sort_values(by="Coefficient", ascending=True)

    colors = [TEAL_ACCENT if c >= 0 else CORAL_ACCENT for c in plot_data["Coefficient"]]

    fig = go.Figure(go.Bar(
        y=plot_data["Feature"],
        x=plot_data["Coefficient"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{currency_symbol}{c:+,.0f}" for c in plot_data["Coefficient"]],
        textposition="outside"
    ))

    fig.add_vline(x=0, line_dash="solid", line_color=NAVY_PRIMARY, line_width=1.5)

    fig.update_layout(
        xaxis_title=f"Coefficient Value ({currency_symbol} per unit change, ceteris paribus)",
        yaxis_title="Feature",
        margin=dict(l=140, r=40, t=60, b=40)
    )

    fig = apply_common_layout(fig, f"Learned MLR Coefficients (Top {top_n} by Magnitude)", height=560)
    return fig
