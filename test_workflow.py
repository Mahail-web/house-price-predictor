"""
End-to-End Integration and Stress Testing for House Price Predictor
"""

import sys
import pandas as pd
import numpy as np

from data_loader import load_dataset, inspect_dataset, DEFAULT_DATA_PATH
from model import train_and_evaluate_model, predict_price
import visualizations as viz


def test_full_workflow():
    print("=== Step 1: Loading Dataset ===")
    df, err = load_dataset(DEFAULT_DATA_PATH)
    assert err is None, f"Dataset load failed: {err}"
    assert len(df) == 4600, f"Expected 4600 rows, got {len(df)}"
    print(f"Dataset successfully loaded with {len(df)} records and {len(df.columns)} columns.")

    print("\n=== Step 2: Dataset Quality & Audit ===")
    report = inspect_dataset(df)
    assert report["total_rows"] == 4600
    assert report["zero_target_count"] == 49
    print(f"Data audit verified: 49 zero-price rows detected.")

    print("\n=== Step 3: Training & Evaluating MLR Pipeline ===")
    results = train_and_evaluate_model(df, test_size=0.20, random_state=42)
    test_metrics = results["test_metrics"]
    train_metrics = results["train_metrics"]
    
    print(f"Test R2 Score: {test_metrics['r2']:.4f}")
    print(f"Test MAE: ${test_metrics['mae']:,.2f}")
    print(f"Test RMSE: ${test_metrics['rmse']:,.2f}")
    assert test_metrics["r2"] > 0.65, f"R2 score lower than expected: {test_metrics['r2']}"
    assert test_metrics["mae"] > 0, "MAE should be positive"
    assert test_metrics["rmse"] > 0, "RMSE should be positive"

    print("\n=== Step 4: Testing Multiple Property Inferences ===")
    test_properties = [
        {
            "name": "Standard Family Home in Seattle",
            "data": {
                "bedrooms": 3.0, "bathrooms": 2.0, "sqft_living": 1900, "sqft_lot": 5500,
                "floors": 1.5, "waterfront": 0, "view": 0, "condition": 3,
                "sqft_basement": 0, "yr_built": 1975, "yr_renovated": 0, "city": "Seattle"
            }
        },
        {
            "name": "Luxury Waterfront Estate in Medina",
            "data": {
                "bedrooms": 5.0, "bathrooms": 4.5, "sqft_living": 5400, "sqft_lot": 22000,
                "floors": 2.0, "waterfront": 1, "view": 4, "condition": 5,
                "sqft_basement": 1200, "yr_built": 2002, "yr_renovated": 2012, "city": "Medina"
            }
        },
        {
            "name": "Affordable Starter Home in Auburn",
            "data": {
                "bedrooms": 2.0, "bathrooms": 1.0, "sqft_living": 1100, "sqft_lot": 4800,
                "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3,
                "sqft_basement": 0, "yr_built": 1955, "yr_renovated": 0, "city": "Auburn"
            }
        }
    ]

    for prop in test_properties:
        pred_val = predict_price(results["pipeline"], prop["data"])
        assert np.isfinite(pred_val), f"Prediction for {prop['name']} is not finite: {pred_val}"
        print(f"  {prop['name']}: Estimated Price = ${pred_val:,.2f}")

    print("\n=== Step 5: Testing All Visualizations ===")
    fig1 = viz.plot_price_distribution(df)
    assert fig1 is not None
    fig2 = viz.plot_correlation_heatmap(df)
    assert fig2 is not None
    fig3 = viz.plot_living_area_vs_price(df)
    assert fig3 is not None
    fig4 = viz.plot_price_by_city(df)
    assert fig4 is not None
    fig5 = viz.plot_price_by_waterfront_and_view(df)
    assert fig5 is not None
    fig6 = viz.plot_price_by_bedrooms(df)
    assert fig6 is not None
    fig7 = viz.plot_actual_vs_predicted(results["y_test"], results["y_test_pred"])
    assert fig7 is not None
    fig8 = viz.plot_residuals_vs_predicted(results["y_test_pred"], results["residuals"])
    assert fig8 is not None
    fig9 = viz.plot_residual_distribution(results["residuals"])
    assert fig9 is not None
    fig10 = viz.plot_coefficients(results["coefficients_df"])
    assert fig10 is not None
    print("All 10 interactive Plotly chart functions executed successfully without errors.")

    print("\n=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_full_workflow()
