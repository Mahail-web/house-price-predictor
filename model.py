"""
Machine Learning Model module for Multiple Linear Regression.
Implements data cleaning, feature selection, pipeline construction,
training, evaluation, coefficient extraction, and interactive prediction.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Default feature specifications based on the verified USA Housing (King County) dataset
DEFAULT_TARGET_COL = "price"

DEFAULT_NUMERICAL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
]

DEFAULT_CATEGORICAL_FEATURES = ["city"]

EXCLUDED_FEATURES_EXPLANATION = {
    "date": "Transaction timestamp; represents sale timing rather than intrinsic physical or location property characteristics.",
    "street": "High-cardinality exact address (4,525 unique values); would lead to severe overfitting without generalizable predictive value.",
    "country": "Single constant value ('USA') with zero statistical variance across all records.",
    "statezip": "77 postal zip codes; strongly redundant with the 'city' feature and substantially increases one-hot dimensionality without significant R² gain.",
    "sqft_above": "Collinear with 'sqft_living' and 'sqft_basement' (sqft_living = sqft_above + sqft_basement exactly). Omitted to maintain a non-singular design matrix in OLS regression."
}


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Clean raw dataset and isolate target and predictors.
    Filters out invalid non-positive prices and extracts working copies.
    """
    if numerical_features is None:
        numerical_features = [f for f in DEFAULT_NUMERICAL_FEATURES if f in df.columns]
    if categorical_features is None:
        categorical_features = [f for f in DEFAULT_CATEGORICAL_FEATURES if f in df.columns]

    # Create isolated working copy to keep raw data pristine
    clean_df = df.copy()

    # Handle duplicate rows
    initial_len = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    duplicates_removed = initial_len - len(clean_df)

    # Filter invalid prices (real estate transactions with price <= 0 represent gifts, errors, or unrecorded amounts)
    raw_target_records = len(clean_df)
    clean_df = clean_df[clean_df[target_col] > 0].copy()
    invalid_prices_removed = raw_target_records - len(clean_df)

    # Ensure required features exist
    available_num = [f for f in numerical_features if f in clean_df.columns]
    available_cat = [f for f in categorical_features if f in clean_df.columns]
    selected_features = available_num + available_cat

    X = clean_df[selected_features].copy()
    y = clean_df[target_col].copy()

    metadata = {
        "initial_rows": initial_len,
        "clean_rows": len(clean_df),
        "duplicates_removed": duplicates_removed,
        "invalid_prices_removed": invalid_prices_removed,
        "numerical_features": available_num,
        "categorical_features": available_cat,
        "selected_features": selected_features,
        "target_col": target_col,
    }
    return X, y, metadata


def build_mlr_pipeline(
    numerical_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    """
    Construct a leak-free Scikit-learn Pipeline with ColumnTransformer.
    Uses median imputation for numeric features and most-frequent imputation + OneHotEncoder for categoricals.
    Categorical encoder drops the first category to avoid the dummy variable trap in OLS.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numerical_features:
        transformers.append(("num", num_transformer, numerical_features))
    if categorical_features:
        transformers.append(("cat", cat_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    return pipeline


def train_and_evaluate_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = DEFAULT_TARGET_COL,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Execute the complete MLR workflow:
    1. Data preparation & cleaning
    2. 80:20 Train-Test split
    3. Pipeline training (fit on train only)
    4. Test set evaluation (R2, MAE, RMSE)
    5. Coefficient extraction with named features and reference categories
    """
    X, y, prep_meta = prepare_training_data(
        df,
        target_col=target_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    if len(X) < 50:
        raise ValueError(f"Insufficient data samples ({len(X)}) to train a valid Multiple Linear Regression model.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build and fit pipeline
    num_feats = prep_meta["numerical_features"]
    cat_feats = prep_meta["categorical_features"]
    pipeline = build_mlr_pipeline(num_feats, cat_feats)

    pipeline.fit(X_train, y_train)

    # Predictions on train and test
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics on test set
    test_r2 = float(r2_score(y_test, y_test_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # Metrics on training set (for overfitting check)
    train_r2 = float(r2_score(y_train, y_train_pred))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))

    # Extract coefficients and feature names
    regressor: LinearRegression = pipeline.named_steps["regressor"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    feature_names = []
    # 1. Numerical names
    if num_feats:
        feature_names.extend(num_feats)

    # 2. Categorical one-hot names
    reference_categories = {}
    if cat_feats:
        cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        onehot_names = cat_encoder.get_feature_names_out(cat_feats).tolist()
        feature_names.extend(onehot_names)
        
        # Identify dropped reference categories
        for i, col in enumerate(cat_feats):
            if cat_encoder.drop_idx_ is not None and len(cat_encoder.drop_idx_) > i:
                drop_idx = cat_encoder.drop_idx_[i]
                ref_cat = cat_encoder.categories_[i][drop_idx]
                reference_categories[col] = ref_cat

    coefficients = regressor.coef_
    intercept = float(regressor.intercept_)

    # Build coefficient DataFrame
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Impact": np.abs(coefficients)
    }).sort_values(by="Absolute_Impact", ascending=False).reset_index(drop=True)

    # Residuals on test set
    residuals = y_test - y_test_pred

    results = {
        "pipeline": pipeline,
        "prep_metadata": prep_meta,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "residuals": residuals,
        "test_metrics": {
            "r2": test_r2,
            "mae": test_mae,
            "rmse": test_rmse
        },
        "train_metrics": {
            "r2": train_r2,
            "mae": train_mae,
            "rmse": train_rmse
        },
        "intercept": intercept,
        "coefficients_df": coef_df,
        "reference_categories": reference_categories,
        "test_size": test_size,
        "random_state": random_state
    }
    return results


def predict_price(
    pipeline: Pipeline,
    property_data: Dict[str, Any]
) -> float:
    """
    Generate a price prediction for a single property using the fitted pipeline.
    Validates inputs and matches expected schema.
    """
    df_input = pd.DataFrame([property_data])
    predicted_val = pipeline.predict(df_input)[0]
    return float(predicted_val)
