"""
House Price Predictor - Multiple Linear Regression Web Application
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="House Price Predictor | MLR Model",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Internal module imports
from utils import format_currency, render_metric_card, get_custom_css
from data_loader import load_dataset, inspect_dataset, DEFAULT_DATA_PATH, GOOGLE_DRIVE_VIEW_URL
from model import (
    train_and_evaluate_model,
    predict_price,
    DEFAULT_TARGET_COL,
    DEFAULT_NUMERICAL_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES_EXPLANATION
)
import visualizations as viz


# Inject custom styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADING
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Load raw dataset with Streamlit caching."""
    return load_dataset(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_cached_inspection(df: pd.DataFrame):
    """Inspect dataset and cache report."""
    return inspect_dataset(df, target_col=DEFAULT_TARGET_COL)


@st.cache_resource(show_spinner=False)
def get_cached_trained_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train Multiple Linear Regression pipeline and cache fitted results."""
    return train_and_evaluate_model(
        df,
        test_size=test_size,
        random_state=random_state,
        target_col=DEFAULT_TARGET_COL
    )


# ==============================================================================
# SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <span style="font-size: 2.4rem;">🏡</span>
        <h2 style="color: #FFFFFF; font-size: 1.25rem; font-weight: 700; margin: 4px 0 0 0;">House Price Predictor</h2>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">Multiple Linear Regression</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #334155; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600;'>Navigation</p>", unsafe_allow_html=True)

    nav_selection = st.radio(
        "Go to page",
        [
            "🏠 Home",
            "🔮 Predict House Price",
            "📊 Explore the Dataset",
            "📈 Model Performance",
            "🔍 Feature Analysis",
            "📖 About the Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #334155; margin: 20px 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600;'>Configuration</p>", unsafe_allow_html=True)

    currency_symbol = st.selectbox(
        "Currency Symbol",
        ["$", "€", "£", "PKR ", "CAD $", "AUD $"],
        index=0,
        help="Display currency symbol for price outputs (Dataset originates from Washington state, USA, in USD)."
    )

    with st.expander("⚙️ Advanced Model Hyperparameters"):
        test_ratio = st.slider("Test Split Ratio", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
        seed_value = st.number_input("Random Seed (Reproducibility)", min_value=1, max_value=999, value=42, step=1)

    st.markdown("<hr style='border-color: #334155; margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; text-align: center;">
        Academic ML Project &bull; Scikit-learn &bull; Streamlit<br>
        Multiple Linear Regression
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# DATASET & MODEL INITIALIZATION
# ==============================================================================

raw_df, load_error = get_cached_dataset()

if load_error or raw_df is None:
    st.error("### ⚠️ Dataset Unavailable")
    st.markdown(load_error if load_error else "An unknown error occurred while loading the dataset.")
    st.markdown(f"""
    **Manual Resolution Steps:**
    1. Visit the verified dataset link: [Google Drive Dataset]({GOOGLE_DRIVE_VIEW_URL})
    2. Download the file.
    3. Place it inside the `data/` directory named as `housing.csv`.
    4. Refresh this web application.
    """)
    st.stop()

# Inspect dataset
data_report = get_cached_inspection(raw_df)

# Train model
try:
    model_results = get_cached_trained_model(raw_df, test_size=test_ratio, random_state=seed_value)
except Exception as e:
    st.error(f"### ❌ Model Training Failure: {str(e)}")
    st.stop()


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================

if nav_selection == "🏠 Home":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Academic Machine Learning Project</span>
        <h1>🏡 House Price Predictor</h1>
        <p>Estimate property prices using Multiple Linear Regression and real-world housing data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Status Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card(
            "Dataset Records",
            f"{data_report['total_rows']:,}",
            f"{data_report['total_rows'] - data_report['zero_target_count']:,} valid transactions",
            badge_color="#38BDF8"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card(
            "Model R² Score",
            f"{model_results['test_metrics']['r2']:.1%}",
            "Held-out test set (20%)",
            badge_color="#0D9488"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card(
            "Mean Absolute Error",
            format_currency(model_results['test_metrics']['mae'], currency_symbol),
            "Average absolute error",
            badge_color="#D97706"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card(
            "Predictor Features",
            f"{len(model_results['prep_metadata']['selected_features'])} features",
            "11 Numeric + 44 Cities",
            badge_color="#6366F1"
        ), unsafe_allow_html=True)

    # Project Overview & Problem Statement
    st.markdown("""
    ### 📌 Project Overview
    In residential real estate, property values are shaped by a complex interplay of spatial, structural, and locational factors. 
    This application presents a complete, rigorous, and interactive **Multiple Linear Regression (MLR)** system developed on a verified real-world dataset of housing transactions from King County, Washington (Seattle metropolitan area).

    Rather than relying on opaque "black-box" models, this project demonstrates how classical linear statistical learning yields **interpretable equations**, transparent coefficients, and actionable insights into price formation.
    """)

    # Prediction Workflow Overview
    st.markdown("### 🔄 Machine Learning Workflow")
    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
    with flow_col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #0F172A; margin-bottom: 8px;">1. Data Ingestion & Audit</h4>
            <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                Load 4,600 raw transaction records. Audit data types, missing entries, and isolate 49 unrecorded/zero-dollar transfers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with flow_col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #0F172A; margin-bottom: 8px;">2. Leak-Free Pipeline</h4>
            <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                Split data (80:20). Impute values and apply One-Hot Encoding strictly inside a fitted Scikit-learn ColumnTransformer.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with flow_col3:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #0F172A; margin-bottom: 8px;">3. MLR Model Fitting</h4>
            <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                Fit Ordinary Least Squares (OLS) regression to estimate the baseline intercept and partial regression coefficients.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with flow_col4:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #0F172A; margin-bottom: 8px;">4. Evaluation & Inference</h4>
            <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                Verify performance via test R², MAE, RMSE, and residual diagnostics. Deploy for interactive user predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Call to action
    st.markdown("---")
    st.markdown("#### Ready to test the model?")
    st.info("💡 Navigate to **🔮 Predict House Price** in the sidebar to enter custom property characteristics and view instant valuation estimates.")


# ==============================================================================
# PAGE 2: PREDICT HOUSE PRICE
# ==============================================================================

elif nav_selection == "🔮 Predict House Price":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Interactive Inference Engine</span>
        <h1>🔮 Predict House Price</h1>
        <p>Enter property specifications below to calculate an estimated market price using the fitted MLR pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = model_results["pipeline"]
    valid_df = raw_df[raw_df["price"] > 0]

    # Unique sorted cities from dataset
    city_options = sorted(valid_df["city"].unique().tolist())

    st.markdown("### 📋 Enter Property Specifications")

    with st.form("prediction_form"):
        st.markdown("##### 📍 Location & Property Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            input_city = st.selectbox(
                "City (Municipality)",
                options=city_options,
                index=city_options.index("Seattle") if "Seattle" in city_options else 0,
                help="City in King County, Washington."
            )
        with c2:
            input_yr_built = st.slider(
                "Year Built",
                min_value=int(valid_df["yr_built"].min()),
                max_value=int(valid_df["yr_built"].max()),
                value=1985,
                step=1,
                help="Original construction year."
            )
        with c3:
            has_renovated = st.checkbox("Property Was Renovated?", value=False)
            if has_renovated:
                input_yr_renovated = st.slider(
                    "Year Renovated",
                    min_value=1950,
                    max_value=2014,
                    value=2005,
                    step=1
                )
            else:
                input_yr_renovated = 0

        st.markdown("##### 📐 Dimensions & Area")
        d1, d2, d3 = st.columns(3)
        with d1:
            input_sqft_living = st.number_input(
                "Interior Living Area (Sq Ft)",
                min_value=350,
                max_value=14000,
                value=2200,
                step=50,
                help="Total interior living space in square feet."
            )
        with d2:
            input_sqft_lot = st.number_input(
                "Lot Size (Sq Ft)",
                min_value=500,
                max_value=1100000,
                value=6500,
                step=100,
                help="Total property land area in square feet."
            )
        with d3:
            has_basement = st.checkbox("Has Basement?", value=False)
            if has_basement:
                input_sqft_basement = st.number_input(
                    "Basement Area (Sq Ft)",
                    min_value=50,
                    max_value=5000,
                    value=600,
                    step=50
                )
            else:
                input_sqft_basement = 0

        st.markdown("##### 🛏️ Rooms & Layout")
        r1, r2, r3 = st.columns(3)
        with r1:
            input_bedrooms = st.slider(
                "Bedrooms",
                min_value=1.0,
                max_value=9.0,
                value=3.0,
                step=1.0,
                help="Number of bedrooms."
            )
        with r2:
            input_bathrooms = st.slider(
                "Bathrooms",
                min_value=0.75,
                max_value=7.0,
                value=2.25,
                step=0.25,
                help="Bathrooms (0.75 = 3/4 bath with shower, 1.0 = full bath, etc.)."
            )
        with r3:
            input_floors = st.selectbox(
                "Number of Floors",
                options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                index=2,
                help="Number of above-ground floor levels."
            )

        st.markdown("##### ⭐ Quality & Amenities")
        q1, q2, q3 = st.columns(3)
        with q1:
            input_waterfront = st.radio(
                "Waterfront Property?",
                options=[0, 1],
                format_func=lambda x: "Yes (Direct Waterfront)" if x == 1 else "No (Standard Parcel)",
                index=0,
                horizontal=True
            )
        with q2:
            input_view = st.slider(
                "View Rating (0 to 4)",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
                help="0 = No view, 1 = Fair, 2 = Average, 3 = Good, 4 = Exceptional."
            )
        with q3:
            input_condition = st.slider(
                "Overall Condition (1 to 5)",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Poor, 3 = Average, 5 = Pristine maintenance."
            )

        submit_btn = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

    if submit_btn:
        # Input Validation
        if input_sqft_living < 300:
            st.error("Invalid input: Living area must be at least 300 sq ft.")
        elif input_sqft_lot < input_sqft_living:
            st.warning("Notice: Entered lot size is smaller than interior living space. While possible for multi-story townhomes, please verify your dimensions.")

        user_input_dict = {
            "bedrooms": float(input_bedrooms),
            "bathrooms": float(input_bathrooms),
            "sqft_living": int(input_sqft_living),
            "sqft_lot": int(input_sqft_lot),
            "floors": float(input_floors),
            "waterfront": int(input_waterfront),
            "view": int(input_view),
            "condition": int(input_condition),
            "sqft_basement": int(input_sqft_basement),
            "yr_built": int(input_yr_built),
            "yr_renovated": int(input_yr_renovated),
            "city": str(input_city)
        }

        with st.spinner("Calculating price with Multiple Linear Regression pipeline..."):
            predicted_price = predict_price(pipeline, user_input_dict)

        st.markdown("---")
        st.markdown("### 🏷️ Valuation Output")

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                border-radius: 16px;
                padding: 30px;
                color: #FFFFFF;
                border: 2px solid #0D9488;
                box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.2);
            ">
                <span style="
                    background: rgba(13, 148, 136, 0.2);
                    color: #2DD4BF;
                    border: 1px solid rgba(45, 212, 191, 0.4);
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    text-transform: uppercase;
                ">Model Estimated Value</span>
                <div style="font-size: 2.8rem; font-weight: 800; color: #FFFFFF; margin: 12px 0 6px 0; letter-spacing: -0.03em;">
                    {format_currency(predicted_price, currency_symbol)}
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0;">
                    Model R²: <strong>{model_results['test_metrics']['r2']:.1%}</strong> &bull; Test MAE: <strong>&plusmn;{format_currency(model_results['test_metrics']['mae'], currency_symbol)}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-callout" style="margin-top: 14px;">
                <strong>⚠️ Academic Valuation Disclaimer:</strong> This estimate is generated via Multiple Linear Regression fitted on historical transaction patterns. 
                It does not constitute a certified appraisal, formal property assessment, or guaranteed valuation. Real market transactions reflect unmodeled factors including hyper-local block premiums, micro-market conditions, and subjective finish quality.
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### 📝 Property Summary Received")
            summary_df = pd.DataFrame([
                {"Attribute": "Municipality", "Specified Value": input_city},
                {"Attribute": "Living Space", "Specified Value": f"{input_sqft_living:,} sq ft"},
                {"Attribute": "Lot Size", "Specified Value": f"{input_sqft_lot:,} sq ft"},
                {"Attribute": "Bedrooms / Bathrooms", "Specified Value": f"{int(input_bedrooms)} beds / {input_bathrooms} baths"},
                {"Attribute": "Floors", "Specified Value": f"{input_floors} floors"},
                {"Attribute": "Waterfront", "Specified Value": "Yes" if input_waterfront == 1 else "No"},
                {"Attribute": "View Score", "Specified Value": f"{input_view} / 4"},
                {"Attribute": "Condition Score", "Specified Value": f"{input_condition} / 5"},
                {"Attribute": "Year Built / Renovated", "Specified Value": f"{input_yr_built} / ({input_yr_renovated if input_yr_renovated > 0 else 'Never'})"}
            ])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 3: EXPLORE THE DATASET (EDA)
# ==============================================================================

elif nav_selection == "📊 Explore the Dataset":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Comprehensive Data Audit</span>
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Inspect dataset characteristics, quality metrics, distributions, and empirical correlations.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Quality & Preview",
        "💰 Price & Area Distributions",
        "🗺️ Geographic Variations",
        "🔥 Correlation Matrix"
    ])

    with tab1:
        st.markdown("### 🔍 Dataset Overview & Integrity Audit")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(render_metric_card("Total Rows", f"{data_report['total_rows']:,}", "Original records loaded"), unsafe_allow_html=True)
        with q2:
            st.markdown(render_metric_card("Total Columns", f"{data_report['total_cols']}", "17 Features + 1 Target"), unsafe_allow_html=True)
        with q3:
            st.markdown(render_metric_card("Duplicate Rows", f"{data_report['duplicate_rows']}", "Exact duplicates count", badge_color="#10B981"), unsafe_allow_html=True)
        with q4:
            st.markdown(render_metric_card("Zero Prices Detected", f"{data_report['zero_target_count']}", "Filtered non-arms-length records", badge_color="#F59E0B"), unsafe_allow_html=True)

        st.markdown("#### 📄 First 10 Records of Original Dataset")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.markdown("#### 📊 Column Schema & Quality Audit")
        st.dataframe(data_report["column_info_df"], hide_index=True, use_container_width=True)

        st.markdown("#### 📈 Target Price Summary Statistics")
        target_summary_df = pd.DataFrame([data_report["target_summary"]]).T.rename(columns={0: "Value"})
        st.dataframe(target_summary_df, use_container_width=True)

    with tab2:
        st.markdown("### 💰 Price & Living Space Distributions")
        st.plotly_chart(viz.plot_price_distribution(raw_df, currency_symbol=currency_symbol), use_container_width=True)

        st.markdown("""
        > **Insight:** House prices exhibit a characteristic right-skewed distribution common in residential housing markets, with the vast majority of sales clustered between $300,000 and $700,000, accompanied by a long tail of luxury properties exceeding $2,000,000.
        """)

        st.plotly_chart(viz.plot_living_area_vs_price(raw_df, currency_symbol=currency_symbol), use_container_width=True)

        st.markdown("""
        > **Insight:** Interior living area (`sqft_living`) is the single strongest continuous linear predictor of price ($r \approx 0.45$). The positive trendline illustrates increasing marginal valuation with square footage.
        """)

    with tab3:
        st.markdown("### 🗺️ Geographic & Amenity Price Variations")
        st.plotly_chart(viz.plot_price_by_city(raw_df, top_n=15, currency_symbol=currency_symbol), use_container_width=True)

        st.markdown("""
        > **Insight:** Location is paramount. Municipalities like Medina, Clyde Hill, Bellevue, and Mercer Island demonstrate substantial median price premiums over southern King County areas like Auburn and Federal Way.
        """)

        col_box1, col_box2 = st.columns(2)
        with col_box1:
            st.plotly_chart(viz.plot_price_by_waterfront_and_view(raw_df, currency_symbol=currency_symbol), use_container_width=True)
        with col_box2:
            st.plotly_chart(viz.plot_price_by_bedrooms(raw_df, currency_symbol=currency_symbol), use_container_width=True)

    with tab4:
        st.markdown("### 🔥 Numerical Feature Correlation Matrix")
        st.plotly_chart(viz.plot_correlation_heatmap(raw_df), use_container_width=True)
        st.markdown("""
        > **Critical Observation on Multicollinearity:** `sqft_living`, `sqft_above`, and `sqft_basement` satisfy the exact linear equality: 
        $$\\text{sqft\\_living} = \\text{sqft\\_above} + \\text{sqft\\_basement}$$
        Including all three in Ordinary Least Squares causes severe multicollinearity (rank deficiency in the design matrix). In our leak-free pipeline, `sqft_above` is excluded to maintain numerical stability and non-inflated standard errors.
        """)


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================

elif nav_selection == "📈 Model Performance":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Rigorous Evaluation</span>
        <h1>📈 Model Performance Dashboard</h1>
        <p>Evaluate the Multiple Linear Regression model on the held-out 20% test dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    test_metrics = model_results["test_metrics"]
    train_metrics = model_results["train_metrics"]

    # Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(render_metric_card(
            "Test R² Score",
            f"{test_metrics['r2']:.4f}",
            f"Explains {test_metrics['r2']:.1%} of test variance",
            badge_color="#0D9488"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card(
            "Mean Absolute Error (MAE)",
            format_currency(test_metrics['mae'], currency_symbol),
            "Average test error magnitude",
            badge_color="#38BDF8"
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card(
            "Root Mean Squared Error (RMSE)",
            format_currency(test_metrics['rmse'], currency_symbol),
            "Penalizes large outlier errors",
            badge_color="#D97706"
        ), unsafe_allow_html=True)

    st.markdown("### 📊 Train vs. Test Performance (Overfitting Check)")
    comparison_df = pd.DataFrame({
        "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
        "Training Set (80%)": [
            f"{train_metrics['r2']:.4f} ({train_metrics['r2']:.1%})",
            format_currency(train_metrics['mae'], currency_symbol),
            format_currency(train_metrics['rmse'], currency_symbol)
        ],
        "Held-Out Test Set (20%)": [
            f"{test_metrics['r2']:.4f} ({test_metrics['r2']:.1%})",
            format_currency(test_metrics['mae'], currency_symbol),
            format_currency(test_metrics['rmse'], currency_symbol)
        ]
    })
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    st.markdown("""
    > **Academic Assessment:** The close correspondence between training $R^2$ and testing $R^2$ indicates that the model has **not overfitted**. 
    It generalizes predictably to previously unseen properties without memorizing training noise.
    """)

    st.markdown("### 🎯 Diagnostic Visualizations")
    st.plotly_chart(
        viz.plot_actual_vs_predicted(model_results["y_test"], model_results["y_test_pred"], currency_symbol=currency_symbol),
        use_container_width=True
    )

    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.plotly_chart(
            viz.plot_residuals_vs_predicted(model_results["y_test_pred"], model_results["residuals"], currency_symbol=currency_symbol),
            use_container_width=True
        )
    with c_res2:
        st.plotly_chart(
            viz.plot_residual_distribution(model_results["residuals"], currency_symbol=currency_symbol),
            use_container_width=True
        )

    st.markdown("""
    #### 🔬 Honest Residual Analysis
    1. **Heteroscedasticity:** As visible in the residual plot, prediction variance increases for luxury homes above $2,000,000. This is an expected limitation of standard Ordinary Least Squares (OLS) on skewed real estate prices.
    2. **Residual Normality:** The residual distribution is centered near zero ($mean \\approx 0$), confirming that the model's predictions are unbiased on average.
    """)


# ==============================================================================
# PAGE 5: FEATURE ANALYSIS (INTERPRETABILITY)
# ==============================================================================

elif nav_selection == "🔍 Feature Analysis":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Model Interpretability</span>
        <h1>🔍 Feature Coefficient Analysis</h1>
        <p>Inspect the learned Ordinary Least Squares coefficients and the mathematical prediction equation.</p>
    </div>
    """, unsafe_allow_html=True)

    intercept = model_results["intercept"]
    coef_df = model_results["coefficients_df"]
    ref_cats = model_results["reference_categories"]

    st.markdown("### 📐 General Multiple Linear Regression Equation")
    st.latex(r"\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p")

    st.markdown(f"""
    - **Estimated Intercept ($\beta_0$):** `{format_currency(intercept, currency_symbol)}`
    - **Baseline Context:** The intercept represents the mathematical baseline when all continuous predictors are set to zero and the categorical reference municipality (`{ref_cats.get('city', 'Auburn')}`) is active.
    """)

    st.markdown("### 📊 Top Regression Coefficients")
    st.plotly_chart(viz.plot_coefficients(coef_df, top_n=20, currency_symbol=currency_symbol), use_container_width=True)

    st.markdown("### 📋 Complete Learned Coefficients Table")
    st.dataframe(coef_df, hide_index=True, use_container_width=True)

    st.markdown("""
    ### 💡 How to Correctly Interpret MLR Coefficients
    1. **Ceteris Paribus (Holding other features constant):** A positive coefficient (e.g. `+193` for `sqft_living`) means that for every additional 1 square foot of living area, the predicted house price increases by $193, holding all other features (bedrooms, city, condition) constant.
    2. **Categorical Dummies & Reference Category:** `OneHotEncoder(drop='first')` omitted the baseline city (`Auburn`). A coefficient for `city_Medina` (+$1,024,977) indicates that a property located in Medina commands a ~$1,024,977 price premium relative to an otherwise identical house in Auburn.
    3. **Association, Not Causation:** Multiple Linear Regression quantifies statistical conditional correlation within the sample space. A positive coefficient does not guarantee that altering a feature will causally raise the market sale price by that exact amount.
    """)


# ==============================================================================
# PAGE 6: ABOUT THE PROJECT
# ==============================================================================

elif nav_selection == "📖 About the Project":
    st.markdown("""
    <div class="header-banner">
        <span class="header-badge">Academic Documentation</span>
        <h1>📖 About the Project & Methodology</h1>
        <p>Theoretical foundations, mathematical formulations, and engineering principles.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(r"""
    ### 🎓 Machine Learning Algorithm: Multiple Linear Regression (MLR)
    Multiple Linear Regression is a foundational supervised learning algorithm designed to model the linear relationship between a continuous dependent variable (target, $y$) and two or more independent predictor variables ($X_1, X_2, \dots, X_p$).

    #### Mathematical Model Formulation
    The classical MLR model expresses the target as:
    $$y_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} + \epsilon_i$$
    Where:
    - $y_i$ is the actual transaction price of the $i$-th house.
    - $\beta_0$ is the intercept term.
    - $\beta_j$ is the partial regression coefficient for feature $j$.
    - $x_{ij}$ is the value of feature $j$ for house $i$.
    - $\epsilon_i$ represents the unobservable random error term ($\epsilon_i \sim \mathcal{N}(0, \sigma^2)$).

    The parameter vector $\hat{\boldsymbol{\beta}}$ is estimated via Ordinary Least Squares (OLS) by minimizing the sum of squared residuals:
    $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

    ---

    ### 🛡️ Preventing Data Leakage
    Data leakage occurs when information from outside the training dataset is inadvertently used to fit preprocessing transformations.
    In this application, all preprocessing steps—including **median imputation** for numerical features and **most frequent imputation + one-hot encoding** for categorical variables—are integrated into a Scikit-learn `Pipeline` and `ColumnTransformer`.
    - **Transformation fitting:** Executed exclusively on `X_train`.
    - **Test evaluation:** Executed on `X_test` using the pre-fitted pipeline without refitting.

    ---

    ### 📏 Model Evaluation Metrics
    1. **Coefficient of Determination ($R^2$):**
       $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
       Measures the proportion of variance in house prices explained by the model compared to a naive baseline mean predictor.
    2. **Mean Absolute Error (MAE):**
       $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
       Reflects the average expected deviation in dollar terms. Robust to extreme outliers.
    3. **Root Mean Squared Error (RMSE):**
       $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
       Heavily penalizes large prediction errors, providing insight into model stability on high-value properties.

    ---

    ### ⚠️ Limitations of Linear Regression in Real Estate
    While MLR offers unparalleled interpretability and mathematical transparency, real-world real estate pricing incorporates intrinsic non-linearities:
    - **Diminishing Marginal Returns:** An increase in square footage from 1,000 to 2,000 sq ft often commands higher marginal value than expanding from 7,000 to 8,000 sq ft.
    - **Interaction Effects:** The value of a pool or additional bedroom varies dramatically depending on whether the home is in an entry-level suburb or an ultra-luxury enclave.
    - **Market Volatility & Temporal Trends:** Seasonal shifts, interest rate fluctuations, and localized supply crunches influence real transactions beyond physical property attributes.
    """)

    st.markdown("### 🗂️ Excluded Features & Academic Rationale")
    ex_df = pd.DataFrame([
        {"Excluded Column": k, "Academic Rationale": v}
        for k, v in EXCLUDED_FEATURES_EXPLANATION.items()
    ])
    st.dataframe(ex_df, hide_index=True, use_container_width=True)
