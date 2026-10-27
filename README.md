# 🏡 House Price Predictor Using Multiple Linear Regression

An academic, end-to-end Machine Learning web application developed using **Python**, **Pandas**, **NumPy**, **Scikit-learn**, and **Streamlit**. 

This application demonstrates the complete machine learning lifecycle for **Multiple Linear Regression (MLR)**: from raw data ingestion and leak-free preprocessing to model training, evaluation on held-out test data, coefficient interpretability, and interactive property price prediction.

---

## 📌 1. Project Overview

Predicting residential real estate values is a canonical continuous regression challenge. Property prices depend simultaneously on physical structural dimensions (living area, lot size, bedroom/bathroom layout), property condition, amenities (waterfront, scenic view ratings), and geographic municipality.

### Core Objectives:
- Implement a genuine **Multiple Linear Regression** model using Scikit-learn's `LinearRegression`.
- Train and evaluate exclusively on the verified **USA Housing (King County, Washington)** dataset provided in the course assignment.
- Build a leak-free pipeline utilizing `Pipeline` and `ColumnTransformer` (fitting preprocessing exclusively on training data).
- Provide an interactive, responsive Streamlit dashboard with real-time inference, exploratory data analysis (EDA), diagnostic residual plots, and coefficient interpretability.

---

## 📂 2. Dataset Description & Provenance

- **Dataset Source:** Real-world King County / Seattle metropolitan area house sales dataset.
- **Google Drive Access URL:** [https://drive.google.com/file/d/1kicLZ_mUT1f5O1STqb-azATZtkmjsxov/view?usp=sharing](https://drive.google.com/file/d/1kicLZ_mUT1f5O1STqb-azATZtkmjsxov/view?usp=sharing)
- **Local Location:** `data/housing.csv`
- **Total Records:** 4,600 transactions
- **Target Variable:** `price` (Continuous float, ranging up to $26,590,000 USD; median ~$460,943).

### Data Quality & Preprocessing Decisions:
1. **Invalid Target Handling:** 49 records in the raw dataset recorded a transaction price of `$0.0`. In real estate transaction registers, zero-dollar transfers signify non-arms-length transfers, gift deeds, inheritance, or unrecorded transaction data. These records are filtered from model training as property prices must be strictly positive ($price > 0$).
2. **Duplicate Records:** 0 exact duplicate rows found in the raw dataset.
3. **Missing Values:** No null values present in the raw file; however, the pipeline includes `SimpleImputer` (median for numeric, most frequent for categorical) to ensure zero data leakage on unseen inference inputs.

### Feature Selection & Academic Rationale:
| Feature | Role | Type | Rationale |
| :--- | :--- | :--- | :--- |
| `price` | **Target ($y$)** | Continuous Float | House sale price in USD |
| `sqft_living` | Predictor ($X_1$) | Numeric (Integer) | Interior square footage; strongest linear predictor ($r \approx 0.45$) |
| `bedrooms` | Predictor ($X_2$) | Numeric (Float) | Number of bedrooms (1 to 9) |
| `bathrooms` | Predictor ($X_3$) | Numeric (Float) | Number of bathrooms (0.75 to 7.0) |
| `sqft_lot` | Predictor ($X_4$) | Numeric (Integer) | Parcel land area in square feet |
| `floors` | Predictor ($X_5$) | Numeric (Float) | Above-ground floor levels (1.0 to 3.5) |
| `waterfront` | Predictor ($X_6$) | Binary (0 or 1) | Direct shoreline frontage |
| `view` | Predictor ($X_7$) | Ordinal (0 to 4) | Scenery / view quality rating |
| `condition` | Predictor ($X_8$) | Ordinal (1 to 5) | Property maintenance rating |
| `sqft_basement`| Predictor ($X_9$) | Numeric (Integer) | Basement square footage |
| `yr_built` | Predictor ($X_{10}$) | Numeric (Integer) | Year of initial construction (1900–2014) |
| `yr_renovated` | Predictor ($X_{11}$) | Numeric (Integer) | Year of renovation (0 if never renovated) |
| `city` | Predictor ($X_{12}$) | Categorical (44) | Municipality in King County (One-Hot Encoded) |

### Excluded Columns:
- `date`: Sale timestamp; temporal metadata rather than an intrinsic physical or spatial feature.
- `street`: 4,525 unique addresses across 4,600 rows. Including unique street strings causes severe over-parameterization without generalizability.
- `country`: Single constant value (`'USA'`); zero variance across all records.
- `statezip`: 77 postal zip codes; strongly collinear with `city` and inflates dummy dimensionality without meaningful $R^2$ gain.
- `sqft_above`: Satisfies the exact mathematical identity $\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$. Retaining all three creates severe multicollinearity (matrix rank deficiency).

---

## 🔬 3. Machine Learning Methodology

### The Multiple Linear Regression Model
The model estimates the conditional mean of the target variable $y$ given the predictor matrix $\mathbf{X}$:
$$\hat{y}_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \dots + \beta_p X_{ip}$$

The parameter vector $\hat{\boldsymbol{\beta}}$ is estimated via Ordinary Least Squares (OLS):
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Train-Test Split:
- **80% Training Set** (3,640 samples)
- **20% Held-Out Testing Set** (911 samples)
- **Random Seed:** Fixed at `42` for strict reproducibility.

### Leak-Free ColumnTransformer Pipeline:
- **Numerical Pipeline:** `SimpleImputer(strategy='median')`
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(drop='first', handle_unknown='ignore')`
- Dropping the first category (`Auburn`) avoids the "dummy variable trap" (perfect multicollinearity with the intercept $\beta_0$) and provides an interpretable reference municipality.

---

## 📊 4. Model Performance & Evaluation

Evaluated strictly on the held-out 20% test dataset:

| Metric | Test Set (Held-Out) | Training Set | Interpretation |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | **0.6924 (69.2%)** | **0.6552 (65.5%)** | The model explains ~69.2% of price variance on unseen houses. |
| **Mean Absolute Error (MAE)** | **$130,480.84** | **$124,197.68** | Average expected prediction error in price units ($ USD). |
| **Root Mean Squared Error (RMSE)** | **$213,903.09** | **$217,994.51** | Standard deviation of prediction residuals; penalizes large errors. |

### Key Diagnostic Observations:
1. **No Overfitting:** The test score closely tracks the training score ($R^2 \approx 0.69$ vs $0.66$), demonstrating healthy generalization.
2. **Residual Normality:** Prediction residuals are symmetrically distributed around zero ($mean \approx 0$).
3. **Heteroscedasticity:** Prediction variance increases for luxury homes ($>\$2,000,000$), highlighting the classical real-world limitation of unweighted OLS on skewed real estate distributions.

---

## 🔍 5. Feature Interpretability Highlights

- **Estimated Model Intercept ($\beta_0$):** `+$1,843,050` (mathematical reference baseline).
- **Living Area (`sqft_living`):** `+$193.25 / sq ft`. Each additional square foot of living area increases estimated price by ~$193, holding location and condition constant.
- **Waterfront (`waterfront`):** `+$552,020`. Direct waterfront access adds an estimated half-million dollar premium.
- **Top Location Premiums (relative to Auburn baseline):**
  - `Medina`: `+$1,024,977`
  - `Clyde Hill`: `+$705,576`
  - `Mercer Island`: `+$329,748`
  - `Bellevue`: `+$263,450`

---

## 🚀 6. Installation and Running Locally

### Prerequisites:
- Python 3.9, 3.10, 3.11, or 3.12 (or `uv`)
- Web browser

### Step 1: Clone or Navigate to Directory
```bash
cd "d:\Datacrumbs assignments\Mutiple linear regression model"
```

### Step 2: Set up a Virtual Environment
```bash
# Using standard Python venv
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 📁 7. Project Architecture

```
Mutiple linear regression model/
├── app.py                 # Streamlit multi-page web application entry point
├── model.py               # Preprocessing, MLR pipeline, training, evaluation, inference
├── data_loader.py         # Dataset loading, Google Drive download, data-quality audit
├── visualizations.py      # Interactive Plotly charts (EDA, Evaluation, Coefficients)
├── utils.py               # Currency formatting, custom CSS theme, metric card components
├── data/
│   └── housing.csv        # Downloaded King County housing dataset
├── requirements.txt       # Production dependencies
├── .gitignore             # Excluded environments and temporary caches
└── README.md              # Academic documentation and setup manual
```

---

## ⚠️ 8. Academic Limitations & Real-Estate Context

1. **Non-Causal Interpretability:** Coefficients indicate statistical partial correlations within the King County sample space; they do not establish causal relationships.
2. **Linearity Assumption:** OLS assumes linear marginal effects. In reality, additions exhibit diminishing returns (e.g. expanding from 7,000 to 8,000 sq ft adds less relative utility than expanding from 1,000 to 2,000 sq ft).
3. **Unmodeled Micro-Factors:** Property valuation in practice is influenced by unrecorded characteristics including micro-neighborhood boundaries, school attendance zones, architectural finishes, noise levels, and macroeconomic interest rate fluctuations.

---

## 👨‍💻 Academic Deliverable
Developed as a rigorous, leak-free Multiple Linear Regression demonstration for the **Datacrumbs Academic Machine Learning Assignment**.
