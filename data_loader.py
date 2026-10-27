"""
Dataset loading, validation, and quality assessment module.
Handles loading from local disk or downloading from the designated Google Drive source.
Generates comprehensive data-quality reports while keeping raw data immutable.
"""

import os
import re
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import requests


GOOGLE_DRIVE_FILE_ID = "1kicLZ_mUT1f5O1STqb-azATZtkmjsxov"
DEFAULT_DATA_PATH = os.path.join("data", "housing.csv")
GOOGLE_DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view?usp=sharing"


def download_from_google_drive(file_id: str = GOOGLE_DRIVE_FILE_ID, destination: str = DEFAULT_DATA_PATH) -> bool:
    """
    Download dataset directly from Google Drive if local file does not exist.
    Handles Google Drive virus scan confirmation redirects for large files.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = session.get(url, headers=headers, stream=True, timeout=15)
        content = response.content

        # Check if Google Drive returned a confirmation warning page
        text = content.decode("utf-8", errors="ignore")
        if "<html" in text.lower():
            confirm_match = re.search(r'href="(/uc\?export=download[^"]+)"', text)
            if confirm_match:
                confirm_url = "https://drive.google.com" + confirm_match.group(1).replace("&amp;", "&")
                r2 = session.get(confirm_url, headers=headers, stream=True, timeout=20)
                content = r2.content
            else:
                token_match = re.search(r'confirm=([^"&]+)', text)
                if token_match:
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={token_match.group(1)}&id={file_id}"
                    r2 = session.get(confirm_url, headers=headers, stream=True, timeout=20)
                    content = r2.content
                else:
                    return False

        # Validate that the downloaded content is indeed CSV
        if len(content) > 1000 and (b"," in content[:500] or b"\n" in content[:500]):
            with open(destination, "wb") as f:
                f.write(content)
            return True
        return False
    except Exception:
        return False


def load_dataset(file_path: str = DEFAULT_DATA_PATH) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load raw dataset from disk, attempting download if absent.
    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    if not os.path.exists(file_path):
        success = download_from_google_drive(GOOGLE_DRIVE_FILE_ID, file_path)
        if not success or not os.path.exists(file_path):
            error_msg = (
                f"The dataset file could not be located at `{file_path}` and automated download from Google Drive "
                f"did not complete. Please download the dataset manually from:\n"
                f"[{GOOGLE_DRIVE_VIEW_URL}]({GOOGLE_DRIVE_VIEW_URL})\n\n"
                f"Save the file as `{file_path}` in your project directory and refresh this page."
            )
            return None, error_msg

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None, f"Dataset loaded from `{file_path}` is empty (0 rows)."
        return df, None
    except Exception as e:
        return None, f"Failed to read dataset from `{file_path}`: {str(e)}"


def inspect_dataset(df: pd.DataFrame, target_col: str = "price") -> Dict[str, Any]:
    """
    Perform a complete dataset inspection and generate a data-quality audit report.
    Identifies column types, null values, duplicates, and invalid target entries.
    """
    total_rows, total_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())

    # Column type categorization
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target inspection
    target_exists = target_col in df.columns
    zero_target_count = 0
    negative_target_count = 0
    target_summary = {}

    if target_exists and pd.api.types.is_numeric_dtype(df[target_col]):
        zero_target_count = int((df[target_col] == 0).sum())
        negative_target_count = int((df[target_col] < 0).sum())
        desc = df[target_col].describe()
        target_summary = {
            "count": int(desc["count"]),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "25%": float(desc["25%"]),
            "50%": float(desc["50%"]),
            "75%": float(desc["75%"]),
            "max": float(desc["max"]),
        }

    # Summary table of columns
    column_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        dtype_str = str(df[col].dtype)
        col_type = "Numeric" if col in numeric_cols else "Categorical"
        sample_val = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
        column_info.append({
            "Column": col,
            "Data Type": dtype_str,
            "Type Category": col_type,
            "Missing Values": null_count,
            "Missing (%)": f"{(null_count / total_rows) * 100:.1f}%",
            "Unique Count": unique_count,
            "Sample Value": sample_val
        })

    report = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "total_missing": total_missing,
        "missing_per_col": missing_series[missing_series > 0].to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "target_exists": target_exists,
        "zero_target_count": zero_target_count,
        "negative_target_count": negative_target_count,
        "target_summary": target_summary,
        "column_info_df": pd.DataFrame(column_info)
    }
    return report
