"""
Data preprocessing module for the Customer Churn Prediction system.

Handles loading, cleaning, type correction, outlier treatment,
and saving of the processed dataset.
"""

import logging
import os
import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict for business reporting and notebook display."""
    report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {},
    }
    return report


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Correct dtypes: TotalCharges has spaces that prevent numeric parsing."""
    df = df.copy()

    # TotalCharges stored as object due to blank strings for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Converted TotalCharges to numeric")

    # SeniorCitizen is 0/1 integer — convert to categorical for clarity
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    logger.info("Mapped SeniorCitizen 0/1 -> No/Yes")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is NaN for customers with tenure=0 (brand-new customers).
    Impute with 0 since they haven't been billed yet.
    """
    df = df.copy()
    before = df.isnull().sum().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    after = df.isnull().sum().sum()
    logger.info(f"Imputed {before - after} missing values (TotalCharges -> 0)")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Removed {removed} duplicate rows")
    return df


def standardize_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Encode target: Yes -> 1, No -> 0."""
    df = df.copy()
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    logger.info(f"Encoded target column '{target_col}': Yes->1, No->0")
    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and standardize string columns."""
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    logger.info(f"Standardized {len(str_cols)} categorical columns")
    return df


def treat_outliers(df: pd.DataFrame, columns: list, method: str = "clip") -> pd.DataFrame:
    """
    Clip outliers at 1st and 99th percentile.
    For churn data, extreme values are often genuine (very high charges), so we clip rather than remove.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=low, upper=high)
    logger.info(f"Clipped outliers in: {columns}")
    return df


def drop_id_column(df: pd.DataFrame, id_col: str = "customerID") -> tuple:
    """Separate and return the ID column, then drop from features."""
    ids = df[id_col].copy() if id_col in df.columns else None
    df = df.drop(columns=[id_col], errors="ignore")
    logger.info(f"Separated ID column '{id_col}'")
    return df, ids


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path} ({df.shape[0]} rows)")


def run_preprocessing_pipeline(config_path: str = "config.yaml") -> pd.DataFrame:
    """Full end-to-end preprocessing. Returns the cleaned DataFrame."""
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    numerical_cols = config["features"]["numerical"]

    df = load_raw_data(raw_path)

    report = inspect_data(df)
    logger.info(f"Data shape: {report['shape']}")
    logger.info(f"Duplicates: {report['duplicates']}")
    missing = {k: v for k, v in report['missing_values'].items() if v > 0}
    logger.info(f"Missing values: {missing}")
    logger.info(f"Target distribution: {report['target_distribution']}")

    df = standardize_categoricals(df)
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = treat_outliers(df, columns=numerical_cols)
    df = standardize_target(df, target_col=config["data"]["target_column"])

    save_processed_data(df, processed_path)
    return df


if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(f"\nPreprocessing complete. Shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    print(df.head(3))
