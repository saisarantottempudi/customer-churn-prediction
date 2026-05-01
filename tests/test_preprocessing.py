"""
Unit tests for the data preprocessing module.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    fix_data_types,
    handle_missing_values,
    remove_duplicates,
    standardize_target,
    standardize_categoricals,
    treat_outliers,
    inspect_data,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["A001", "A002", "A003", "A001"],  # A001 duplicated
        "gender": ["Female", " Male", "Female", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "tenure": [5, 24, 0, 5],
        "MonthlyCharges": [29.85, 56.95, 53.85, 29.85],
        "TotalCharges": ["149.25", " ", "0.0", "149.25"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Month-to-month"],
        "Churn": ["No", "Yes", "No", "No"],
    })


def test_fix_data_types_total_charges(sample_df):
    result = fix_data_types(sample_df)
    assert result["TotalCharges"].dtype in [float, np.float64]
    assert pd.isna(result["TotalCharges"].iloc[1])  # blank -> NaN


def test_fix_data_types_senior_citizen(sample_df):
    result = fix_data_types(sample_df)
    assert set(result["SeniorCitizen"].unique()).issubset({"Yes", "No"})


def test_handle_missing_values(sample_df):
    df = fix_data_types(sample_df)
    result = handle_missing_values(df)
    assert result["TotalCharges"].isnull().sum() == 0
    assert result["TotalCharges"].iloc[1] == 0.0


def test_remove_duplicates(sample_df):
    result = remove_duplicates(sample_df)
    assert len(result) == len(sample_df) - 1


def test_standardize_target(sample_df):
    result = standardize_target(sample_df, "Churn")
    assert set(result["Churn"].unique()).issubset({0, 1})
    assert result["Churn"].iloc[1] == 1  # "Yes" -> 1
    assert result["Churn"].iloc[0] == 0  # "No" -> 0


def test_standardize_categoricals_strips_whitespace(sample_df):
    result = standardize_categoricals(sample_df)
    assert result["gender"].iloc[1] == "Male"  # " Male" -> "Male"


def test_treat_outliers_clips_values(sample_df):
    df = fix_data_types(sample_df)
    df = handle_missing_values(df)
    result = treat_outliers(df, ["tenure", "MonthlyCharges"])
    assert result["tenure"].max() <= df["tenure"].quantile(0.99)
    assert result["MonthlyCharges"].min() >= df["MonthlyCharges"].quantile(0.01)


def test_inspect_data_shape(sample_df):
    report = inspect_data(sample_df)
    assert report["shape"] == sample_df.shape
    assert report["duplicates"] == 1  # one duplicate row


def test_inspect_data_missing(sample_df):
    df = fix_data_types(sample_df)
    report = inspect_data(df)
    assert report["missing_values"]["TotalCharges"] >= 1


def test_full_pipeline_produces_clean_df():
    """Integration test: run the full preprocessing pipeline."""
    from src.data_preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline()
    assert len(df) > 0
    assert "Churn" in df.columns
    assert df["Churn"].dtype in [int, np.int64]
    assert df["TotalCharges"].isnull().sum() == 0
    assert df.duplicated().sum() == 0
