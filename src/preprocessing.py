"""
Data Preprocessing Utilities

This module provides functions for data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    columns: Optional[List[str]] = None,
    fill_value: Optional[Union[int, float, str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'ffill', 'bfill', 'constant')
        columns: Specific columns to process (None = all columns)
        fill_value: Value to use for 'constant' strategy

    Returns:
        DataFrame with missing values handled

    Example:
        >>> df_clean = handle_missing_values(df, strategy='mean', columns=['speed', 'volume'])
    """
    df_copy = df.copy()
    target_cols = columns if columns else df_copy.columns

    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=target_cols)
    elif strategy == 'mean':
        for col in target_cols:
            if df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in target_cols:
            if df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
    elif strategy == 'ffill':
        df_copy[target_cols] = df_copy[target_cols].fillna(method='ffill')
    elif strategy == 'bfill':
        df_copy[target_cols] = df_copy[target_cols].fillna(method='bfill')
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'constant' strategy")
        df_copy[target_cols] = df_copy[target_cols].fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_copy


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame
        subset: Column names to consider for identifying duplicates
        keep: Which duplicates to keep ('first', 'last', False)

    Returns:
        DataFrame with duplicates removed
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)

    df_copy = df_copy.drop_duplicates(subset=subset, keep=keep)

    removed_rows = initial_rows - len(df_copy)
    print(f"Removed {removed_rows:,} duplicate rows ({removed_rows/initial_rows*100:.2f}%)")

    return df_copy


def standardize_timestamps(
    df: pd.DataFrame,
    timestamp_col: str,
    target_format: str = '%Y-%m-%d %H:%M:%S'
) -> pd.DataFrame:
    """
    Standardize timestamp column to consistent format.

    Args:
        df: Input DataFrame
        timestamp_col: Name of the timestamp column
        target_format: Desired datetime format

    Returns:
        DataFrame with standardized timestamps
    """
    df_copy = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_col]):
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    print(f"✓ Standardized {timestamp_col} to datetime format")

    return df_copy


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a numerical column.

    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (IQR multiplier or z-score)

    Returns:
        Boolean Series indicating outliers (True = outlier)

    Example:
        >>> outliers = detect_outliers(df, 'speed', method='iqr', threshold=1.5)
        >>> df_clean = df[~outliers]
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    n_outliers = outliers.sum()
    print(f"Detected {n_outliers:,} outliers in '{column}' ({n_outliers/len(df)*100:.2f}%)")

    return outliers


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from DataFrame.

    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    outliers = detect_outliers(df, column, method, threshold)
    df_clean = df[~outliers].copy()

    return df_clean


def normalize_column(
    df: pd.DataFrame,
    column: str,
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize a numerical column.

    Args:
        df: Input DataFrame
        column: Column to normalize
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()

    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)

    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_copy
