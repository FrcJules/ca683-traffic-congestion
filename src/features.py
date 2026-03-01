"""
Feature Engineering Utilities

This module provides functions to create features for traffic prediction models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Create time-based features from timestamp.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with time features added

    Example:
        >>> df_features = create_time_features(df)
    """
    df_copy = df.copy()

    # Ensure datetime type
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    # Extract temporal features
    df_copy['hour'] = df_copy[timestamp_col].dt.hour
    df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
    df_copy['day_of_month'] = df_copy[timestamp_col].dt.day
    df_copy['month'] = df_copy[timestamp_col].dt.month
    df_copy['week_of_year'] = df_copy[timestamp_col].dt.isocalendar().week
    df_copy['year'] = df_copy[timestamp_col].dt.year

    # Binary features
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
    df_copy['is_rush_hour'] = df_copy['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_copy['is_morning_rush'] = df_copy['hour'].isin([7, 8, 9]).astype(int)
    df_copy['is_evening_rush'] = df_copy['hour'].isin([17, 18, 19]).astype(int)

    # Cyclical encoding for hour (sine/cosine transformation)
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)

    # Cyclical encoding for day of week
    df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

    # Cyclical encoding for month
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)

    print(f"✓ Created {15} time-based features")

    return df_copy


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 24],
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create lag features for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lag features for
        lags: List of lag periods (e.g., [1, 2, 24] for 1h, 2h, 24h lags)
        group_by: Optional column to group by (e.g., location_id)

    Returns:
        DataFrame with lag features added

    Example:
        >>> df_features = create_lag_features(df, columns=['speed', 'volume'], lags=[1, 24])
    """
    df_copy = df.copy()

    for col in columns:
        for lag in lags:
            lag_col_name = f'{col}_lag_{lag}'

            if group_by:
                df_copy[lag_col_name] = df_copy.groupby(group_by)[col].shift(lag)
            else:
                df_copy[lag_col_name] = df_copy[col].shift(lag)

    n_features = len(columns) * len(lags)
    print(f"✓ Created {n_features} lag features")

    return df_copy


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [3, 6, 24],
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics features.

    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for
        windows: List of window sizes (e.g., [3, 6, 24] for 3h, 6h, 24h windows)
        group_by: Optional column to group by (e.g., location_id)

    Returns:
        DataFrame with rolling features added

    Example:
        >>> df_features = create_rolling_features(df, columns=['speed'], windows=[3, 24])
    """
    df_copy = df.copy()

    for col in columns:
        for window in windows:
            # Rolling mean
            mean_col_name = f'{col}_rolling_mean_{window}'
            if group_by:
                df_copy[mean_col_name] = df_copy.groupby(group_by)[col].rolling(window, min_periods=1).mean().reset_index(drop=True)
            else:
                df_copy[mean_col_name] = df_copy[col].rolling(window, min_periods=1).mean()

            # Rolling std
            std_col_name = f'{col}_rolling_std_{window}'
            if group_by:
                df_copy[std_col_name] = df_copy.groupby(group_by)[col].rolling(window, min_periods=1).std().reset_index(drop=True)
            else:
                df_copy[std_col_name] = df_copy[col].rolling(window, min_periods=1).std()

    n_features = len(columns) * len(windows) * 2  # mean + std
    print(f"✓ Created {n_features} rolling window features")

    return df_copy


def create_weather_features(
    df: pd.DataFrame,
    temperature_col: str = 'temperature',
    precipitation_col: str = 'precipitation',
    visibility_col: str = 'visibility',
    wind_speed_col: str = 'wind_speed'
) -> pd.DataFrame:
    """
    Create weather-derived features.

    Args:
        df: Input DataFrame with weather columns
        temperature_col: Temperature column name
        precipitation_col: Precipitation column name
        visibility_col: Visibility column name
        wind_speed_col: Wind speed column name

    Returns:
        DataFrame with weather features added
    """
    df_copy = df.copy()

    # Binary weather conditions
    if precipitation_col in df_copy.columns:
        df_copy['is_raining'] = (df_copy[precipitation_col] > 0).astype(int)
        df_copy['heavy_rain'] = (df_copy[precipitation_col] > 5).astype(int)

    if temperature_col in df_copy.columns:
        df_copy['is_cold'] = (df_copy[temperature_col] < 5).astype(int)
        df_copy['is_hot'] = (df_copy[temperature_col] > 25).astype(int)

    if visibility_col in df_copy.columns:
        df_copy['poor_visibility'] = (df_copy[visibility_col] < 1000).astype(int)

    if wind_speed_col in df_copy.columns:
        df_copy['high_wind'] = (df_copy[wind_speed_col] > 50).astype(int)

    # Composite weather severity index (0-10 scale)
    severity_components = []

    if precipitation_col in df_copy.columns:
        severity_components.append(np.clip(df_copy[precipitation_col] / 10, 0, 3))

    if visibility_col in df_copy.columns:
        severity_components.append(np.clip((10000 - df_copy[visibility_col]) / 3000, 0, 3))

    if wind_speed_col in df_copy.columns:
        severity_components.append(np.clip(df_copy[wind_speed_col] / 30, 0, 3))

    if severity_components:
        df_copy['weather_severity'] = sum(severity_components)

    print(f"✓ Created weather-derived features")

    return df_copy


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple]
) -> pd.DataFrame:
    """
    Create interaction features by multiplying feature pairs.

    Args:
        df: Input DataFrame
        feature_pairs: List of tuples, each containing two feature names to interact

    Returns:
        DataFrame with interaction features added

    Example:
        >>> df = create_interaction_features(df, [('is_raining', 'is_rush_hour')])
    """
    df_copy = df.copy()

    for feat1, feat2 in feature_pairs:
        if feat1 in df_copy.columns and feat2 in df_copy.columns:
            interaction_name = f'{feat1}_x_{feat2}'
            df_copy[interaction_name] = df_copy[feat1] * df_copy[feat2]

    print(f"✓ Created {len(feature_pairs)} interaction features")

    return df_copy


def create_target_variable(
    df: pd.DataFrame,
    speed_col: str = 'speed',
    target_type: str = 'binary',
    threshold: float = 30
) -> pd.DataFrame:
    """
    Create target variable for congestion prediction.

    Args:
        df: Input DataFrame
        speed_col: Column containing speed values
        target_type: Type of target ('binary', 'multiclass', 'regression')
        threshold: Speed threshold for binary classification (km/h)

    Returns:
        DataFrame with target variable added
    """
    df_copy = df.copy()

    if target_type == 'binary':
        # Binary: congested vs not congested
        df_copy['is_congested'] = (df_copy[speed_col] < threshold).astype(int)
        print(f"✓ Created binary target: {df_copy['is_congested'].sum():,} congested samples")

    elif target_type == 'multiclass':
        # Multi-class: low/medium/high congestion
        df_copy['congestion_level'] = pd.cut(
            df_copy[speed_col],
            bins=[0, 20, 40, 100],
            labels=['high', 'medium', 'low']
        )
        print(f"✓ Created multiclass target with 3 levels")

    elif target_type == 'regression':
        # Regression: predict speed directly
        df_copy['target_speed'] = df_copy[speed_col]
        print(f"✓ Created regression target (speed)")

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return df_copy
