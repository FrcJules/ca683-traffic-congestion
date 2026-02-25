"""
Data Fusion Utilities

This module provides functions to merge multiple data sources.
"""

import pandas as pd
import numpy as np
from typing import Optional


def merge_traffic_weather(
    df_traffic: pd.DataFrame,
    df_weather: pd.DataFrame,
    traffic_time_col: str = 'timestamp',
    weather_time_col: str = 'datetime',
    merge_strategy: str = 'nearest'
) -> pd.DataFrame:
    """
    Merge traffic and weather datasets on timestamp.

    Args:
        df_traffic: Traffic DataFrame
        df_weather: Weather DataFrame
        traffic_time_col: Name of timestamp column in traffic data
        weather_time_col: Name of timestamp column in weather data
        merge_strategy: How to handle time mismatches ('nearest', 'left', 'right')

    Returns:
        Merged DataFrame

    Example:
        >>> df_merged = merge_traffic_weather(df_traffic, df_weather)
    """
    # Ensure timestamps are datetime
    df_traffic[traffic_time_col] = pd.to_datetime(df_traffic[traffic_time_col])
    df_weather[weather_time_col] = pd.to_datetime(df_weather[weather_time_col])

    # Rename weather timestamp to match traffic
    df_weather_renamed = df_weather.rename(columns={weather_time_col: traffic_time_col})

    if merge_strategy == 'nearest':
        # Sort both DataFrames by timestamp
        df_traffic_sorted = df_traffic.sort_values(traffic_time_col)
        df_weather_sorted = df_weather_renamed.sort_values(traffic_time_col)

        # Merge using nearest timestamp (merge_asof)
        df_merged = pd.merge_asof(
            df_traffic_sorted,
            df_weather_sorted,
            on=traffic_time_col,
            direction='nearest',
            tolerance=pd.Timedelta('1 hour')
        )
    else:
        # Standard merge
        df_merged = pd.merge(
            df_traffic,
            df_weather_renamed,
            on=traffic_time_col,
            how=merge_strategy
        )

    print(f"✓ Merged traffic and weather data: {df_merged.shape[0]:,} rows")

    return df_merged


def add_events_data(
    df: pd.DataFrame,
    df_events: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    event_start_col: str = 'start_date',
    event_end_col: str = 'end_date'
) -> pd.DataFrame:
    """
    Add event proximity features to the main dataset.

    Args:
        df: Main DataFrame (traffic + weather)
        df_events: Events DataFrame
        timestamp_col: Timestamp column in main DataFrame
        event_start_col: Event start date column
        event_end_col: Event end date column

    Returns:
        DataFrame with event features added
    """
    df_copy = df.copy()

    # Ensure datetime types
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    df_events[event_start_col] = pd.to_datetime(df_events[event_start_col])
    df_events[event_end_col] = pd.to_datetime(df_events[event_end_col])

    # Initialize event feature columns
    df_copy['is_event_active'] = 0
    df_copy['event_count'] = 0
    df_copy['hours_to_next_event'] = np.nan

    # For each row, check if there's an active event
    for idx, row in df_copy.iterrows():
        current_time = row[timestamp_col]

        # Check active events
        active_events = df_events[
            (df_events[event_start_col] <= current_time) &
            (df_events[event_end_col] >= current_time)
        ]

        if len(active_events) > 0:
            df_copy.at[idx, 'is_event_active'] = 1
            df_copy.at[idx, 'event_count'] = len(active_events)

        # Find next event
        future_events = df_events[df_events[event_start_col] > current_time]
        if len(future_events) > 0:
            next_event = future_events.iloc[0]
            hours_to_event = (next_event[event_start_col] - current_time).total_seconds() / 3600
            df_copy.at[idx, 'hours_to_next_event'] = hours_to_event

    print(f"✓ Added event features: {df_copy['is_event_active'].sum():,} rows with active events")

    return df_copy


def align_temporal_resolution(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    target_freq: str = '1H',
    agg_method: str = 'mean'
) -> pd.DataFrame:
    """
    Align data to a common temporal resolution (e.g., hourly).

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        target_freq: Target frequency ('1H' for hourly, '15min' for 15-minute)
        agg_method: Aggregation method ('mean', 'sum', 'median', 'max', 'min')

    Returns:
        Resampled DataFrame

    Example:
        >>> df_hourly = align_temporal_resolution(df, target_freq='1H', agg_method='mean')
    """
    df_copy = df.copy()

    # Set timestamp as index
    df_copy = df_copy.set_index(timestamp_col)

    # Resample based on aggregation method
    if agg_method == 'mean':
        df_resampled = df_copy.resample(target_freq).mean()
    elif agg_method == 'sum':
        df_resampled = df_copy.resample(target_freq).sum()
    elif agg_method == 'median':
        df_resampled = df_copy.resample(target_freq).median()
    elif agg_method == 'max':
        df_resampled = df_copy.resample(target_freq).max()
    elif agg_method == 'min':
        df_resampled = df_copy.resample(target_freq).min()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")

    # Reset index to make timestamp a column again
    df_resampled = df_resampled.reset_index()

    print(f"✓ Resampled to {target_freq} using {agg_method}: {df_resampled.shape[0]:,} rows")

    return df_resampled


def create_location_groups(
    df: pd.DataFrame,
    location_col: str = 'location_id',
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Group sensors into clusters based on proximity or traffic patterns.

    Args:
        df: Input DataFrame
        location_col: Column containing location identifiers
        n_clusters: Number of location clusters to create

    Returns:
        DataFrame with location_cluster column added
    """
    # Placeholder: In practice, you'd use K-means on lat/long or traffic patterns
    df_copy = df.copy()

    # Simple approach: assign clusters by location_id hash
    unique_locations = df_copy[location_col].unique()
    location_to_cluster = {loc: i % n_clusters for i, loc in enumerate(unique_locations)}
    df_copy['location_cluster'] = df_copy[location_col].map(location_to_cluster)

    print(f"✓ Created {n_clusters} location clusters")

    return df_copy
