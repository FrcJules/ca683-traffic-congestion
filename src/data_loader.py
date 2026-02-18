"""
Data loading utilities for CA683 Traffic Congestion Project

This module provides functions to load the three datasets:
- Traffic: 6 monthly SCATS files (Jan-Jun 2023)
- Weather: 6 monthly weather files (Jan-Jun 2023)
- Events: Single synthetic events file (Jan-Jun 2023)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def load_traffic_data(
    data_dir: str = '../data/raw/traffic',
    months: Optional[List[str]] = None,
    sample_frac: Optional[float] = None
) -> pd.DataFrame:
    """
    Load Dublin SCATS traffic data from multiple monthly files.

    Parameters:
    -----------
    data_dir : str
        Directory containing traffic CSV files
    months : list, optional
        List of month names to load. If None, loads all months.
        Options: ['January', 'February', 'March', 'April', 'May', 'June']
    sample_frac : float, optional
        Fraction of data to sample (0-1). Useful for large datasets.

    Returns:
    --------
    pd.DataFrame
        Combined traffic data from all requested months
    """
    if months is None:
        months = ['January', 'February', 'March', 'April', 'May', 'June']

    data_dir = Path(data_dir)
    dfs = []

    for month in months:
        file_path = data_dir / f'SCATS{month}2023.csv'

        if not file_path.exists():
            print(f"⚠️  Warning: {file_path.name} not found, skipping...")
            continue

        print(f"Loading {month} 2023...", end=' ')

        try:
            df = pd.read_csv(file_path)

            # Sample if requested
            if sample_frac:
                df = df.sample(frac=sample_frac, random_state=42)

            dfs.append(df)
            print(f"✓ {len(df):,} rows")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not dfs:
        raise FileNotFoundError("No traffic data files found!")

    # Combine all months
    traffic_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total traffic data: {len(traffic_df):,} rows, {traffic_df.shape[1]} columns")

    return traffic_df


def load_weather_data(data_dir: str = '../data/raw/weather') -> pd.DataFrame:
    """
    Load Dublin weather data from multiple monthly files.

    Parameters:
    -----------
    data_dir : str
        Directory containing weather CSV files

    Returns:
    --------
    pd.DataFrame
        Combined weather data from all months
    """
    data_dir = Path(data_dir)

    weather_files = [
        'dublin_weather_2023_01_h1.csv',  # January
        'dublin_weather_2023_02_h1.csv',  # February
        'dublin_weather_2023_03_h1.csv',  # March
        'dublin_weather_2023_04_h1.csv',  # April
        'dublin_weather_2023_05_h1.csv',  # May
        'dublin_weather_2023_06_h1.csv',  # June
    ]

    dfs = []
    month_names = ['January', 'February', 'March', 'April', 'May', 'June']

    for file_name, month in zip(weather_files, month_names):
        file_path = data_dir / file_name

        if not file_path.exists():
            print(f"⚠️  Warning: {file_name} not found, skipping...")
            continue

        print(f"Loading {month} weather...", end=' ')

        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"✓ {len(df):,} rows")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not dfs:
        raise FileNotFoundError("No weather data files found!")

    # Combine all months
    weather_df = pd.concat(dfs, ignore_index=True)

    # Parse datetime
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

    # Remove duplicates if any
    weather_df = weather_df.drop_duplicates(subset=['datetime'], keep='first')

    # Sort by datetime
    weather_df = weather_df.sort_values('datetime').reset_index(drop=True)

    print(f"\n✓ Total weather data: {len(weather_df):,} rows, {weather_df.shape[1]} columns")
    print(f"  Date range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")

    return weather_df


def load_events_data(data_dir: str = '../data/raw/events') -> pd.DataFrame:
    """
    Load Dublin events data (synthetic dataset).

    Parameters:
    -----------
    data_dir : str
        Directory containing events CSV file

    Returns:
    --------
    pd.DataFrame
        Events data with parsed datetime columns
    """
    data_dir = Path(data_dir)
    file_path = data_dir / 'dublin_events_jan_jun_2023.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"Events file not found: {file_path}")

    print("Loading events data...", end=' ')

    try:
        events_df = pd.read_csv(file_path)

        # Parse datetime columns
        datetime_cols = ['start_datetime', 'end_datetime', 'start_date']
        for col in datetime_cols:
            if col in events_df.columns:
                events_df[col] = pd.to_datetime(events_df[col])

        print(f"✓ {len(events_df):,} events")
        print(f"  Date range: {events_df['start_datetime'].min()} to {events_df['start_datetime'].max()}")
        print(f"  Categories: {events_df['category'].nunique()} unique")
        print(f"  Venues: {events_df['venue_name'].nunique()} unique")

        return events_df

    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def load_all_data(
    sample_traffic: Optional[float] = None,
    traffic_months: Optional[List[str]] = None
) -> tuple:
    """
    Load all three datasets at once.

    Parameters:
    -----------
    sample_traffic : float, optional
        Fraction of traffic data to sample (0-1)
    traffic_months : list, optional
        Specific months to load for traffic data

    Returns:
    --------
    tuple of (traffic_df, weather_df, events_df)
    """
    print("=" * 70)
    print("LOADING ALL DATASETS")
    print("=" * 70)
    print()

    print("1. TRAFFIC DATA")
    print("-" * 70)
    traffic_df = load_traffic_data(
        sample_frac=sample_traffic,
        months=traffic_months
    )
    print()

    print("2. WEATHER DATA")
    print("-" * 70)
    weather_df = load_weather_data()
    print()

    print("3. EVENTS DATA")
    print("-" * 70)
    events_df = load_events_data()
    print()

    print("=" * 70)
    print("✓ ALL DATA LOADED SUCCESSFULLY")
    print("=" * 70)

    return traffic_df, weather_df, events_df


def get_data_summary(traffic_df: pd.DataFrame, weather_df: pd.DataFrame, events_df: pd.DataFrame):
    """
    Print summary statistics for all datasets.

    Parameters:
    -----------
    traffic_df : pd.DataFrame
        Traffic data
    weather_df : pd.DataFrame
        Weather data
    events_df : pd.DataFrame
        Events data
    """
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\n📊 TRAFFIC DATA:")
    print(f"   Rows: {len(traffic_df):,}")
    print(f"   Columns: {traffic_df.shape[1]}")
    print(f"   Memory: {traffic_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Missing values: {traffic_df.isnull().sum().sum():,}")

    print(f"\n🌤️  WEATHER DATA:")
    print(f"   Rows: {len(weather_df):,}")
    print(f"   Columns: {weather_df.shape[1]}")
    print(f"   Missing values: {weather_df.isnull().sum().sum():,}")
    print(f"   Temperature range: {weather_df['temp'].min():.1f}°C to {weather_df['temp'].max():.1f}°C")

    print(f"\n🎉 EVENTS DATA:")
    print(f"   Rows: {len(events_df):,}")
    print(f"   Columns: {events_df.shape[1]}")
    print(f"   Missing values: {events_df.isnull().sum().sum():,}")
    print(f"   Free events: {events_df['is_free'].sum()} ({events_df['is_free'].mean()*100:.1f}%)")
    print(f"   Weekend events: {events_df['is_weekend'].sum()} ({events_df['is_weekend'].mean()*100:.1f}%)")
    print(f"   High-impact events (≥7): {(events_df['traffic_impact_score'] >= 7).sum()}")


# Quick usage example
if __name__ == "__main__":
    # Load all data (sample 10% of traffic for testing)
    traffic, weather, events = load_all_data(sample_traffic=0.1)

    # Show summary
    get_data_summary(traffic, weather, events)
