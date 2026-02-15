"""
Fáilte Ireland Open Data API - Dublin Events Collector (Jan-Jun 2023)

Collects public events from Fáilte Ireland's Open Data API and filters for:
- Location: Dublin (addressRegion = "Dublin")
- Date range: 2023-01-01 to 2023-06-30
- Events with valid GPS coordinates

Usage:
    python src/collect_failte_events.py

API Documentation:
    https://data.gov.ie/dataset/events
    https://failteireland.developer.azure-api.net/
"""

import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
API_BASE_URL = "https://failteireland.azure-api.net/opendata-api/v2/events"
OUTPUT_DIR = Path("data/raw/events")
OUTPUT_FILE = OUTPUT_DIR / "dublin_events_jan_jun_2023.csv"

# Date range for filtering
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 6, 30)

# Dublin regions to filter (addressRegion field)
DUBLIN_REGIONS = [
    "Dublin",
    "Dublin City",
    "County Dublin",
    "Co. Dublin",
    "Baile Átha Cliath"  # Irish name
]


def fetch_all_events() -> List[Dict]:
    """
    Fetch all events from Fáilte Ireland API with pagination.

    Returns:
        list: List of all event dictionaries
    """
    all_events = []
    next_url = API_BASE_URL
    page = 1

    print("Fetching events from Fáilte Ireland Open Data API...")
    print("=" * 60)

    while next_url:
        try:
            response = requests.get(
                next_url,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'CA683-Traffic-Research/1.0'
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            events = data.get('value', [])
            print(f"Page {page}: Retrieved {len(events)} events")
            all_events.extend(events)

            # Check for next page
            next_url = data.get('@odata.nextLink')
            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching events: {e}")
            break

    print(f"\nTotal events retrieved: {len(all_events)}")
    return all_events


def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    """
    Parse various date formats from the API.

    Args:
        date_string: Date string from API

    Returns:
        datetime object or None if parsing fails
    """
    if not date_string:
        return None

    # Try different formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_string[:26], fmt)
        except (ValueError, TypeError):
            continue

    return None


def is_dublin_event(event: Dict) -> bool:
    """
    Check if event is in Dublin based on location data.

    Args:
        event: Event dictionary from API

    Returns:
        bool: True if event is in Dublin
    """
    location = event.get('location', {})
    address = location.get('address', {})
    region = address.get('addressRegion', '')

    # Check if region matches Dublin
    if any(dublin in region for dublin in DUBLIN_REGIONS):
        return True

    # Fallback: check if has Dublin GPS coordinates
    # Dublin roughly: 53.2 to 53.5 lat, -6.5 to -6.0 lon
    geo = location.get('geo', {})
    lat = geo.get('latitude')
    lon = geo.get('longitude')

    if lat and lon:
        if 53.2 <= lat <= 53.5 and -6.5 <= lon <= -6.0:
            return True

    return False


def is_in_date_range(event: Dict) -> bool:
    """
    Check if event falls within Jan-Jun 2023.

    Args:
        event: Event dictionary from API

    Returns:
        bool: True if event is in date range
    """
    # Check main startDate
    start_date = parse_date(event.get('startDate'))
    if start_date and START_DATE <= start_date <= END_DATE:
        return True

    # Check eventSchedule for recurring events
    event_schedule = event.get('eventSchedule', [])
    for schedule in event_schedule:
        schedule_start = parse_date(schedule.get('startDate'))
        if schedule_start and START_DATE <= schedule_start <= END_DATE:
            return True

    return False


def extract_event_data(event: Dict) -> Dict:
    """
    Extract relevant fields from raw event data.

    Args:
        event: Raw event dictionary from API

    Returns:
        dict: Cleaned event data for traffic analysis
    """
    location = event.get('location', {})
    address = location.get('address', {})
    geo = location.get('geo', {})
    offers = event.get('offers', {})
    organizer = event.get('organizer', {})

    # Parse dates
    start_date = parse_date(event.get('startDate'))
    end_date = parse_date(event.get('endDate'))

    # Extract event schedule (recurring events)
    event_schedule = event.get('eventSchedule', [])
    schedule_dates = []
    for schedule in event_schedule:
        sched_start = parse_date(schedule.get('startDate'))
        if sched_start and START_DATE <= sched_start <= END_DATE:
            schedule_dates.append(sched_start.isoformat())

    return {
        'event_id': event.get('id'),
        'name': event.get('name'),
        'description': event.get('description', '')[:500],  # Truncate
        'url': event.get('url'),

        # Main dates
        'start_datetime': start_date.isoformat() if start_date else None,
        'end_datetime': end_date.isoformat() if end_date else None,

        # Recurring schedule
        'is_recurring': len(event_schedule) > 1,
        'schedule_count': len(schedule_dates),
        'schedule_dates': '; '.join(schedule_dates) if schedule_dates else None,

        # Location
        'venue_name': location.get('name'),
        'address_region': address.get('addressRegion'),
        'address_country': address.get('addressCountry'),
        'latitude': geo.get('latitude'),
        'longitude': geo.get('longitude'),

        # Event details
        'is_free': event.get('isAccessibleForFree', True),
        'price': offers.get('price'),
        'price_currency': offers.get('priceCurrency'),

        # Categories
        'event_type': event.get('@type', [None])[0] if isinstance(event.get('@type'), list) else event.get('@type'),
        'additional_type': '; '.join(event.get('additionalType', [])) if event.get('additionalType') else None,

        # Contact
        'organizer_phone': organizer.get('telephone'),
        'seller_phone': offers.get('seller', {}).get('telephone'),
        'seller_url': offers.get('seller', {}).get('url')
    }


def calculate_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add traffic-relevant features to the events dataframe.

    Args:
        df: Raw events dataframe

    Returns:
        pd.DataFrame: Enhanced dataframe with traffic features
    """
    # Convert datetime strings to pandas datetime
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])

    # Extract temporal features
    df['start_date'] = df['start_datetime'].dt.date
    df['day_of_week'] = df['start_datetime'].dt.day_name()
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['month'] = df['start_datetime'].dt.month
    df['is_weekend'] = df['start_datetime'].dt.dayofweek >= 5

    # Calculate duration
    df['duration_hours'] = (
        (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600
    ).fillna(3)  # Default 3 hours if no end time

    # Simple traffic impact heuristic
    # Higher impact for: non-free events, recurring events, longer duration
    df['estimated_attendance'] = 200  # Base estimate
    df.loc[~df['is_free'], 'estimated_attendance'] = 500  # Paid events likely larger
    df.loc[df['is_recurring'], 'estimated_attendance'] *= 1.5  # Recurring = popular

    df['traffic_impact_score'] = (
        (df['estimated_attendance'] / 100) *
        (df['duration_hours'] / 4).clip(upper=2) *
        (0.7 if df['is_weekend'].any() else 1.0)
    ).round(2)

    return df


def save_data(events: List[Dict]):
    """
    Save processed events to CSV file.

    Args:
        events: List of processed event dictionaries
    """
    if not events:
        print("\nNo events found matching criteria.")
        return

    df = pd.DataFrame(events)

    # Add traffic features
    df = calculate_traffic_features(df)

    # Sort by date
    df = df.sort_values('start_datetime')

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Data saved to: {OUTPUT_FILE}")
    print(f"Total Dublin events (Jan-Jun 2023): {len(df)}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")
    print(f"\nEvents by month:")
    print(df.groupby('month').size().to_dict())
    print(f"\nEvents by day of week:")
    print(df['day_of_week'].value_counts().to_dict())
    print(f"\nFree events: {df['is_free'].sum()} ({df['is_free'].mean()*100:.1f}%)")
    print(f"Recurring events: {df['is_recurring'].sum()} ({df['is_recurring'].mean()*100:.1f}%)")
    print(f"Weekend events: {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"\nAverage duration: {df['duration_hours'].mean():.1f} hours")
    print(f"Average traffic impact score: {df['traffic_impact_score'].mean():.2f}")
    print(f"High-impact events (score >= 5): {(df['traffic_impact_score'] >= 5).sum()}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Fáilte Ireland Events Data Collection")
    print("Dublin Events: January - June 2023")
    print("=" * 60)
    print()

    # Fetch all events
    all_events = fetch_all_events()

    if not all_events:
        print("\nNo events retrieved from API. Exiting.")
        return

    print(f"\n{'='*60}")
    print("FILTERING EVENTS")
    print(f"{'='*60}")

    # Filter for Dublin and date range
    dublin_2023_events = []
    for event in all_events:
        if is_dublin_event(event) and is_in_date_range(event):
            dublin_2023_events.append(extract_event_data(event))

    print(f"Events after Dublin filter: {len(dublin_2023_events)}")

    # Save data
    save_data(dublin_2023_events)

    print(f"\n{'='*60}")
    print("✓ Data collection complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Review the data: {OUTPUT_FILE}")
    print(f"2. Explore in notebook: notebooks/01_data_exploration.ipynb")
    print(f"3. Merge with traffic data: notebooks/03_data_fusion.ipynb")


if __name__ == "__main__":
    main()
