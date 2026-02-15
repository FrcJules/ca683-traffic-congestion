"""
Eventbrite API data collector for Dublin events (Jan-Jun 2023)

This script fetches public events from Eventbrite API filtered by:
- Location: Dublin, Ireland
- Date range: 2023-01-01 to 2023-06-30
- Event types: Concerts, festivals, sports, conferences, etc.

Usage:
    python src/collect_events_data.py

Requirements:
    - Eventbrite API OAuth token (free tier available)
    - Set EVENTBRITE_API_KEY in .env file
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "https://www.eventbriteapi.com/v3"
API_KEY = os.getenv("EVENTBRITE_API_KEY")
OUTPUT_DIR = Path("data/raw/events")
OUTPUT_FILE = OUTPUT_DIR / "dublin_events_jan_jun_2023.csv"

# Search parameters for Dublin, Jan-Jun 2023
SEARCH_PARAMS = {
    "location.address": "Dublin, Ireland",
    "location.within": "25km",  # 25km radius around Dublin city center
    "start_date.range_start": "2023-01-01T00:00:00Z",
    "start_date.range_end": "2023-06-30T23:59:59Z",
    "expand": "venue,category",
    "page_size": 50  # Max results per page
}

# Event categories relevant to traffic impact
RELEVANT_CATEGORIES = [
    "Music", "Sports & Fitness", "Performing & Visual Arts",
    "Community & Culture", "Business & Professional",
    "Film, Media & Entertainment", "Food & Drink"
]


def get_headers():
    """Generate API request headers with authentication."""
    if not API_KEY:
        raise ValueError(
            "EVENTBRITE_API_KEY not found in environment variables. "
            "Please create a .env file with your API key."
        )
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }


def fetch_events():
    """
    Fetch all events matching the search criteria from Eventbrite API.

    Returns:
        list: List of event dictionaries
    """
    all_events = []
    continuation = None
    page = 1

    print(f"Fetching Dublin events from Jan-Jun 2023...")
    print(f"Search radius: {SEARCH_PARAMS['location.within']} around Dublin")

    while True:
        params = SEARCH_PARAMS.copy()
        if continuation:
            params["continuation"] = continuation

        try:
            response = requests.get(
                f"{API_BASE_URL}/events/search/",
                headers=get_headers(),
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            pagination = data.get("pagination", {})

            print(f"Page {page}: Retrieved {len(events)} events")
            all_events.extend(events)

            # Check if there are more pages
            continuation = pagination.get("continuation")
            if not continuation or not events:
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching events: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            break

    print(f"Total events retrieved: {len(all_events)}")
    return all_events


def parse_event(event):
    """
    Extract relevant fields from raw Eventbrite event data.

    Args:
        event (dict): Raw event data from API

    Returns:
        dict: Parsed event with traffic-relevant fields
    """
    # Basic event info
    parsed = {
        "event_id": event.get("id"),
        "name": event.get("name", {}).get("text"),
        "description": event.get("description", {}).get("text", "")[:500],  # Truncate long descriptions
        "url": event.get("url"),

        # Dates and times
        "start_datetime": event.get("start", {}).get("utc"),
        "end_datetime": event.get("end", {}).get("utc"),
        "start_local": event.get("start", {}).get("local"),
        "end_local": event.get("end", {}).get("local"),
        "timezone": event.get("start", {}).get("timezone"),

        # Location (from expanded venue data)
        "venue_name": None,
        "venue_address": None,
        "venue_latitude": None,
        "venue_longitude": None,

        # Event characteristics
        "category": event.get("category", {}).get("name") if event.get("category") else None,
        "is_free": event.get("is_free", False),
        "capacity": event.get("capacity"),

        # Traffic impact indicators
        "is_online": event.get("online_event", False),
        "status": event.get("status")
    }

    # Extract venue information if available
    venue = event.get("venue")
    if venue:
        parsed["venue_name"] = venue.get("name")

        address = venue.get("address", {})
        if address:
            address_parts = [
                address.get("address_1"),
                address.get("city"),
                address.get("postal_code")
            ]
            parsed["venue_address"] = ", ".join([p for p in address_parts if p])
            parsed["venue_latitude"] = address.get("latitude")
            parsed["venue_longitude"] = address.get("longitude")

    return parsed


def process_events(events):
    """
    Convert raw events to pandas DataFrame with cleaned data.

    Args:
        events (list): List of raw event dictionaries

    Returns:
        pd.DataFrame: Processed events data
    """
    parsed_events = [parse_event(event) for event in events]
    df = pd.DataFrame(parsed_events)

    # Convert datetime strings to pandas datetime
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = pd.to_datetime(df["end_datetime"])

    # Filter out online-only events (no traffic impact)
    df = df[df["is_online"] == False]

    # Filter out events with no venue data (can't locate them)
    df = df[df["venue_latitude"].notna()]

    # Add derived features
    df["duration_hours"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 3600
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["hour_of_day"] = df["start_datetime"].dt.hour
    df["month"] = df["start_datetime"].dt.month
    df["is_weekend"] = df["start_datetime"].dt.dayofweek >= 5

    # Traffic impact score (basic heuristic)
    df["estimated_attendance"] = df["capacity"].fillna(100)  # Default to 100 if unknown
    df["traffic_impact_score"] = df.apply(
        lambda row: calculate_traffic_impact(
            row["estimated_attendance"],
            row["duration_hours"],
            row["is_weekend"]
        ),
        axis=1
    )

    return df


def calculate_traffic_impact(attendance, duration, is_weekend):
    """
    Calculate a simple traffic impact score based on event characteristics.

    Args:
        attendance (int): Estimated number of attendees
        duration (float): Event duration in hours
        is_weekend (bool): Whether event is on weekend

    Returns:
        float: Traffic impact score (0-10)
    """
    # Base score from attendance
    if attendance < 50:
        base_score = 1
    elif attendance < 200:
        base_score = 3
    elif attendance < 1000:
        base_score = 5
    elif attendance < 5000:
        base_score = 7
    else:
        base_score = 9

    # Duration modifier (longer events = more sustained impact)
    duration_modifier = min(duration / 4, 1.5)  # Cap at 1.5x

    # Weekend modifier (less commuter traffic impact)
    weekend_modifier = 0.7 if is_weekend else 1.0

    return round(base_score * duration_modifier * weekend_modifier, 2)


def save_data(df):
    """
    Save processed events data to CSV file.

    Args:
        df (pd.DataFrame): Processed events data
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nData saved to: {OUTPUT_FILE}")
    print(f"Total events with location data: {len(df)}")

    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Average attendance: {df['estimated_attendance'].mean():.0f}")
    print(f"Weekend events: {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"High-impact events (score >= 7): {(df['traffic_impact_score'] >= 7).sum()}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Dublin Events Data Collection (Jan-Jun 2023)")
    print("Source: Eventbrite API")
    print("=" * 60)

    # Fetch raw events
    events = fetch_events()

    if not events:
        print("\nNo events found. Please check:")
        print("1. Your API key is valid")
        print("2. The date range and location parameters")
        print("3. Your internet connection")
        return

    # Process and save data
    df = process_events(events)
    save_data(df)

    print("\n✓ Data collection complete!")
    print(f"\nNext steps:")
    print(f"1. Review the data: {OUTPUT_FILE}")
    print(f"2. Run data exploration in notebooks/01_data_exploration.ipynb")
    print(f"3. Merge with traffic and weather data in notebooks/03_data_fusion.ipynb")


if __name__ == "__main__":
    main()
