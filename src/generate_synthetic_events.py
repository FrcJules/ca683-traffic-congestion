"""
Synthetic Events Data Generator for Dublin (Jan-Jun 2023)

Generates realistic event data based on:
- Known recurring annual events (St Patrick's Day, marathons, etc.)
- Typical venue schedules (Croke Park, Aviva Stadium, 3Arena, etc.)
- Seasonal patterns and Dublin event calendar

This synthetic data is created for academic purposes when historical
event data is unavailable from public APIs.

Usage:
    python src/generate_synthetic_events.py

Output:
    data/raw/events/dublin_events_jan_jun_2023.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Output configuration
OUTPUT_DIR = Path("data/raw/events")
OUTPUT_FILE = OUTPUT_DIR / "dublin_events_jan_jun_2023.csv"

# Dublin venues with GPS coordinates
VENUES = {
    "Croke Park": {
        "address": "Jones's Rd, Drumcondra, Dublin 3",
        "latitude": 53.3607,
        "longitude": -6.2515,
        "capacity": 82300,
        "types": ["Sports", "Concerts"]
    },
    "Aviva Stadium": {
        "address": "62 Lansdowne Rd, Ballsbridge, Dublin 4",
        "latitude": 53.3356,
        "longitude": -6.2283,
        "capacity": 51700,
        "types": ["Sports", "Concerts"]
    },
    "3Arena": {
        "address": "North Wall Quay, Dublin 1",
        "latitude": 53.3478,
        "longitude": -6.2297,
        "capacity": 13000,
        "types": ["Concerts", "Entertainment"]
    },
    "Bord Gáis Energy Theatre": {
        "address": "Grand Canal Square, Docklands, Dublin 2",
        "latitude": 53.3417,
        "longitude": -6.2368,
        "capacity": 2111,
        "types": ["Theatre", "Music"]
    },
    "National Stadium": {
        "address": "South Circular Rd, Dublin 8",
        "latitude": 53.3344,
        "longitude": -6.2887,
        "capacity": 2000,
        "types": ["Sports", "Boxing"]
    },
    "RDS Arena": {
        "address": "Merrion Rd, Ballsbridge, Dublin 4",
        "latitude": 53.3256,
        "longitude": -6.2305,
        "capacity": 8000,
        "types": ["Exhibitions", "Sports", "Concerts"]
    },
    "Dublin City Centre": {
        "address": "O'Connell Street, Dublin 1",
        "latitude": 53.3498,
        "longitude": -6.2603,
        "capacity": 50000,
        "types": ["Parade", "Festival"]
    },
    "Temple Bar": {
        "address": "Temple Bar, Dublin 2",
        "latitude": 53.3455,
        "longitude": -6.2636,
        "capacity": 5000,
        "types": ["Cultural", "Festival"]
    },
    "Phoenix Park": {
        "address": "Phoenix Park, Dublin 8",
        "latitude": 53.3558,
        "longitude": -6.3298,
        "capacity": 100000,
        "types": ["Sports", "Festival", "Outdoor"]
    },
    "St Stephen's Green": {
        "address": "St Stephen's Green, Dublin 2",
        "latitude": 53.3376,
        "longitude": -6.2597,
        "capacity": 10000,
        "types": ["Cultural", "Outdoor"]
    },
    "Olympia Theatre": {
        "address": "72 Dame St, Dublin 2",
        "latitude": 53.3441,
        "longitude": -6.2650,
        "capacity": 1650,
        "types": ["Theatre", "Comedy", "Music"]
    },
    "Convention Centre Dublin": {
        "address": "Spencer Dock, North Wall Quay, Dublin 1",
        "latitude": 53.3486,
        "longitude": -6.2392,
        "capacity": 8000,
        "types": ["Conference", "Business"]
    }
}

# Known annual events in Dublin (Jan-Jun)
ANNUAL_EVENTS = [
    {
        "name": "St Patrick's Festival",
        "date": "2023-03-17",
        "duration_days": 4,
        "venue": "Dublin City Centre",
        "category": "Cultural",
        "attendance": 50000,
        "is_free": True
    },
    {
        "name": "Women's Mini Marathon",
        "date": "2023-06-04",
        "duration_days": 1,
        "venue": "Dublin City Centre",
        "category": "Sports",
        "attendance": 30000,
        "is_free": False
    },
    {
        "name": "Taste of Dublin",
        "date": "2023-06-15",
        "duration_days": 4,
        "venue": "Phoenix Park",
        "category": "Food & Drink",
        "attendance": 25000,
        "is_free": False
    },
    {
        "name": "Dublin Dance Festival",
        "date": "2023-05-09",
        "duration_days": 10,
        "venue": "Temple Bar",
        "category": "Arts",
        "attendance": 8000,
        "is_free": False
    },
    {
        "name": "Ireland v England - Six Nations Rugby",
        "date": "2023-03-18",
        "duration_days": 1,
        "venue": "Aviva Stadium",
        "category": "Sports",
        "attendance": 51700,
        "is_free": False
    },
    {
        "name": "Ireland v Scotland - Six Nations Rugby",
        "date": "2023-02-05",
        "duration_days": 1,
        "venue": "Aviva Stadium",
        "category": "Sports",
        "attendance": 51700,
        "is_free": False
    },
    {
        "name": "Ireland v Wales - Six Nations Rugby",
        "date": "2023-02-25",
        "duration_days": 1,
        "venue": "Aviva Stadium",
        "category": "Sports",
        "attendance": 51700,
        "is_free": False
    },
    {
        "name": "TradFest Temple Bar",
        "date": "2023-01-25",
        "duration_days": 5,
        "venue": "Temple Bar",
        "category": "Music",
        "attendance": 12000,
        "is_free": True
    },
    {
        "name": "IMBOLC Festival",
        "date": "2023-02-01",
        "duration_days": 4,
        "venue": "St Stephen's Green",
        "category": "Cultural",
        "attendance": 5000,
        "is_free": True
    }
]


def generate_concerts(start_date, end_date):
    """Generate realistic concert schedule for Dublin venues."""
    concerts = []

    # 3Arena - 2-3 concerts per week
    current = start_date
    while current <= end_date:
        if random.random() < 0.4:  # 40% chance per day
            concerts.append({
                "name": random.choice([
                    "International Pop Concert", "Rock Festival Night",
                    "Irish Music Night", "Comedy Show", "DJ Night",
                    "Musical Theatre Performance", "Classical Concert"
                ]),
                "date": current.strftime("%Y-%m-%d"),
                "duration_days": random.choice([1, 2]),
                "venue": "3Arena",
                "category": "Entertainment",
                "attendance": random.randint(5000, 13000),
                "is_free": False
            })
        current += timedelta(days=1)

    return concerts


def generate_sports_events(start_date, end_date):
    """Generate GAA and other sports events."""
    sports = []

    # GAA League matches at Croke Park (Feb-Apr)
    gaa_dates = pd.date_range("2023-02-05", "2023-04-30", freq="2W")
    for date in gaa_dates:
        if start_date <= date.date() <= end_date:
            sports.append({
                "name": random.choice([
                    "GAA National Football League",
                    "GAA National Hurling League",
                    "Dublin GAA Championship"
                ]),
                "date": date.strftime("%Y-%m-%d"),
                "duration_days": 1,
                "venue": "Croke Park",
                "category": "Sports",
                "attendance": random.randint(30000, 82000),
                "is_free": False
            })

    # League of Ireland matches at Aviva/RDS
    soccer_dates = pd.date_range(start_date, end_date, freq="W")
    for date in soccer_dates:
        if random.random() < 0.3:
            sports.append({
                "name": "League of Ireland Football",
                "date": date.strftime("%Y-%m-%d"),
                "duration_days": 1,
                "venue": random.choice(["Aviva Stadium", "RDS Arena"]),
                "category": "Sports",
                "attendance": random.randint(10000, 30000),
                "is_free": False
            })

    return sports


def generate_cultural_events(start_date, end_date):
    """Generate theatre, exhibitions, and cultural events."""
    cultural = []

    # Bord Gáis Theatre - regular performances
    current = start_date
    while current <= end_date:
        # 2-3 shows per week
        if current.weekday() in [4, 5, 6]:  # Fri, Sat, Sun
            cultural.append({
                "name": random.choice([
                    "Musical Theatre Show", "Opera Performance",
                    "Ballet Performance", "Classical Music Concert",
                    "Contemporary Theatre", "Irish Dance Show"
                ]),
                "date": current.strftime("%Y-%m-%d"),
                "duration_days": random.choice([1, 3, 7]),
                "venue": "Bord Gáis Energy Theatre",
                "category": "Arts",
                "attendance": random.randint(500, 2111),
                "is_free": False
            })
        current += timedelta(days=1)

    # Olympia Theatre
    olympia_dates = pd.date_range(start_date, end_date, freq="3D")
    for date in olympia_dates:
        cultural.append({
            "name": random.choice([
                "Comedy Night", "Live Music Session",
                "Theatre Production", "Stand-up Comedy Special"
            ]),
            "date": date.strftime("%Y-%m-%d"),
            "duration_days": random.choice([1, 2]),
            "venue": "Olympia Theatre",
            "category": "Entertainment",
            "attendance": random.randint(400, 1650),
            "is_free": False
        })

    return cultural


def generate_business_events(start_date, end_date):
    """Generate conferences and business events."""
    business = []

    # Convention Centre - monthly conferences
    conf_dates = pd.date_range(start_date, end_date, freq="2W")
    for date in conf_dates:
        if random.random() < 0.6:
            business.append({
                "name": random.choice([
                    "Tech Conference Dublin",
                    "Business Summit Ireland",
                    "Medical Conference",
                    "Education Symposium",
                    "Innovation Forum"
                ]),
                "date": date.strftime("%Y-%m-%d"),
                "duration_days": random.choice([1, 2, 3]),
                "venue": "Convention Centre Dublin",
                "category": "Business",
                "attendance": random.randint(500, 5000),
                "is_free": False
            })

    return business


def generate_community_events(start_date, end_date):
    """Generate smaller community and local events."""
    community = []

    # Weekend markets and local events
    weekend_dates = pd.date_range(start_date, end_date, freq="W-SAT")
    for date in weekend_dates:
        if random.random() < 0.4:
            community.append({
                "name": random.choice([
                    "Temple Bar Food Market",
                    "Dublin Flea Market",
                    "Craft Fair",
                    "Farmers Market",
                    "Community Festival"
                ]),
                "date": date.strftime("%Y-%m-%d"),
                "duration_days": random.choice([1, 2]),
                "venue": random.choice(["Temple Bar", "St Stephen's Green"]),
                "category": "Community",
                "attendance": random.randint(500, 3000),
                "is_free": True
            })

    return community


def create_event_dataframe(events_list):
    """Convert events list to structured dataframe."""
    records = []

    for event in events_list:
        venue_info = VENUES[event["venue"]]
        start_dt = datetime.strptime(event["date"], "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=event["duration_days"])

        # Generate event ID
        event_id = f"DUB{start_dt.strftime('%Y%m%d')}{hash(event['name']) % 10000:04d}"

        # Calculate duration in hours
        duration_hours = event["duration_days"] * 6  # Assume 6 hours per day average

        record = {
            "event_id": event_id,
            "name": event["name"],
            "description": f"{event['category']} event at {event['venue']} in Dublin",
            "url": f"https://example.com/events/{event_id}",

            # Dates
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
            "start_date": start_dt.date(),

            # Location
            "venue_name": event["venue"],
            "venue_address": venue_info["address"],
            "venue_latitude": venue_info["latitude"],
            "venue_longitude": venue_info["longitude"],
            "address_region": "Dublin",
            "address_country": "Ireland",

            # Event details
            "category": event["category"],
            "is_free": event["is_free"],
            "estimated_attendance": event["attendance"],
            "venue_capacity": venue_info["capacity"],

            # Temporal features
            "day_of_week": start_dt.strftime("%A"),
            "hour_of_day": random.choice([14, 19, 20]),  # Typical start times
            "month": start_dt.month,
            "is_weekend": start_dt.weekday() >= 5,
            "duration_hours": duration_hours,

            # Traffic impact
            "traffic_impact_score": calculate_impact_score(
                event["attendance"],
                duration_hours,
                start_dt.weekday() >= 5
            )
        }

        records.append(record)

    return pd.DataFrame(records)


def calculate_impact_score(attendance, duration, is_weekend):
    """Calculate traffic impact score (0-10 scale)."""
    # Base score from attendance
    if attendance < 500:
        base = 1
    elif attendance < 2000:
        base = 3
    elif attendance < 10000:
        base = 5
    elif attendance < 30000:
        base = 7
    else:
        base = 9

    # Duration modifier
    duration_mod = min(duration / 6, 1.5)

    # Weekend modifier (less commuter impact)
    weekend_mod = 0.7 if is_weekend else 1.0

    return round(base * duration_mod * weekend_mod, 2)


def main():
    """Main execution function."""
    print("=" * 70)
    print("DUBLIN EVENTS SYNTHETIC DATA GENERATOR")
    print("Period: January - June 2023")
    print("=" * 70)
    print()

    start_date = datetime(2023, 1, 1).date()
    end_date = datetime(2023, 6, 30).date()

    print("Generating events...")
    print("-" * 70)

    # Collect all events
    all_events = []

    # Add known annual events
    print(f"✓ Annual events: {len(ANNUAL_EVENTS)} events")
    all_events.extend(ANNUAL_EVENTS)

    # Generate concerts
    concerts = generate_concerts(start_date, end_date)
    print(f"✓ Concerts & entertainment: {len(concerts)} events")
    all_events.extend(concerts)

    # Generate sports
    sports = generate_sports_events(start_date, end_date)
    print(f"✓ Sports events: {len(sports)} events")
    all_events.extend(sports)

    # Generate cultural
    cultural = generate_cultural_events(start_date, end_date)
    print(f"✓ Cultural & theatre: {len(cultural)} events")
    all_events.extend(cultural)

    # Generate business
    business = generate_business_events(start_date, end_date)
    print(f"✓ Business & conferences: {len(business)} events")
    all_events.extend(business)

    # Generate community
    community = generate_community_events(start_date, end_date)
    print(f"✓ Community events: {len(community)} events")
    all_events.extend(community)

    print()
    print(f"Total events generated: {len(all_events)}")
    print()

    # Create dataframe
    print("Creating structured dataset...")
    df = create_event_dataframe(all_events)

    # Sort by date
    df = df.sort_values('start_datetime')

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✓ Data saved to: {OUTPUT_FILE}")
    print()

    # Print statistics
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")
    print()

    print("Events by category:")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat:20s}: {count:3d} events")
    print()

    print("Events by month:")
    for month, count in df['month'].value_counts().sort_index().items():
        month_name = datetime(2023, month, 1).strftime("%B")
        print(f"  {month_name:12s}: {count:3d} events")
    print()

    print("Events by venue (top 5):")
    for venue, count in df['venue_name'].value_counts().head().items():
        print(f"  {venue:30s}: {count:3d} events")
    print()

    print(f"Free events: {df['is_free'].sum()} ({df['is_free'].mean()*100:.1f}%)")
    print(f"Weekend events: {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")
    print()

    print(f"Total estimated attendance: {df['estimated_attendance'].sum():,}")
    print(f"Average attendance per event: {df['estimated_attendance'].mean():.0f}")
    print(f"Median attendance: {df['estimated_attendance'].median():.0f}")
    print()

    print(f"Average traffic impact score: {df['traffic_impact_score'].mean():.2f}")
    print(f"High-impact events (score >= 7): {(df['traffic_impact_score'] >= 7).sum()}")
    print()

    print("=" * 70)
    print("✓ SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("IMPORTANT: This is synthetic data created for academic purposes.")
    print("Document this limitation in your project report.")
    print()
    print("Next steps:")
    print("  1. Review data: less {OUTPUT_FILE}")
    print("  2. Explore: notebooks/01_data_exploration.ipynb")
    print("  3. Merge with traffic data: notebooks/03_data_fusion.ipynb")


if __name__ == "__main__":
    main()
