# Dataset Documentation

This directory contains all datasets used for the CA683 traffic congestion prediction project.

## Directory Structure

```
data/
├── raw/                    # Original, unmodified datasets
│   ├── traffic/            # SCATS traffic sensor data
│   ├── weather/            # Weather measurements
│   └── events/             # Public events data
├── processed/              # Cleaned and merged datasets
└── README.md               # This file
```

**Note**: The `raw/` and `processed/` directories are excluded from Git (see `.gitignore`). Data files must be downloaded separately.

---

## 1. Dublin SCATS Traffic Data

### Source
- **Provider**: Dublin City Council
- **Platform**: [data.gov.ie](https://data.gov.ie/)
- **Direct link**: https://data.smartdublin.ie/dataset/scats-traffic-data

### Description
The SCATS (Sydney Coordinated Adaptive Traffic System) dataset contains real-time traffic measurements from sensors installed across Dublin.

### Data Format
- **File type**: CSV
- **Temporal range**: January 2023 - June 2023
- **Update frequency**: 15-minute intervals
- **Key columns**:
  - `location_id`: Sensor identifier
  - `timestamp`: Measurement time (ISO format)
  - `volume`: Number of vehicles detected
  - `speed`: Average speed (km/h)
  - `occupancy`: Road occupancy percentage
  - `latitude`, `longitude`: Sensor GPS coordinates

### Download Instructions
```bash
# Option 1: Direct download from data.gov.ie
# Visit: https://data.smartdublin.ie/dataset/scats-traffic-data
# Download: "Traffic Data January-June 2023"

# Option 2: API (if available)
# curl -O https://data.smartdublin.ie/api/scats/traffic_2023_h1.csv
```

### Storage Location
Save to: `data/raw/traffic/scats_dublin_2023_h1.csv`

---

## 2. Weather Data

### Sources

#### Option A: Visual Crossing API
- **Website**: https://www.visualcrossing.com/
- **API Docs**: https://www.visualcrossing.com/resources/documentation/weather-api/
- **Free tier**: 1000 requests/day
- **Coverage**: Dublin, Ireland

#### Option B: Met Éireann (Irish Meteorological Service)
- **Website**: https://www.met.ie/
- **Historical Data**: https://www.met.ie/climate/available-data/historical-data
- **Station**: Dublin Airport (closest to city)

### Data Format
- **File type**: CSV / JSON
- **Temporal range**: January 2023 - June 2023
- **Temporal resolution**: Hourly
- **Key variables**:
  - `datetime`: Timestamp
  - `temperature`: Temperature (°C)
  - `precipitation`: Rainfall (mm)
  - `humidity`: Relative humidity (%)
  - `wind_speed`: Wind speed (km/h)
  - `visibility`: Visibility (km)
  - `conditions`: Weather description (rain, clear, fog, etc.)

### Download Instructions

#### Using Visual Crossing API
```python
import requests
import pandas as pd

API_KEY = 'YOUR_API_KEY'  # Get from https://www.visualcrossing.com/sign-up
location = 'Dublin,Ireland'
start_date = '2023-01-01'
end_date = '2023-06-30'

url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}'

params = {
    'unitGroup': 'metric',
    'key': API_KEY,
    'contentType': 'csv'
}

response = requests.get(url, params=params)
with open('data/raw/weather/dublin_weather_2023_h1.csv', 'wb') as f:
    f.write(response.content)
```

### Storage Location
Save to: `data/raw/weather/dublin_weather_2023_h1.csv`

---

## 3. Public Events Data ⚠️ **SYNTHETIC DATA**

### ⚠️ Important Notice

**This dataset contains synthetic (generated) data for academic purposes.**

Historical events data for 2023 was unavailable from public APIs:
- Eventbrite API deprecated location-based search in 2020
- Fáilte Ireland API only provides future events (2026+)
- data.gov.ie has no comprehensive historical events dataset

The synthetic data was generated using realistic patterns based on:
- Known recurring annual events (St Patrick's Festival, Six Nations Rugby, etc.)
- Typical venue schedules for major Dublin locations
- Historical attendance patterns and seasonal trends

### Data Generation Method

**Generator Script**: `src/generate_synthetic_events.py`

The script creates events based on:

#### Known Annual Events (Jan-Jun 2023)
- St Patrick's Festival (March 17-20)
- Six Nations Rugby matches at Aviva Stadium
- Women's Mini Marathon (June)
- Taste of Dublin food festival
- TradFest Temple Bar
- IMBOLC Festival
- Dublin Dance Festival

#### Venue-Based Generation
Events generated for 12 major Dublin venues with real GPS coordinates:

| Venue | Capacity | Event Types |
|-------|----------|-------------|
| Croke Park | 82,300 | GAA, Concerts |
| Aviva Stadium | 51,700 | Rugby, Soccer, Concerts |
| 3Arena | 13,000 | Concerts, Entertainment |
| Bord Gáis Energy Theatre | 2,111 | Theatre, Music |
| Convention Centre Dublin | 8,000 | Conferences |
| RDS Arena | 8,000 | Exhibitions, Sports |
| Phoenix Park | 100,000 | Outdoor Events |
| Olympia Theatre | 1,650 | Theatre, Comedy |
| Temple Bar | 5,000 | Cultural Events |
| St Stephen's Green | 10,000 | Community Events |
| National Stadium | 2,000 | Boxing, Sports |

#### Event Generation Logic
- **Concerts**: 2-3 per week at 3Arena, Olympia Theatre
- **Sports**: GAA matches every 2 weeks, League of Ireland weekly
- **Theatre**: 2-3 shows per week at Bord Gáis, daily at Olympia
- **Conferences**: Bi-weekly at Convention Centre
- **Community**: Weekend markets and festivals

### Data Format

**File**: `data/raw/events/dublin_events_jan_jun_2023.csv`
**Period**: January 1 - June 30, 2023
**Total Events**: 247

#### Key Columns

- `event_id`: Unique identifier (format: DUB{YYYYMMDD}{hash})
- `name`: Event name
- `description`: Brief description
- `url`: Placeholder URL
- `start_datetime`, `end_datetime`: Event times (ISO format)
- `start_date`: Date only
- `venue_name`: Venue name
- `venue_address`: Full address in Dublin
- `venue_latitude`, `venue_longitude`: GPS coordinates
- `address_region`: "Dublin"
- `address_country`: "Ireland"
- `category`: Event category (Entertainment, Sports, Arts, etc.)
- `is_free`: Free admission flag
- `estimated_attendance`: Attendee estimate
- `venue_capacity`: Maximum venue capacity
- `day_of_week`: Day name
- `hour_of_day`: Start hour (14, 19, or 20)
- `month`: Month number (1-6)
- `is_weekend`: Saturday/Sunday flag
- `duration_hours`: Event duration
- `traffic_impact_score`: 0-10 impact score

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Events | 247 |
| Total Estimated Attendance | 1,520,606 |
| Average Attendance | 6,156 |
| Median Attendance | 1,665 |
| Free Events | 14 (5.7%) |
| Weekend Events | 119 (48.2%) |
| High-Impact Events (score ≥ 7) | 43 (17.4%) |

**Events by Category**:
- Entertainment: 132 (53.4%)
- Arts: 78 (31.6%)
- Sports: 16 (6.5%)
- Community: 11 (4.5%)
- Business: 6 (2.4%)
- Cultural: 2 (0.8%)
- Music: 1 (0.4%)
- Food & Drink: 1 (0.4%)

**Events by Month**:
- January: 37
- February: 42
- March: 46
- April: 44
- May: 39
- June: 39

### Traffic Impact Score Calculation

The `traffic_impact_score` is a heuristic estimate based on:

**Attendance-based scoring**:
- < 500 people → Base score 1
- 500-2,000 → Base score 3
- 2,000-10,000 → Base score 5
- 10,000-30,000 → Base score 7
- 30,000+ → Base score 9

**Modifiers**:
- Duration modifier: `min(duration_hours / 6, 1.5)`
- Weekend modifier: `0.7` if weekend, `1.0` if weekday

**Formula**:
```python
score = base_score × duration_modifier × weekend_modifier
```

### Data Generation

To regenerate or modify the synthetic data:

```bash
python src/generate_synthetic_events.py
```

The script will:
1. Generate known annual events
2. Create realistic concert schedules
3. Generate sports events (GAA, rugby, soccer)
4. Add cultural/theatre performances
5. Include business conferences
6. Add community events
7. Calculate traffic impact scores
8. Save to CSV with statistics

**Runtime**: < 5 seconds

### Data Quality

✅ **Strengths**:
- Realistic venue locations (real GPS coordinates)
- Based on actual Dublin venues and their capacities
- Includes known major events (St Patrick's, Six Nations)
- Reasonable temporal distribution
- Consistent data structure

⚠️ **Limitations**:
- Not real historical data
- Attendance estimates are approximations
- May not capture one-off events (protests, emergencies)
- Traffic impact scores are heuristic, not measured
- Exact dates for recurring events may differ from actual 2023

### Academic Use

**When using this dataset, you MUST**:
1. ✅ Disclose it's synthetic data in your report
2. ✅ Explain the generation methodology
3. ✅ Discuss limitations in your analysis
4. ✅ Justify why synthetic data was necessary

**Example acknowledgment for report**:
> "Due to unavailability of historical events data from public APIs,
> synthetic event data was generated based on known recurring events
> and typical venue schedules for major Dublin locations. While this
> data represents realistic patterns, it should be considered an
> approximation for the purposes of demonstrating data fusion and
> predictive modeling techniques."

### Future Enhancements

To improve the dataset:
- Add road closure data from Dublin City Council
- Include weather-cancelled events
- Add Croke Park/Aviva Stadium official calendars
- Incorporate public transport disruptions
- Add construction/roadworks schedules

---

## Data Processing Pipeline

### Step 1: Data Collection
1. Download all three datasets following the instructions above
2. Save to respective `raw/` subdirectories
3. Verify file integrity (check file size, open in spreadsheet)

### Step 2: Data Exploration
Use `notebooks/01_data_exploration.ipynb` to:
- Load each dataset
- Inspect columns, data types, missing values
- Visualize temporal distributions
- Identify data quality issues

### Step 3: Data Cleaning
Use `notebooks/02_data_cleaning.ipynb` to:
- Standardize timestamp formats (ISO 8601)
- Handle missing values (imputation or removal)
- Remove duplicates
- Detect and handle outliers

### Step 4: Data Fusion
Use `notebooks/03_data_fusion.ipynb` to:
- Merge datasets on timestamp
- Align temporal resolutions (aggregate to hourly)
- Join events based on temporal proximity
- Create unified dataset: `data/processed/merged_traffic_weather_events.csv`

---

## Data Quality Notes

### Known Issues
- **Traffic data**: Some sensors may have gaps during maintenance
- **Weather data**: API limits may require splitting requests
- **Events data**: Not all events are officially recorded

### Validation Checklist
- [ ] All files downloaded successfully
- [ ] Timestamps are in correct format
- [ ] No major data gaps (>5% missing)
- [ ] GPS coordinates are within Dublin area
- [ ] Weather values are within reasonable ranges

---

## Contact & Support

For data access issues:
- **SCATS data**: smartdublin@dublincity.ie
- **Weather API**: Visual Crossing support
- **Events data**: Dublin City Council open data team

Last updated: 2024-03-23
