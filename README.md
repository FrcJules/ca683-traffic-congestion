# Predicting Urban Traffic Congestion in Dublin

CA683 — Main Continuous Assessment

## Overview

This project predicts urban traffic congestion in Dublin using multi-source data fusion combining:
- Traffic sensor data (SCATS system)
- Weather conditions
- Public events

The goal is to build machine learning models that can accurately forecast traffic congestion patterns to help with urban planning and real-time traffic management.

## Datasets

### 1. Dublin City SCATS Traffic Data
- **Period**: January - June 2023
- **Source**: Dublin City Council / data.gov.ie
- **Content**: Traffic flow, volume, and speed measurements from sensors across Dublin

### 2. Weather Data
- **Sources**: Visual Crossing API / Met Éireann
- **Content**: Temperature, precipitation, wind speed, visibility
- **Temporal resolution**: Hourly/daily measurements

### 3. Public Events Data ⚠️ **Synthetic Data**
- **Source**: Generated synthetic dataset (historical data unavailable)
- **Generator**: `src/generate_synthetic_events.py`
- **Content**: 247 events including concerts, sports, festivals, cultural events
- **Venues**: 12 major Dublin venues with real GPS coordinates
- **Impact**: Events that may influence traffic patterns
- **Note**: Based on realistic patterns but not actual historical records

## Project Structure

```
├── data/
│   ├── raw/                   # Original datasets (not tracked in Git)
│   │   ├── traffic/
│   │   ├── weather/
│   │   └── events/
│   ├── processed/             # Cleaned and merged datasets
│   └── README.md              # Dataset documentation and sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_fusion.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Cleaning and transformation
│   ├── fusion.py              # Multi-source data fusion
│   ├── features.py            # Feature engineering
│   └── models.py              # ML model training and evaluation
├── reports/
│   ├── figures/               # Exported visualizations
│   └── final_report.pdf       # Final project report
└── tests/
    └── test_preprocessing.py  # Unit tests
```

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ca683-traffic-congestion.git
cd ca683-traffic-congestion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Workflow

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Explore each dataset individually
   - Identify patterns, outliers, missing values
   - Understand temporal and spatial distributions

2. **Data Cleaning** (`02_data_cleaning.ipynb`)
   - Handle missing values
   - Normalize timestamps
   - Remove duplicates and outliers

3. **Data Fusion** (`03_data_fusion.ipynb`)
   - Merge traffic, weather, and events data
   - Temporal alignment
   - Spatial matching (location-based)

4. **Feature Engineering** (`04_feature_engineering.ipynb`)
   - Create time-based features (hour, day of week, holidays)
   - Lag features (previous traffic conditions)
   - Weather-derived features
   - Event proximity features

5. **Modelling** (`05_modelling.ipynb`)
   - Train multiple models (Random Forest, XGBoost, LightGBM, etc.)
   - Hyperparameter tuning
   - Cross-validation

6. **Evaluation** (`06_evaluation.ipynb`)
   - Model comparison (RMSE, MAE, R²)
   - Feature importance analysis
   - Final model selection

## Technologies

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **API Access**: requests
- **Environment**: jupyter, python-dotenv

## Team

- Jules Francois

## License

This project is for academic purposes (DCU CA683).
