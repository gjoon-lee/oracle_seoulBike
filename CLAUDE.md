# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data collection and analysis system for Seoul's public bike-sharing service. The system continuously monitors bike availability across ~2,729 stations, stores historical data, and provides analytics on usage patterns.

## Common Commands

### Environment Setup
```bash
# Install required dependencies
pip install -r requirements.txt

# Set up database tables (run once)
python bikeList_db.py
```

### Running the Data Collection Pipeline
```bash
# Start continuous bike data collection (runs every 5 minutes)
python bikeList_load.py
```

### API Testing
```bash
# Test API connections and view sample data
python api_test.py
```

### Running Analysis
```bash
# Launch Jupyter notebook for exploratory data analysis
jupyter notebook eda.ipynb
```

### Database Operations
```bash
# Connect to the SQLite database
sqlite3 seoul_bike.db

# View database schema
.schema

# Query recent data
SELECT * FROM bike_availability ORDER BY timestamp DESC LIMIT 10;

# Check data health
SELECT COUNT(DISTINCT station_id) FROM bike_availability WHERE timestamp = (SELECT MAX(timestamp) FROM bike_availability);
```

## Architecture

### Data Flow
1. **API Connection**: Uses Seoul's OpenAPI to fetch real-time bike availability data
2. **Data Collection**: `bikeList_load.py` runs every 5 minutes with batched requests (999 stations per API call)
3. **Storage**: SQLite database stores time-series data with automatic coordinate updates
4. **Analysis**: Jupyter notebook performs EDA to identify patterns, busiest stations, and problematic locations

### Key Components

- **Database Setup** (`bikeList_db.py`): Creates initial database schema with two tables
- **Data Collection** (`bikeList_load.py`): Main pipeline with scheduling, error handling, and health checks
- **API Testing** (`api_test.py`): Simple test script for API connectivity
- **Analysis** (`eda.ipynb`): Exploratory data analysis with time patterns and station analysis

### Database Schema

Two main tables:
```sql
bike_availability (
  station_id TEXT,
  timestamp DATETIME,
  available_bikes INTEGER,
  available_racks INTEGER,
  PRIMARY KEY (station_id, timestamp)
)

stations (
  station_id TEXT PRIMARY KEY,
  station_name TEXT,
  latitude REAL NOT NULL,
  longitude REAL NOT NULL,
  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### Environment Variables

Required in `.env` file:
- `KEY_BIKE_LIST`: For real-time bike availability API
- `KEY_BIKE_STATION_MASTER`: For station location data API

### Data Collection Process

- Fetches data in batches of 999 stations per API request
- 0.5 second delay between requests to avoid rate limiting
- Automatic coordinate updates when available in API response
- Data health checks ensure ~2,700 stations collected per cycle
- Comprehensive logging to `bike_fetch.log`

### Error Handling

- API failures logged but don't crash collection process
- Invalid coordinate data warnings without stopping process
- Health checks with thresholds (expects ≥2,700 stations)
- Graceful handling of missing or malformed API responses