# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seoul Bike Share System (따릉이) data pipeline and ML prediction system that:
- Collects real-time bike availability data from Seoul Open API (2,731+ stations)
- Processes historical trip data with Korean text encoding (CP949)
- Engineers 53+ features for time series prediction
- Trains XGBoost models to predict bike net flow 2 hours ahead

## Essential Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with Seoul Open API keys:
KEY_BIKE_STATION_MASTER=<your_api_key>
KEY_BIKE_LIST=<your_api_key>
```

### Running the Pipeline
```bash
# 1. Test API connectivity
python api_test.py

# 2. Start real-time data collection (runs every 5 minutes)
python bikeList_load.py

# 3. Process historical CSVs into PostgreSQL
python test.py  # Processes all files in bike_historical_data/

# 4. Run exploratory data analysis
python bike_eda.py

# 5. Create ML features
python feature_engineering.py

# 6. Train XGBoost model
python xgboost_train.py
```

### Database Operations
```bash
# PostgreSQL: bike_data @ localhost:5432
# User: postgres (password in db_connection.py - needs .env migration)

# Check data:
psql -U postgres -d bike_data -c "SELECT COUNT(*) FROM station_hourly_flow;"
```

## Architecture and Data Flow

### Data Pipeline Components

1. **Real-time Collection** (`bikeList_load.py`)
   - Fetches from Seoul Open API every 5 minutes
   - Stores in SQLite (`seoul_bike.db`)
   - Handles API pagination (max 1000 stations/request)

2. **Historical Processing** (`bike_data_cleaner.py`)
   - Reads Korean CSV files (CP949 encoding)
   - Creates hourly aggregations
   - Loads to PostgreSQL tables:
     - `raw_bike_trips`: Individual trips
     - `station_hourly_flow`: Hourly stats
     - `data_quality_log`: Processing metadata

3. **Feature Engineering** (`feature_engineering.py`)
   - Time features with cyclical encoding
   - Lag features (1h-48h)
   - Rolling statistics (6h-24h windows)
   - Station type indicators (residential/office/leisure/mixed)
   - Rush hour interactions

4. **Model Training** (`xgboost_train.py`)
   - Target: Net flow 2 hours ahead
   - Time-based split: train_end_date = '2025-05-31'
   - Saves to `models/` with timestamps

### Key Data Structures

**Historical CSV Format (Korean headers)**:
```
기준_날짜 (record_date): YYYYMMDD
집계_기준 (aggregation_type): 출발시간/도착시간
기준_시간대 (time_slot): HHMM
시작_대여소_ID/종료_대여소_ID: Station IDs
전체_건수 (trip_count): Number of trips
```

**Station Hourly Flow Table**:
```sql
station_id, flow_date, flow_hour
bikes_departed, bikes_arrived, net_flow
day_of_week, is_weekend, season
avg_trip_duration_min, avg_trip_distance_m
```

## Critical Implementation Notes

1. **Korean Encoding**: Always use `encoding='cp949'` for historical CSV files
2. **Time Series Split**: Maintain temporal validation with fixed train_end_date
3. **API Rate Limits**: 5-minute intervals for real-time collection
4. **Database Chunks**: Use chunked inserts for large datasets (10,000 rows/chunk)
5. **Station Types**: Pre-computed in `station_profiles.csv`
6. **Logging**: Check `logs/bike_fetch.log` for collection issues

## Directory Structure

```
bike_historical_data/YYYY_MM/  # Historical CSV files by month
netflow_data/                   # ML feature datasets
models/                         # Trained model artifacts
logs/                          # System logs
```

## Current TODO Items (from ToDo.md)

- Implement LightGBM classifier for 2-hour stockout prediction
- Add SHAP values for model explainability
- Integrate weather data features
- Add historical bike availability data