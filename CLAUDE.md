# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seoul Bike Share System (따릉이) demand prediction ML pipeline that processes real-time API data and historical trip records to predict bike station net flow 2 hours ahead using XGBoost.

## Essential Commands

```bash
# Setup environment
pip install -r requirements.txt

# Create .env file with API keys:
echo "KEY_BIKE_STATION_MASTER=<your_key>" >> .env
echo "KEY_BIKE_LIST=<your_key>" >> .env

# Run full pipeline
python api_test.py                 # Test API connectivity
python bikeList_load.py            # Start real-time collection (5-min intervals)
python test.py                     # Process historical CSVs to PostgreSQL
python feature_engineering.py      # Generate ML features
python xgboost_train.py            # Train prediction model

# Station and availability data
python update_station_m.py         # Update station master from API (run once)
python bike_availability_cleaner.py # Process availability Excel files

# Database checks
python db_connection.py            # Test PostgreSQL connection
psql -U postgres -d bike_data -c "SELECT COUNT(*) FROM station_hourly_flow;"
```

## High-Level Architecture

### Data Flow Pipeline

1. **Data Ingestion** → Two parallel streams:
   - **Real-time**: Seoul Open API → SQLite (`seoul_bike.db`) via scheduled fetching
   - **Historical**: Korean CSV files → PostgreSQL via batch processing

2. **Processing Layer** → Handles encoding and aggregation:
   - `BikeDataCleaner` class manages CP949 Korean encoding issues
   - Creates hourly aggregations from raw trip records
   - Validates data quality and logs processing metrics

3. **Feature Engineering** → Time series feature generation:
   - `BikeFeatureEngineer` creates 53+ ML features
   - Lag features (1-48 hours), rolling windows (6-24 hours)
   - Cyclical time encoding (hour, day, season)
   - Station profiling (residential/office/leisure/mixed types)

4. **Model Training** → XGBoost regression:
   - Target: `net_flow_target_2h` (bikes arrived - departed, 2 hours ahead)
   - Time-based train/test split at 2025-05-31
   - Outputs: model artifacts, feature importance, predictions

### Database Schema

**PostgreSQL** (main storage):
- `raw_bike_trips`: Individual trip records with Korean column mappings
- `station_hourly_flow`: Aggregated hourly metrics per station
- `station_master`: Station metadata from API (coordinates, addresses)
- `bike_availability_hourly`: Processed availability data with stockout indicators
- `rental_station_mapping`: Maps rental numbers to station IDs
- `data_quality_log`: ETL pipeline monitoring

**SQLite** (real-time buffer):
- `seoul_bike.db`: Latest station availability snapshots

### Critical Design Patterns

1. **Encoding Management**: All historical data requires `encoding='cp949'` due to Korean text. The `column_mapping` dict in `BikeDataCleaner` translates garbled headers.

2. **Temporal Integrity**: Features use strict time-based ordering. Never use future data for past predictions. The `train_end_date` parameter ensures proper validation splits.

3. **Chunked Processing**: Large datasets are processed in 10,000-row chunks to prevent memory issues. See `bike_data_cleaner.py` for implementation.

4. **Dual Storage Strategy**: SQLite for real-time buffering (fast writes), PostgreSQL for analytical queries (complex joins).

5. **Station Profiling**: Pre-computed station types in `station_profiles.csv` classify usage patterns without repeated calculations.

6. **SQLAlchemy 2.0 Compatibility**: All raw SQL strings must be wrapped with `text()` from `sqlalchemy`. Use named parameters (`:param`) instead of `%s` placeholders, and pass parameters as dictionaries.

## Configuration Notes

- **Database credentials**: Currently hardcoded in `db_connection.py:15` - migrate to .env
- **API column names**: Station API uses `RNTLS_ID` (not `RNTL_ID`) and `ADDR2` for station names
- **API pagination**: Max 1000 stations per request, handled in `bikeList_load.py:25-50`
- **Logging**: Separate logs for data collection (`bike_fetch.log`) and processing (`bike_cleaning.log`)
- **Model versioning**: Timestamps appended to all model artifacts in `models/`