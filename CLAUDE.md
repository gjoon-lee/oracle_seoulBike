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

# Run full pipeline (in order)
python api_test.py                 # Test API connectivity
python bikeList_load.py            # Start real-time collection (5-min intervals, Ctrl+C to stop)
python test.py                     # Process historical CSVs to PostgreSQL
python feature_engineering.py      # Generate ML features (or ft_eng_6months.py for 6-month batch)
python xgboost_train.py            # Train prediction model

# Data processing utilities
python update_station_m.py         # Update station master from API (run once/year)
python bike_availability_cleaner.py # Process availability Excel files
python weather_processer.py        # Process weather data

# Database operations
python db_connection.py            # Test PostgreSQL connection
psql -U postgres -d bike_data -c "SELECT COUNT(*) FROM station_hourly_flow;"

# Monitoring logs
tail -f bike_fetch.log             # Monitor real-time data collection
tail -f bike_cleaning.log          # Monitor data processing
```

## High-Level Architecture

### Data Flow Pipeline

```
Historical CSVs (CP949) → BikeDataCleaner → PostgreSQL → Feature Engineering → XGBoost
                                ↓                               ↑
                        station_hourly_flow              BikeFeatureEngineer
                                                               ↓
Seoul APIs (5min) → SQLite Buffer → Real-time Features → net_flow_target_2h
```

### Core Components

1. **Data Ingestion Layer**
   - **Historical**: `test.py` → `BikeDataCleaner` processes Korean CSV files with CP949 encoding
   - **Real-time**: `bikeList_load.py` polls Seoul APIs every 5 minutes → SQLite buffer
   - **Weather**: `weather_processer.py` integrates meteorological data

2. **Processing & Storage**
   - **PostgreSQL** (`bike_data` database): Main analytical storage for aggregated data
   - **SQLite** (`seoul_bike.db`): Real-time buffer for API snapshots
   - **Chunked processing**: 10,000 rows at a time to manage memory
   - **Column mapping**: Korean headers → English in `BikeDataCleaner.column_mapping`

3. **Feature Engineering**
   - **Classes**: `BikeFeatureEngineer` (standard), `NetFlowFeatureEngineer` (6-month optimized)
   - **Features**: 53+ time series features including:
     - Lag features: 1-48 hours historical values
     - Rolling statistics: 6, 12, 24-hour windows
     - Cyclical encoding: hour_sin/cos, day_sin/cos, month_sin/cos
     - Station profiles: residential/office/leisure/mixed from `station_profiles.csv`
   - **Target**: `net_flow_target_2h` = bikes_arrived - bikes_departed (2 hours ahead)

4. **Model Training**
   - **Algorithm**: XGBoost regression with early stopping
   - **Validation**: Time-based split at 2025-05-31
   - **Artifacts**: Saved in `models/` with timestamp suffix
   - **Performance**: Test MAE ~3.4 bikes, R² ~0.61

### Database Schema

**PostgreSQL Tables:**
- `raw_bike_trips`: Individual trips with Korean→English column mapping
- `station_hourly_flow`: Hourly aggregations (key table for ML)
- `station_master`: Station metadata from API
- `bike_availability_hourly`: Processed availability with stockout flags
- `rental_station_mapping`: Maps rental IDs to station IDs
- `data_quality_log`: ETL monitoring and error tracking

**Key Relationships:**
- `station_hourly_flow.station_id` → `station_master.station_id`
- Temporal joins on `flow_date` + `flow_hour` for feature engineering

### Critical Implementation Details

1. **Korean Encoding**: ALWAYS use `encoding='cp949'` for historical CSV files. The `column_mapping` dictionary in `BikeDataCleaner` handles garbled headers.

2. **API Quirks**:
   - Station API uses `RNTLS_ID` (not `RNTL_ID`)
   - Station names in `ADDR2` field
   - Max 1000 records per page (pagination handled automatically)

3. **SQLAlchemy 2.0**: 
   - Wrap all SQL with `text()`: `db.execute(text("SELECT * FROM table"))`
   - Use named params: `text("WHERE id = :id"), {"id": value}`
   - Never use `%s` placeholders

4. **Temporal Integrity**: Never leak future data into past predictions. The `train_end_date` parameter enforces this split.

5. **Memory Management**: Large datasets processed in chunks (`chunksize=10000`). Use `gc.collect()` after processing each month.

6. **Station Types**: Pre-computed in `station_profiles.csv` to avoid recalculation. Categories: residential, office, leisure, mixed.

## Configuration & Environment

### Required Environment Variables
```bash
# API Keys
KEY_BIKE_STATION_MASTER=<Seoul Station Master API Key>
KEY_BIKE_LIST=<Seoul Real-time Bike Status API Key>

# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bike_data
DB_USER=postgres
DB_PASSWORD=<your_password>
```

### Database Connection
- **PostgreSQL**: Configured via environment variables in `.env` file
- Connection string built dynamically from env vars in `db_connection.py`

### File Locations
- **Historical data**: `bike_historical_data/YYYY_MM/tpss_bcycl_od_statnhm_YYYYMMDD.csv`
- **ML features**: `netflow_data/` and `*.parquet` files
- **Models**: `models/netflow_model_YYYYMMDD_HHMMSS.pkl`
- **Logs**: `bike_fetch.log`, `bike_cleaning.log`

### Performance Benchmarks
- CSV processing: ~50,000 rows/second
- Feature engineering: ~2 minutes for 1 month of data
- Model training: ~5 minutes for 6 months
- API collection: 1000 stations in ~2 seconds