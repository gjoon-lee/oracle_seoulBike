# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seoul Bike Share System (따릉이) demand prediction ML pipeline that processes historical trip records to predict bike station net flow 2 hours ahead using XGBoost. The system handles Korean CSV files with CP949 encoding and aggregates hourly station flow data.

## Essential Commands

```bash
# Setup environment
pip install -r requirements.txt

# Create .env file with API keys and database credentials:
cat > .env << 'EOF'
KEY_BIKE_STATION_MASTER=<your_key>
KEY_BIKE_LIST=<your_key>
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bike_data
DB_USER=postgres
DB_PASSWORD=<your_password>
EOF

# Run full pipeline (in order)
python api_test.py                 # Test Seoul API connectivity
python test.py                     # Process historical CSVs to PostgreSQL
python feature_engineering.py      # Generate ML features
python xgboost_train.py            # Train prediction model

# Data processing utilities
python update_station_m.py         # Update station master from API
python bike_availability_cleaner.py # Process availability Excel files
python weather_processer.py        # Process weather data

# Database operations
python db_connection.py            # Test PostgreSQL connection
psql -U postgres -d bike_data -c "SELECT COUNT(*) FROM station_hourly_flow;"

# Monitor processing
tail -f logs/bike_cleaning.log     # Monitor data processing
```

## High-Level Architecture

### Data Flow Pipeline

```
Historical CSVs (CP949) → BikeDataCleaner → PostgreSQL → Feature Engineering → XGBoost
        ↓                        ↓                              ↓
  tpss_bcycl_od_statnhm    station_hourly_flow         BikeFeatureEngineer
                                                               ↓
                                                    net_flow_target_2h prediction
```

### Core Components

1. **Data Ingestion Layer**
   - `test.py` → `BikeDataCleaner` processes Korean CSV files with CP949 encoding
   - `api_test.py` tests Seoul Open API connectivity for station master and bike list data
   - `weather_processer.py` integrates meteorological data from weather_data/

2. **Processing & Storage**
   - **PostgreSQL** (`bike_data` database): Main analytical storage
   - **Chunked processing**: 10,000 rows at a time to manage memory
   - **Column mapping**: Korean headers → English in `BikeDataCleaner.column_mapping`

3. **Feature Engineering** (`feature_engineering.py`, `ft_eng_6months.py`)
   - **Classes**: `BikeFeatureEngineer` (standard), `NetFlowFeatureEngineer` (6-month optimized)
   - **Features**: 53+ time series features including:
     - Lag features: 1-48 hours historical values
     - Rolling statistics: 6, 12, 24-hour windows  
     - Cyclical encoding: hour_sin/cos, day_sin/cos, month_sin/cos
     - Station profiles: from `station_profiles_fixed.csv`
   - **Target**: `net_flow_target_2h` = bikes_arrived - bikes_departed (2 hours ahead)

4. **Model Training** (`xgboost_train.py`)
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

## Critical Implementation Details

### Korean Data Handling
ALWAYS use `encoding='cp949'` for historical CSV files. The `column_mapping` dictionary in `BikeDataCleaner`:
```python
'기준_날짜': 'record_date'
'시작_대여소_ID': 'start_station_id'
'종료_대여소_ID': 'end_station_id'
'전체_건수': 'trip_count'
```

### Seoul Open API Quirks
- Station API uses `RNTLS_ID` (not `RNTL_ID`)
- Station names in `ADDR2` field
- Max 1000 records per page (pagination handled automatically)
- Endpoints:
  - Station Master: `/bikeStationMaster/`
  - Real-time Status: `/bikeList/`

### SQLAlchemy 2.0 Compatibility
```python
# Wrap all SQL with text()
db.execute(text("SELECT * FROM table"))

# Use named params
text("WHERE id = :id"), {"id": value}

# Never use %s placeholders
```

### Memory Management
- Large datasets processed in chunks (`chunksize=10000`)
- Use `gc.collect()` after processing each month
- CSV processing: ~50,000 rows/second

### File Locations
- **Historical data**: `bike_historical_data/YYYY_MM/tpss_bcycl_od_statnhm_YYYYMMDD.csv` or `bike_historical_data/Y2024/2024_MM/`
- **ML features**: `netflow_data/` and `*.parquet` files
- **Models**: `models/netflow_model_YYYYMMDD_HHMMSS.pkl`
- **Logs**: `logs/bike_cleaning.log`, `logs/bike_fetch.log`
- **Weather data**: `weather_data/OBS_ASOS_*.csv`

## Pending Features (from ToDo.md)

1. LightGBM classifier for stockout prediction
2. SHAP/explainable AI integration
3. Enhanced weather data integration

## Notes

- **Missing file**: `bikeList_load.py` referenced in docs but not present - real-time collection functionality may need implementation
- **Station profiles**: Reference to `station_profiles.csv` but using `station_profiles_fixed.csv`
- **Requirements.txt**: Has encoding issues, may need regeneration with proper UTF-8 encoding