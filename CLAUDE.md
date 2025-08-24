# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seoul Bike Share System (따릉이) ML pipeline with dual model approach: LightGBM for stockout classification (primary) and XGBoost for net flow regression. Includes real-time prediction API using FastAPI. Processes Korean CSV files (CP949 encoding) and serves predictions via REST endpoints.

## Essential Commands

```bash
# Setup environment
pip install -r requirements.txt
source .venv/Scripts/activate      # Windows
source .venv/bin/activate          # Linux/Mac

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
python update_station_m.py         # Update station master from API
python weather_processer.py        # Process weather data (drops station columns)
python bike_availability_cleaner.py # Process availability data (uses cached mapping)
python test.py                     # Process historical trip CSVs
python feature_engineering.py      # Generate ML features (XGBoost)
python ft_eng_6months.py           # Generate 6-month optimized features
python xgboost_train.py            # Train XGBoost regression model

# LightGBM stockout prediction pipeline
python prepare_lightgbm_data.py    # Prepare combined dataset (full year)
python lightgbm_train_classifier.py # Train LightGBM classifier
python prepare_lightgbm_data_test.py # Quick test with January sample

# Real-time Prediction API and Dashboard (Windows)
# Terminal 1: Start API
cd realtime_prediction
python main.py  # Or: ..\\.venv\\Scripts\\python.exe main.py
# Access at http://localhost:8002/docs (port changed from 8001 to avoid conflicts)

# Terminal 2: Start Dashboard  
cd streamlit_app
streamlit run dashboard_v2.py  # Or: ..\\.venv\\Scripts\\streamlit.exe run dashboard_v2.py
# Access at http://localhost:8501

# Alternative: Use batch file to run both
run_dashboard.bat  # Starts both API and dashboard

# Database operations
python db_connection.py            # Test PostgreSQL connection
psql -U postgres -d bike_data -c "SELECT COUNT(*) FROM bike_availability_hourly;"
psql -U postgres -d bike_data -c "SELECT COUNT(DISTINCT station_id) FROM rental_station_mapping;"

# Monitor processing
tail -f availability_processing.log  # Monitor availability processing
tail -f logs/bike_cleaning.log      # Monitor trip data processing
tail -f weather_processing.log      # Monitor weather data processing
tail -f realtime_prediction/logs/prediction_api.log  # Monitor API logs

# Fast batch processing (alternative pipeline)
python fast_batch_processor.py     # Process all historical data in parallel
```

## High-Level Architecture

### Data Flow Pipeline

```
Availability CSVs → Mapping Dictionary → bike_availability_hourly
                           ↓
Historical CSVs → BikeDataCleaner → station_hourly_flow
                                            ↓
Weather CSVs → weather_processer → weather_hourly
                                            ↓
                        Feature Engineering → Unified Features → Models
                                                                     ↓
                                                        XGBoost: net_flow_target_2h (regression)
                                                        LightGBM: is_stockout (classification)
```

### Core Components

1. **Station Mapping System**
   - `rental_station_mapping` table: 대여소번호 (00101) → station_id (ST-101)
   - Multi-strategy: Address matching → Coordinate matching (6-2 decimal precision)
   - 99.96% success rate for stations in station_info.xlsx

2. **Data Ingestion Layer**
   - `bike_availability_cleaner.py`: Processes availability data with cached mapping
   - `test.py` → `BikeDataCleaner`: Processes Korean CSV files with CP949 encoding
   - `weather_processer.py`: Integrates weather data (drops station ID/name columns)

3. **Processing & Storage**
   - **PostgreSQL** (`bike_data` database): Main analytical storage
   - **Chunked processing**: 10,000 rows at a time for memory efficiency
   - **Batch inserts**: 50,000 records at a time for speed

4. **Feature Engineering** 
   - **XGBoost Pipeline** (`feature_engineering.py`, `ft_eng_6months.py`)
     - Classes: `BikeFeatureEngineer` (standard), `NetFlowFeatureEngineer` (6-month optimized)
     - Features: 53+ time series features
     - Target: `net_flow_target_2h` = bikes_arrived - bikes_departed (2 hours ahead)
   - **LightGBM Pipeline** (`prepare_lightgbm_data.py`)
     - Class: `LightGBMDataPreparer`
     - Features: 60+ engineered features including lag (1h-168h), rolling (6h-168h)
     - Target: `is_stockout` (≤2 bikes available) 2 hours ahead

5. **Model Training**
   - **XGBoost** (`xgboost_train.py`)
     - Algorithm: XGBoost regression with early stopping
     - Validation: Time-based split at 2025-05-31
     - Performance: Test MAE ~3.4 bikes, R² ~0.61
   - **LightGBM** (`lightgbm_train_classifier.py`)
     - Algorithm: LightGBM binary classifier
     - Performance: ROC-AUC 0.8955, F1 0.6177, Accuracy 85.53%
     - Handles class imbalance with `is_unbalance=True`

6. **Real-time Prediction API** (`realtime_prediction/`)
   - **FastAPI Application** (`main.py`): REST API serving predictions
   - **Data Collectors**: 
     - `realtime_bike_collector.py`: Seoul Open API (bikeList)
     - `realtime_weather_collector.py`: KMA/OpenWeatherMap APIs
   - **Feature Pipeline**: Generates all 110 features in real-time
   - **Prediction Service**: Uses model 20250819_072922
   - **API Endpoints**:
     - `/predict/all`: Batch predictions for all stations
     - `/predict/{station_id}`: Single station prediction
     - `/high-risk`: Stations with >70% stockout probability
   - **Performance**: <100ms latency with caching

### Database Schema

**Core Tables:**
```sql
-- Station mapping (complete, 2,779 mappings)
rental_station_mapping (
    rental_number VARCHAR(20) PRIMARY KEY,  -- 대여소번호
    station_id VARCHAR(20),                 -- ST-XXX format
    station_info_address TEXT,
    match_method VARCHAR(100),              -- How it was matched
    confidence_score REAL,
    created_at TIMESTAMP
)

-- Availability data (processing, 12 months)
bike_availability_hourly (
    station_id VARCHAR(20),
    date DATE,
    hour INTEGER,
    available_bikes REAL,
    station_capacity INTEGER,
    available_racks REAL,
    is_stockout INTEGER,
    is_nearly_empty INTEGER,
    is_nearly_full INTEGER,
    PRIMARY KEY (station_id, date, hour)
)

-- Historical flow data (complete)
station_hourly_flow (
    station_id VARCHAR(20),
    flow_date DATE,
    flow_hour INTEGER,
    bikes_arrived INTEGER,
    bikes_departed INTEGER,
    net_flow INTEGER,
    net_flow_target_2h INTEGER,
    PRIMARY KEY (station_id, flow_date, flow_hour)
)

-- Weather data (complete)
weather_hourly (
    date DATE,
    hour INTEGER,
    temperature REAL,
    humidity REAL,
    [weather columns...],
    PRIMARY KEY (date, hour)
)

-- Station master (from API)
station_master (
    station_id VARCHAR(20) PRIMARY KEY,
    station_name VARCHAR(100),
    station_address VARCHAR(255),
    latitude REAL,
    longitude REAL
)
```

## Critical Implementation Details

### Model Compatibility Fix
LightGBM models saved as Booster objects don't have `predict_proba`. Use:
```python
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X)[:, 1]
else:
    probabilities = model.predict(X, num_iteration=model.best_iteration)
```

### Korean Data Handling
ALWAYS use `encoding='cp949'` for historical CSV files. The `column_mapping` dictionary in `BikeDataCleaner`:
```python
'기준_날짜': 'record_date'
'시작_대여소_ID': 'start_station_id'
'종료_대여소_ID': 'end_station_id'
'전체_건수': 'trip_count'
```

### Station Mapping Strategy
```python
# Pre-computed mapping cached in database
# Load once, use everywhere
rental_to_station = dict(zip(
    rental_mapping['rental_number'], 
    rental_mapping['station_id']
))
# 99.96% coverage for station_info.xlsx stations
# 87.4% coverage for actual data records
```

### Data Coverage Reality
- **Station Coverage**: 2,779/2,780 stations mapped (99.96%)
- **Data Coverage**: 87.4% of availability records processable
- **Gap Reason**: 386 stations in data but not in station_info.xlsx
- **Solution**: Process with available coverage, save unmapped separately

### Seoul Open API Quirks
- Station API uses `RNTLS_ID` (not `RNTL_ID`)
- Station names in `ADDR2` field
- Max 1000 records per page (pagination handled automatically)

### SQLAlchemy 2.0 Compatibility
```python
# Wrap all SQL with text()
from sqlalchemy import text
db.execute(text("SELECT * FROM table"))

# Use named params
text("WHERE id = :id"), {"id": value}

# Never use %s placeholders
```

### Pydantic Installation for Python 3.13
Use pre-built wheels to avoid Rust compilation:
```bash
.venv\\Scripts\\python.exe -m pip install "pydantic==2.10.4" "pydantic-core==2.27.2" --only-binary :all:
```

### Performance Benchmarks
- CSV processing: ~50,000 rows/second
- Feature engineering: ~2 minutes for 1 month
- Model training: ~5 minutes for 6 months
- Availability processing: ~10-20 hours for 12 months

### Memory Management
- Large datasets processed in chunks (`chunksize=10000`)
- Use `gc.collect()` after processing each month
- Batch database inserts (50,000 records at a time)

### File Locations
- **Availability data**: `availability_data/data_YYMM.csv`
- **Historical data**: `bike_historical_data/YYYY_MM/` or `bike_historical_data/Y2024/2024_MM/`
- **ML features**: 
  - XGBoost: `netflow_data/bike_features_*.csv`, `netflow_features_6m_*.parquet`
  - LightGBM: `lightgbm_train_2024.parquet`, `lightgbm_test_2024.parquet`
- **Models**: 
  - XGBoost: `models/netflow_model_YYYYMMDD_HHMMSS.pkl`
  - LightGBM: `models/lightgbm_stockout_model_YYYYMMDD_HHMMSS.pkl`
- **Logs**: `availability_processing.log`, `logs/bike_cleaning.log`, `weather_processing.log`
- **Weather data**: `weather_data/OBS_ASOS_*.csv`
- **Station info**: `station_info.xlsx` (sheet 2 has coordinates and addresses)

## Data Join Strategy for Next Session

```sql
-- Unified feature table joining
SELECT 
    a.station_id,
    a.date,
    a.hour,
    -- Availability features
    a.available_bikes,
    a.station_capacity,
    a.is_stockout,
    -- Net flow features
    n.bikes_arrived,
    n.bikes_departed,
    n.net_flow_target_2h,  -- Prediction target
    -- Weather features
    w.temperature,
    w.humidity
FROM bike_availability_hourly a
LEFT JOIN station_hourly_flow n 
    ON a.station_id = n.station_id 
    AND a.date = n.flow_date 
    AND a.hour = n.flow_hour
LEFT JOIN weather_hourly w 
    ON a.date = w.date 
    AND a.hour = w.hour
WHERE a.date BETWEEN '2024-01-01' AND '2024-12-31'
```

## Known Issues & Solutions

### Issue 1: API Port Conflicts
- **Problem**: Port 8000/8001 already in use error when starting API
- **Solution**: Port changed to 8002 in `realtime_prediction/config/config.py`. Use `http://localhost:8002`

### Issue 2: Model Prediction Error
- **Problem**: `'Booster' object has no attribute 'predict_proba'`
- **Solution**: Check model type and use appropriate predict method (see Model Compatibility Fix)

### Issue 3: Station Data Mismatch
- **Problem**: 386 stations in availability data not in station_info.xlsx
- **Impact**: 12.6% data loss (238,548 records)
- **Solution**: Accept 87.4% coverage, save unmapped data separately

### Issue 4: No Historical Data Warning
- **Problem**: API shows "No availability/netflow history found"
- **Cause**: API looks for data from current date (2025) but database has 2024 data
- **Solution**: This is expected for historical demo data. API works without lag features

## Pending Features (from ToDo.md)

1. ~~LightGBM classifier for stockout prediction~~ ✅ Completed
2. SHAP/explainable AI integration
3. ~~Enhanced weather data integration~~ ✅ Completed
4. Real-time data collection (`bikeList_load.py` implementation)

## Real-time Prediction API Usage

### Quick Start
```bash
cd realtime_prediction
python main.py
# API runs at http://localhost:8002
# Swagger docs at http://localhost:8002/docs
```

### Example API Calls
```bash
# Get all predictions
curl http://localhost:8002/predict/all?mode=balanced

# Single station prediction
curl http://localhost:8002/predict/ST-101

# High risk stations
curl http://localhost:8002/high-risk?threshold=0.7

# Current station status
curl http://localhost:8002/stations/status
```

### API Features
- Real-time data from Seoul bike API (bikeList endpoint)
- Weather from KMA API with OpenWeatherMap fallback
- 168-hour historical cache for lag features
- Prediction modes: alert (high recall) or balanced
- Automatic 5-minute cache refresh

## Notes for Next Session

1. ~~**Verify Processing**: Check if 12-month availability processing completed~~ ✅
2. ~~**Data Combination**: Join availability + netflow + weather tables~~ ✅
3. ~~**Feature Engineering**: Generate comprehensive feature set~~ ✅
4. ~~**Model Training**: Train XGBoost with combined features~~ ✅
5. ~~**Evaluation**: Test on recent months for accuracy~~ ✅
6. **Deploy API**: Production deployment with Docker/Kubernetes
7. **Add monitoring**: Prometheus metrics and Grafana dashboard
8. **Implement scheduler**: Automated background tasks for data refresh

## Model Performance Summary

### LightGBM Stockout Classifier (Primary Model)
- **Model File**: `models/lightgbm_stockout_model_20250819_072922.pkl`
- **ROC-AUC**: 0.8955
- **F1-Score**: 0.6177  
- **Accuracy**: 85.53%
- **Features**: 110 engineered features
- **Target**: Binary classification (stockout in 2 hours)

### XGBoost Net Flow Regressor
- **Target**: Net flow prediction (bikes arrived - departed)
- **MAE**: ~3.4 bikes
- **R²**: ~0.61
- **Features**: 53+ time series features

## Dashboard Versions

Three dashboard versions exist in `streamlit_app/`:
- **dashboard.py**: Original version (564 lines)
- **dashboard_clean.py**: Refactored version (392 lines)  
- **dashboard_v2.py**: Latest version with enhanced features (597 lines, last modified Aug 21 23:52)

Use `dashboard_v2.py` for the most complete experience.

---
Last Updated: 2025-08-24
Status: Real-time prediction API operational with LightGBM model on port 8002