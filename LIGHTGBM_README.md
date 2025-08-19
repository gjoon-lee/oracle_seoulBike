# LightGBM Stockout Prediction Pipeline

## Overview
Complete end-to-end pipeline for predicting bike station stockouts 2 hours ahead using LightGBM classifier. Combines availability, netflow, and weather data from 2024 to train a robust prediction model.

## Key Results (Test Validation)
- **ROC-AUC: 0.8955** - Excellent predictive discrimination
- **F1-Score: 0.6177** - Good balance of precision and recall  
- **Accuracy: 85.53%** - Strong overall performance
- **Data Coverage**: 2,068 stations with overlapping availability and netflow data
- **Training Records**: ~3.5M samples from Jan-Oct 2024
- **Test Records**: Nov-Dec 2024 for temporal validation

## Pipeline Components

### 1. Data Preparation (`prepare_lightgbm_data.py`)
- **Input**: PostgreSQL tables (bike_availability_hourly, station_hourly_flow, weather_hourly)
- **Processing**: 
  - Joins data for 2,068 overlapping stations
  - Creates temporal, lag, and rolling features
  - Generates target variables (stockout 2h ahead)
  - Temporal train/test split (80/20)
- **Output**: `lightgbm_train_2024.parquet`, `lightgbm_test_2024.parquet`

### 2. Model Training (`lightgbm_train_classifier.py`)
- **Algorithm**: LightGBM binary classifier with early stopping
- **Features**: 60+ engineered features including:
  - Current availability metrics
  - Historical lag features (1h-168h)
  - Rolling statistics (6h-168h windows)
  - Temporal encodings (cyclical hour/day/month)
  - Weather conditions
  - Station profiles
- **Target**: `is_stockout` (≤2 bikes available) 2 hours ahead
- **Validation**: Comprehensive evaluation with feature importance analysis

### 3. Test Scripts
- **`prepare_lightgbm_data_test.py`**: Quick test with January 2024 sample
- **`lightgbm_test_train.py`**: Validation of complete training pipeline

## Key Features by Importance
1. **`is_nearly_empty`** - Current low availability indicator
2. **`is_stockout`** - Current stockout status  
3. **`available_racks`** - Station capacity pressure
4. **`net_flow`** - Station activity balance
5. **`utilization_rate`** - Capacity utilization ratio
6. **`bikes_arrived/departed`** - Station flow metrics
7. **`hour_sin/cos`** - Time of day patterns

## Data Quality
- **Station Coverage**: 99.96% mapping success (2,779/2,780 stations)
- **Availability Data**: 16.6M records, 87.4% coverage of all stations
- **Netflow Data**: 15.7M records from trip analysis
- **Weather Data**: Complete hourly coverage for 2024
- **Station Overlap**: 72.6% stations have both availability and netflow data

## Usage

### Quick Start
```bash
# Install dependencies
pip install lightgbm==4.5.0

# Test pipeline (fast)
python prepare_lightgbm_data_test.py
python lightgbm_test_train.py

# Full pipeline (slow - processes all 2024 data)
python prepare_lightgbm_data.py
python lightgbm_train_classifier.py
```

### Production Training
```bash
# Generate full 2024 dataset (~3.5M samples)
python prepare_lightgbm_data.py

# Train final model with early stopping
python lightgbm_train_classifier.py
```

## File Outputs
- **`lightgbm_train_2024.parquet`**: Training dataset (Jan-Oct 2024)
- **`lightgbm_test_2024.parquet`**: Test dataset (Nov-Dec 2024)
- **`models/lightgbm_stockout_model_TIMESTAMP.pkl`**: Trained model
- **`models/lightgbm_metadata_TIMESTAMP.json`**: Model configuration
- **`lightgbm_feature_importance.png`**: Feature importance plot

## Performance Optimization
- **Memory Management**: Chunked processing with garbage collection
- **Query Optimization**: Indexed database queries on temporal keys
- **Feature Engineering**: Vectorized operations with pandas/numpy
- **Model Training**: Early stopping to prevent overfitting

## Next Steps
1. **Hyperparameter Tuning**: Grid search for optimal LightGBM parameters
2. **Feature Selection**: Remove redundant features for faster inference
3. **Real-time Deployment**: Create API endpoint for live predictions
4. **Model Monitoring**: Track prediction accuracy over time
5. **Multi-target**: Extend to predict `is_nearly_empty` and `available_bikes`

## Technical Notes
- **Class Imbalance**: Uses `is_unbalance=True` parameter in LightGBM
- **Temporal Validation**: Strict time-based splits to avoid data leakage
- **Missing Data**: Forward-fill for lag features, mean imputation for weather
- **Encoding**: All Korean text handled with CP949 encoding compatibility