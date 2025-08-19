"""
LightGBM Data Preparation Pipeline
Combines availability, netflow, and weather data for stockout prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from db_connection import BikeDataDB
import logging
from tqdm import tqdm
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightGBMDataPreparer:
    def __init__(self):
        self.db = BikeDataDB()
        self.db.connect()
        logger.info("Connected to database")
        
    def load_base_data(self, start_date='2024-01-01', end_date='2024-12-31'):
        """Load and join availability, netflow, and weather data"""
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        # Get overlapping stations only (both availability and netflow)
        overlap_query = text(f"""
            SELECT DISTINCT a.station_id
            FROM bike_availability_hourly a
            INNER JOIN station_hourly_flow n ON a.station_id = n.station_id 
                AND a.date = n.flow_date
            WHERE a.date >= '{start_date}' AND a.date <= '{end_date}'
        """)
        
        overlap_stations = self.db.read_query(overlap_query)
        station_list = tuple(overlap_stations['station_id'].tolist())
        logger.info(f"Found {len(station_list)} overlapping stations")
        
        # Load availability data
        station_list_str = "', '".join(station_list)
        availability_query = text(f"""
            SELECT 
                station_id,
                date,
                hour,
                available_bikes,
                station_capacity,
                available_racks,
                is_stockout,
                is_nearly_empty,
                is_nearly_full
            FROM bike_availability_hourly
            WHERE station_id IN ('{station_list_str}')
                AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY station_id, date, hour
        """)
        
        logger.info("Loading availability data...")
        availability_df = self.db.read_query(availability_query)
        logger.info(f"Loaded {len(availability_df):,} availability records")
        
        # Load netflow data
        netflow_query = text(f"""
            SELECT 
                station_id,
                flow_date as date,
                flow_hour as hour,
                bikes_departed,
                bikes_arrived,
                net_flow,
                day_of_week,
                is_weekend,
                season,
                avg_trip_duration_min,
                avg_trip_distance_m
            FROM station_hourly_flow
            WHERE station_id IN ('{station_list_str}')
                AND flow_date >= '{start_date}' AND flow_date <= '{end_date}'
            ORDER BY station_id, flow_date, flow_hour
        """)
        
        logger.info("Loading netflow data...")
        netflow_df = self.db.read_query(netflow_query)
        logger.info(f"Loaded {len(netflow_df):,} netflow records")
        
        # Load weather data
        weather_query = text(f"""
            SELECT 
                date,
                hour,
                temperature,
                humidity,
                precipitation,
                wind_speed,
                feels_like,
                is_raining,
                is_snowing,
                weather_severity
            FROM weather_hourly
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date, hour
        """)
        
        logger.info("Loading weather data...")
        weather_df = self.db.read_query(weather_query)
        logger.info(f"Loaded {len(weather_df):,} weather records")
        
        # Join all data
        logger.info("Joining datasets...")
        
        # Primary join: availability + netflow
        combined_df = pd.merge(
            availability_df,
            netflow_df,
            on=['station_id', 'date', 'hour'],
            how='inner'
        )
        logger.info(f"After availability+netflow join: {len(combined_df):,} records")
        
        # Secondary join: add weather
        combined_df = pd.merge(
            combined_df,
            weather_df,
            on=['date', 'hour'],
            how='left'
        )
        logger.info(f"After adding weather: {len(combined_df):,} records")
        
        # Fill missing weather data
        combined_df['temperature'].fillna(combined_df['temperature'].mean(), inplace=True)
        combined_df['humidity'].fillna(combined_df['humidity'].mean(), inplace=True)
        combined_df['precipitation'].fillna(0, inplace=True)
        combined_df['wind_speed'].fillna(combined_df['wind_speed'].mean(), inplace=True)
        combined_df['feels_like'].fillna(combined_df['feels_like'].mean(), inplace=True)
        combined_df['is_raining'].fillna(0, inplace=True)
        combined_df['is_snowing'].fillna(0, inplace=True)
        combined_df['weather_severity'].fillna(0, inplace=True)
        
        logger.info(f"Final combined dataset: {len(combined_df):,} records, {len(combined_df.columns)} columns")
        return combined_df
    
    def create_temporal_features(self, df):
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # Ensure date is datetime
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rush hour indicators
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (~df['is_weekend'])).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (~df['is_weekend'])).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Additional activity features
        df['total_activity'] = df['bikes_departed'] + df['bikes_arrived']
        df['utilization_rate'] = df['available_bikes'] / df['station_capacity'].clip(lower=1)
        df['capacity_pressure'] = (df['station_capacity'] - df['available_bikes']) / df['station_capacity'].clip(lower=1)
        
        return df
    
    def create_lag_features(self, df, lag_hours=[1, 2, 3, 6, 12, 24, 48, 168]):
        """Create lagged features for time series patterns"""
        logger.info(f"Creating lag features: {lag_hours}")
        
        # Sort data properly
        df = df.sort_values(['station_id', 'datetime'])
        
        # Features to lag
        lag_features = ['available_bikes', 'net_flow', 'total_activity', 'is_stockout', 'utilization_rate']
        
        for station_id in tqdm(df['station_id'].unique(), desc="Creating lags"):
            station_mask = df['station_id'] == station_id
            station_data = df.loc[station_mask].copy()
            
            for feature in lag_features:
                for lag in lag_hours:
                    col_name = f'{feature}_lag_{lag}h'
                    df.loc[station_mask, col_name] = station_data[feature].shift(lag)
        
        logger.info(f"Created {len(lag_features) * len(lag_hours)} lag features")
        return df
    
    def create_rolling_features(self, df, windows=[6, 12, 24, 168]):
        """Create rolling window statistics"""
        logger.info(f"Creating rolling features: {windows}")
        
        # Features for rolling statistics
        rolling_features = ['available_bikes', 'net_flow', 'total_activity', 'temperature', 'feels_like']
        
        for station_id in tqdm(df['station_id'].unique(), desc="Creating rolling stats"):
            station_mask = df['station_id'] == station_id
            station_data = df.loc[station_mask].copy()
            
            for feature in rolling_features:
                for window in windows:
                    # Rolling mean
                    col_name = f'{feature}_roll_mean_{window}h'
                    df.loc[station_mask, col_name] = (
                        station_data[feature].rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Rolling std (only for key features)
                    if feature in ['net_flow', 'total_activity']:
                        col_name = f'{feature}_roll_std_{window}h'
                        df.loc[station_mask, col_name] = (
                            station_data[feature].rolling(window=window, min_periods=1).std().fillna(0)
                        )
        
        logger.info(f"Created rolling statistics")
        return df
    
    def create_station_profiles(self, df):
        """Create station-level aggregate features"""
        logger.info("Creating station profiles...")
        
        # Station-level statistics (excluding target period)
        station_stats = df.groupby('station_id').agg({
            'available_bikes': ['mean', 'std'],
            'net_flow': ['mean', 'std'],
            'total_activity': ['mean', 'std'], 
            'station_capacity': 'mean',
            'is_stockout': 'mean',
            'is_nearly_empty': 'mean'
        }).reset_index()
        
        # Flatten column names
        station_stats.columns = ['station_id'] + [
            f'station_{col[0]}_{col[1]}' if col[1] else f'station_{col[0]}'
            for col in station_stats.columns[1:]
        ]
        
        # Merge back
        df = df.merge(station_stats, on='station_id', how='left')
        
        logger.info(f"Added {len(station_stats.columns)-1} station profile features")
        return df
    
    def create_target_variables(self, df):
        """Create prediction targets"""
        logger.info("Creating target variables...")
        
        # Sort data
        df = df.sort_values(['station_id', 'datetime'])
        
        # Primary target: stockout in 2 hours (classification)
        df['target_stockout_2h'] = df.groupby('station_id')['is_stockout'].shift(-2)
        
        # Secondary targets
        df['target_nearly_empty_2h'] = df.groupby('station_id')['is_nearly_empty'].shift(-2)
        df['target_net_flow_2h'] = df.groupby('station_id')['net_flow'].shift(-2)
        df['target_available_bikes_2h'] = df.groupby('station_id')['available_bikes'].shift(-2)
        
        logger.info("Created prediction targets (2-hour ahead)")
        return df
    
    def prepare_final_dataset(self, df):
        """Clean and prepare final dataset"""
        logger.info("Preparing final dataset...")
        
        # Remove rows with missing targets
        before_count = len(df)
        df = df.dropna(subset=['target_stockout_2h'])
        after_count = len(df)
        logger.info(f"Removed {before_count - after_count:,} rows with missing targets")
        
        # Fill remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove unnecessary columns
        drop_cols = ['datetime']  # Keep other columns for potential analysis
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        logger.info(f"Final dataset shape: {df.shape}")
        return df
    
    def create_train_test_split(self, df, test_start_date='2024-11-01'):
        """Split data temporally for training"""
        logger.info(f"Creating temporal split at {test_start_date}")
        
        # Convert string date to datetime.date object for comparison
        if isinstance(test_start_date, str):
            test_start_date = pd.to_datetime(test_start_date).date()
        
        train_df = df[df['date'] < test_start_date].copy()
        test_df = df[df['date'] >= test_start_date].copy()
        
        logger.info(f"Train: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test: {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")
        
        # Check target distribution
        logger.info("Target distribution:")
        logger.info(f"Train stockout rate: {train_df['target_stockout_2h'].mean():.3f}")
        logger.info(f"Test stockout rate: {test_df['target_stockout_2h'].mean():.3f}")
        
        return train_df, test_df
    
    def run_full_pipeline(self):
        """Execute the complete data preparation pipeline"""
        logger.info("🚀 Starting LightGBM data preparation pipeline")
        
        # Step 1: Load base data
        df = self.load_base_data()
        
        # Step 2: Create temporal features
        df = self.create_temporal_features(df)
        gc.collect()
        
        # Step 3: Create target variables (before lag features to avoid data leakage)
        df = self.create_target_variables(df)
        gc.collect()
        
        # Step 4: Create lag features
        df = self.create_lag_features(df)
        gc.collect()
        
        # Step 5: Create rolling features  
        df = self.create_rolling_features(df)
        gc.collect()
        
        # Step 6: Create station profiles
        df = self.create_station_profiles(df)
        gc.collect()
        
        # Step 7: Final cleanup
        df = self.prepare_final_dataset(df)
        
        # Step 8: Train/test split
        train_df, test_df = self.create_train_test_split(df)
        
        # Step 9: Save datasets
        logger.info("Saving datasets...")
        train_df.to_parquet('lightgbm_train_2024.parquet', index=False)
        test_df.to_parquet('lightgbm_test_2024.parquet', index=False)
        df.to_parquet('lightgbm_full_2024.parquet', index=False)
        
        logger.info("✅ Data preparation complete!")
        logger.info(f"Files saved: lightgbm_train_2024.parquet, lightgbm_test_2024.parquet")
        logger.info(f"Features: {len(df.columns)} columns")
        
        return train_df, test_df

if __name__ == "__main__":
    preparer = LightGBMDataPreparer()
    train_df, test_df = preparer.run_full_pipeline()