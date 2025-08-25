"""
Prepare XGBoost training data from PostgreSQL
Generates proper targets and features for net flow prediction
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import calendar
import json
import logging
from dotenv import load_dotenv
import os
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Features MUST match exactly what API expects (110 features)
REQUIRED_FEATURES = [
    'available_bikes', 'station_capacity', 'available_racks',
    'is_stockout', 'is_nearly_empty', 'is_nearly_full',
    'bikes_departed', 'bikes_arrived', 'net_flow',
    'is_weekend', 'avg_trip_duration_min', 'avg_trip_distance_m',
    'temperature', 'humidity', 'precipitation', 'wind_speed', 'feels_like',
    'is_raining', 'is_snowing', 'weather_severity',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month', 'month_sin', 'month_cos',
    'is_morning_rush', 'is_evening_rush', 'is_late_night',
    'total_activity', 'utilization_rate', 'capacity_pressure',
    # Lag features (8 periods)
    'available_bikes_lag_1h', 'available_bikes_lag_2h', 'available_bikes_lag_3h',
    'available_bikes_lag_6h', 'available_bikes_lag_12h', 'available_bikes_lag_24h',
    'available_bikes_lag_48h', 'available_bikes_lag_168h',
    'net_flow_lag_1h', 'net_flow_lag_2h', 'net_flow_lag_3h',
    'net_flow_lag_6h', 'net_flow_lag_12h', 'net_flow_lag_24h',
    'net_flow_lag_48h', 'net_flow_lag_168h',
    'total_activity_lag_1h', 'total_activity_lag_2h', 'total_activity_lag_3h',
    'total_activity_lag_6h', 'total_activity_lag_12h', 'total_activity_lag_24h',
    'total_activity_lag_48h', 'total_activity_lag_168h',
    'is_stockout_lag_1h', 'is_stockout_lag_2h', 'is_stockout_lag_3h',
    'is_stockout_lag_6h', 'is_stockout_lag_12h', 'is_stockout_lag_24h',
    'is_stockout_lag_48h', 'is_stockout_lag_168h',
    'utilization_rate_lag_1h', 'utilization_rate_lag_2h', 'utilization_rate_lag_3h',
    'utilization_rate_lag_6h', 'utilization_rate_lag_12h', 'utilization_rate_lag_24h',
    'utilization_rate_lag_48h', 'utilization_rate_lag_168h',
    # Rolling features
    'available_bikes_roll_mean_6h', 'available_bikes_roll_mean_12h',
    'available_bikes_roll_mean_24h', 'available_bikes_roll_mean_168h',
    'net_flow_roll_mean_6h', 'net_flow_roll_std_6h',
    'net_flow_roll_mean_12h', 'net_flow_roll_std_12h',
    'net_flow_roll_mean_24h', 'net_flow_roll_std_24h',
    'net_flow_roll_mean_168h', 'net_flow_roll_std_168h',
    'total_activity_roll_mean_6h', 'total_activity_roll_std_6h',
    'total_activity_roll_mean_12h', 'total_activity_roll_std_12h',
    'total_activity_roll_mean_24h', 'total_activity_roll_std_24h',
    'total_activity_roll_mean_168h', 'total_activity_roll_std_168h',
    'temperature_roll_mean_6h', 'temperature_roll_mean_12h',
    'temperature_roll_mean_24h', 'temperature_roll_mean_168h',
    'feels_like_roll_mean_6h', 'feels_like_roll_mean_12h',
    'feels_like_roll_mean_24h', 'feels_like_roll_mean_168h',
    # Station profile features
    'station_available_bikes_mean', 'station_available_bikes_std',
    'station_net_flow_mean', 'station_net_flow_std',
    'station_total_activity_mean', 'station_total_activity_std',
    'station_station_capacity_mean', 'station_is_stockout_mean',
    'station_is_nearly_empty_mean'
]

class XGBoostDataPreparer:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        
    def load_base_data(self, start_date='2024-01-01', end_date='2024-12-31'):
        """Load and join all base tables"""
        
        query = text(f"""
        SELECT 
            a.station_id,
            a.date,
            a.hour,
            a.available_bikes,
            a.station_capacity,
            a.available_racks,
            a.is_stockout,
            a.is_nearly_empty,
            a.is_nearly_full,
            COALESCE(s.bikes_arrived, 0) as bikes_arrived,
            COALESCE(s.bikes_departed, 0) as bikes_departed,
            COALESCE(s.net_flow, 0) as net_flow,
            COALESCE(s.avg_trip_duration_min, 15) as avg_trip_duration_min,
            COALESCE(s.avg_trip_distance_m, 2000) as avg_trip_distance_m,
            EXTRACT(DOW FROM a.date) as day_of_week,
            CASE WHEN EXTRACT(DOW FROM a.date) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
            EXTRACT(MONTH FROM a.date) as month,
            w.temperature,
            w.humidity,
            w.precipitation,
            w.wind_speed,
            w.feels_like,
            w.is_raining,
            w.is_snowing,
            w.weather_severity
        FROM bike_availability_hourly a
        LEFT JOIN station_hourly_flow s
            ON a.station_id = s.station_id 
            AND a.date = s.flow_date 
            AND a.hour = s.flow_hour
        LEFT JOIN weather_hourly w
            ON a.date = w.date 
            AND a.hour = w.hour
        WHERE a.date BETWEEN :start_date AND :end_date
        """)
        
        logger.info(f"Loading data from {start_date} to {end_date}...")
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        
        logger.info(f"Loaded {len(df):,} records")
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['day_of_week'].astype(int)
        df['month'] = df['month'].astype(int)
        df['is_weekend'] = df['is_weekend'].astype(int)
        
        return df
    
    def create_target(self, df):
        """Create target: net_flow_2h (bikes at T+2 - bikes at T)"""
        
        df = df.sort_values(['station_id', 'date', 'hour'])
        
        # Shift available_bikes by -2 hours within each station
        df['bikes_2h_later'] = df.groupby('station_id')['available_bikes'].shift(-2)
        
        # Calculate net flow target
        df['target_net_flow_2h'] = df['bikes_2h_later'] - df['available_bikes']
        
        # Remove rows without target (last 2 hours of each day)
        initial_count = len(df)
        df = df.dropna(subset=['target_net_flow_2h'])
        
        logger.info(f"Dropped {initial_count - len(df):,} rows without target")
        logger.info(f"Target stats: mean={df['target_net_flow_2h'].mean():.2f}, "
                   f"std={df['target_net_flow_2h'].std():.2f}, "
                   f"min={df['target_net_flow_2h'].min():.2f}, "
                   f"max={df['target_net_flow_2h'].max():.2f}")
        
        # Check distribution
        near_zero = (df['target_net_flow_2h'].abs() < 2).sum()
        logger.info(f"Near-zero targets (|x| < 2): {near_zero:,} ({near_zero/len(df)*100:.1f}%)")
        
        return df
    
    def add_temporal_features(self, df):
        """Add time-based features"""
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rush hour flags
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_rush'] = df['hour'].isin([18, 19, 20]).astype(int)
        df['is_late_night'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)
        
        return df
    
    def add_derived_features(self, df):
        """Add calculated features"""
        
        df['total_activity'] = df['bikes_arrived'] + df['bikes_departed']
        df['utilization_rate'] = df['available_bikes'] / df['station_capacity'].clip(lower=1)
        df['capacity_pressure'] = 1 - (df['available_racks'] / df['station_capacity'].clip(lower=1))
        
        # Handle stations where bikes > capacity (overflow parking)
        df.loc[df['utilization_rate'] > 1, 'utilization_rate'] = 1.0
        df.loc[df['capacity_pressure'] > 1, 'capacity_pressure'] = 1.0
        
        return df
    
    def add_lag_features(self, df):
        """Add lag features for each station - process by station to save memory"""
        
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
        lag_cols = ['available_bikes', 'net_flow', 'total_activity', 
                    'is_stockout', 'utilization_rate']
        
        df = df.sort_values(['station_id', 'date', 'hour'])
        
        logger.info("Adding lag features...")
        
        # Process each station separately
        station_dfs = []
        unique_stations = df['station_id'].unique()
        
        for i, station_id in enumerate(unique_stations):
            if i % 200 == 0:
                logger.info(f"  Processing lag features for station {i+1}/{len(unique_stations)}...")
            
            station_df = df[df['station_id'] == station_id].copy()
            
            for col in lag_cols:
                for lag in lag_hours:
                    station_df[f'{col}_lag_{lag}h'] = station_df[col].shift(lag)
            
            station_dfs.append(station_df)
            
            # Free memory periodically
            if i % 500 == 0:
                gc.collect()
        
        # Combine all stations
        logger.info("Combining stations after lag features...")
        df = pd.concat(station_dfs, ignore_index=True)
        
        return df
    
    def add_rolling_features(self, df):
        """Add rolling window features - process by station to save memory"""
        
        windows = [6, 12, 24, 168]
        roll_cols = ['available_bikes', 'net_flow', 'total_activity', 
                     'temperature', 'feels_like']
        
        df = df.sort_values(['station_id', 'date', 'hour'])
        
        logger.info("Adding rolling features (this may take a while)...")
        
        # Process each station separately to avoid memory issues
        station_dfs = []
        unique_stations = df['station_id'].unique()
        
        for i, station_id in enumerate(unique_stations):
            if i % 100 == 0:
                logger.info(f"  Processing station {i+1}/{len(unique_stations)}...")
            
            station_df = df[df['station_id'] == station_id].copy()
            
            for col in roll_cols:
                for window in windows:
                    # Mean
                    station_df[f'{col}_roll_mean_{window}h'] = (
                        station_df[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                    )
                    
                    # Std (for some features)
                    if col in ['net_flow', 'total_activity']:
                        station_df[f'{col}_roll_std_{window}h'] = (
                            station_df[col]
                            .rolling(window=window, min_periods=1)
                            .std()
                            .fillna(0)
                        )
            
            station_dfs.append(station_df)
            
            # Free memory periodically
            if i % 500 == 0:
                gc.collect()
        
        # Combine all stations
        logger.info("Combining processed stations...")
        df = pd.concat(station_dfs, ignore_index=True)
        
        return df
    
    def add_station_profile_features(self, df):
        """Add station-level aggregate features"""
        
        logger.info("Adding station profile features...")
        
        station_profiles = df.groupby('station_id').agg({
            'available_bikes': ['mean', 'std'],
            'net_flow': ['mean', 'std'],
            'total_activity': ['mean', 'std'],
            'station_capacity': 'mean',
            'is_stockout': 'mean',
            'is_nearly_empty': 'mean'
        })
        
        station_profiles.columns = ['station_' + '_'.join(col).strip() for col in station_profiles.columns]
        station_profiles = station_profiles.reset_index()
        
        # Fill NaN std with 0
        std_cols = [col for col in station_profiles.columns if '_std' in col]
        station_profiles[std_cols] = station_profiles[std_cols].fillna(0)
        
        df = df.merge(station_profiles, on='station_id', how='left')
        
        return df
    
    def create_sample_weights(self, df):
        """Create weights to focus on high-activity periods"""
        
        # Weight based on absolute net flow magnitude
        abs_net_flow = np.abs(df['target_net_flow_2h'])
        
        # Higher weight for larger changes (logarithmic scale)
        weights = 1 + np.log1p(abs_net_flow)
        
        # Extra weight for stockout scenarios
        weights = np.where(df['is_stockout'] == 1, weights * 1.5, weights)
        
        # Extra weight for nearly empty scenarios
        weights = np.where(df['is_nearly_empty'] == 1, weights * 1.2, weights)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def prepare_data(self):
        """Main pipeline - process month by month"""
        
        all_dfs = []
        
        # Process each month separately to avoid memory issues
        for month in range(1, 13):
            start_date = f'2024-{month:02d}-01'
            if month == 12:
                end_date = '2024-12-31'
            else:
                end_date = f'2024-{month:02d}-{calendar.monthrange(2024, month)[1]}'
            
            logger.info(f"\nProcessing month {month}/12 ({start_date} to {end_date})...")
            
            # Load one month
            df_month = self.load_base_data(start_date, end_date)
            
            if df_month.empty:
                logger.warning(f"No data for month {month}")
                continue
                
            # Create target for this month
            df_month = self.create_target(df_month)
            
            # Add basic features
            df_month = self.add_temporal_features(df_month)
            df_month = self.add_derived_features(df_month)
            
            all_dfs.append(df_month)
            logger.info(f"Month {month}: {len(df_month):,} records after target creation")
            gc.collect()  # Free memory
        
        # Combine all months
        logger.info("\nCombining all months...")
        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total combined: {len(df):,} records")
        
        # Sort for lag features
        logger.info("Sorting data for lag features...")
        df = df.sort_values(['station_id', 'date', 'hour'])
        
        # Add lag/rolling features on combined data
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_station_profile_features(df)
        
        # Fill missing weather data with defaults
        weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 
                       'feels_like', 'is_raining', 'is_snowing', 'weather_severity']
        
        weather_defaults = {
            'temperature': 20.0,
            'humidity': 60.0,
            'precipitation': 0.0,
            'wind_speed': 2.0,
            'feels_like': 20.0,
            'is_raining': 0,
            'is_snowing': 0,
            'weather_severity': 0
        }
        
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(weather_defaults.get(col, 0))
        
        # Drop rows with NaN in critical features (from lag/rolling)
        initial_count = len(df)
        df = df.dropna(subset=REQUIRED_FEATURES)
        logger.info(f"Dropped {initial_count - len(df):,} rows with NaN features")
        
        # Create sample weights
        df['sample_weight'] = self.create_sample_weights(df)
        
        # Split by date
        train_end = pd.Timestamp('2024-10-31')
        val_end = pd.Timestamp('2024-11-30')
        
        train_df = df[df['date'] <= train_end].copy()
        val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
        test_df = df[df['date'] > val_end].copy()
        
        logger.info(f"\nDataset sizes:")
        logger.info(f"Train: {len(train_df):,} samples (up to {train_end.date()})")
        logger.info(f"Val: {len(val_df):,} samples ({(train_end + timedelta(days=1)).date()} to {val_end.date()})")
        logger.info(f"Test: {len(test_df):,} samples (from {(val_end + timedelta(days=1)).date()})")
        
        # Verify all required features are present
        missing_features = set(REQUIRED_FEATURES) - set(train_df.columns)
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Save to parquet
        logger.info("\nSaving datasets...")
        train_df.to_parquet('xgboost_train_2024.parquet', index=False)
        val_df.to_parquet('xgboost_val_2024.parquet', index=False)
        test_df.to_parquet('xgboost_test_2024.parquet', index=False)
        
        # Save feature list for API compatibility
        config = {
            'features': REQUIRED_FEATURES,
            'num_features': len(REQUIRED_FEATURES),
            'target': 'target_net_flow_2h',
            'prepared_date': datetime.now().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'target_stats': {
                'mean': float(df['target_net_flow_2h'].mean()),
                'std': float(df['target_net_flow_2h'].std()),
                'min': float(df['target_net_flow_2h'].min()),
                'max': float(df['target_net_flow_2h'].max())
            }
        }
        
        with open('xgboost_features_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\nFiles saved:")
        logger.info("- xgboost_train_2024.parquet")
        logger.info("- xgboost_val_2024.parquet")
        logger.info("- xgboost_test_2024.parquet")
        logger.info("- xgboost_features_config.json")
        
        # Print sample of target distribution
        logger.info("\nTarget distribution in training set:")
        target_ranges = [
            ('< -10', train_df['target_net_flow_2h'] < -10),
            ('-10 to -5', (train_df['target_net_flow_2h'] >= -10) & (train_df['target_net_flow_2h'] < -5)),
            ('-5 to -2', (train_df['target_net_flow_2h'] >= -5) & (train_df['target_net_flow_2h'] < -2)),
            ('-2 to 2', (train_df['target_net_flow_2h'] >= -2) & (train_df['target_net_flow_2h'] <= 2)),
            ('2 to 5', (train_df['target_net_flow_2h'] > 2) & (train_df['target_net_flow_2h'] <= 5)),
            ('5 to 10', (train_df['target_net_flow_2h'] > 5) & (train_df['target_net_flow_2h'] <= 10)),
            ('> 10', train_df['target_net_flow_2h'] > 10)
        ]
        
        for label, mask in target_ranges:
            count = mask.sum()
            pct = count / len(train_df) * 100
            logger.info(f"  {label:12s}: {count:8,} ({pct:5.1f}%)")
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    logger.info("Starting XGBoost data preparation...")
    preparer = XGBoostDataPreparer()
    
    try:
        train_df, val_df, test_df = preparer.prepare_data()
        logger.info("\n✅ Data preparation complete!")
        logger.info("Upload these files to Google Drive for training:")
        logger.info("  - xgboost_train_2024.parquet")
        logger.info("  - xgboost_val_2024.parquet")
        logger.info("  - xgboost_test_2024.parquet")
        logger.info("  - xgboost_features_config.json")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise