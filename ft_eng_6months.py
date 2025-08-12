"""
Feature Engineering for 6 Months of Seoul Bike Data
Run this LOCALLY to prepare data for Google Colab training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from db_connection import BikeDataDB
import gc
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NetFlowFeatureEngineer:
    """
    Optimized feature engineering for NetFlow prediction
    Handles 6 months of data efficiently
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def create_features_batch(self, start_date, end_date, output_prefix='bike_features'):
        """
        Main function to create features from PostgreSQL data
        """
        logger.info(f"🚀 Starting feature engineering for {start_date} to {end_date}")
        
        # Step 1: Get list of active stations (to filter out low-activity ones)
        active_stations = self._get_active_stations(start_date, end_date)
        logger.info(f"Found {len(active_stations)} active stations")
        
        # Step 2: Process data in monthly chunks to manage memory
        date_ranges = self._generate_date_ranges(start_date, end_date, freq='M')
        
        all_features = []
        
        for i, (chunk_start, chunk_end) in enumerate(date_ranges):
            logger.info(f"\n📅 Processing chunk {i+1}/{len(date_ranges)}: {chunk_start} to {chunk_end}")
            
            # Load chunk from PostgreSQL
            chunk_df = self._load_data_chunk(chunk_start, chunk_end, active_stations)
            
            if len(chunk_df) == 0:
                logger.warning(f"No data found for {chunk_start} to {chunk_end}")
                continue
            
            # Create features for this chunk
            chunk_features = self._create_features(chunk_df)
            
            # Save intermediate result
            chunk_file = f'{output_prefix}_chunk_{i+1}.parquet'
            chunk_features.to_parquet(chunk_file, compression='snappy')
            logger.info(f"Saved {len(chunk_features):,} rows to {chunk_file}")
            
            all_features.append(chunk_file)
            
            # Clear memory
            del chunk_df, chunk_features
            gc.collect()
        
        # Step 3: Combine all chunks and create train/test split
        logger.info("\n🔄 Combining all features...")
        final_df = self._combine_and_split(all_features, output_prefix)
        
        return final_df
    
    def _get_active_stations(self, start_date, end_date, min_daily_activity=10):
        """Get list of stations with sufficient activity"""
        query = f"""
        SELECT 
            station_id,
            AVG(bikes_departed + bikes_arrived) as avg_daily_activity,
            COUNT(DISTINCT flow_date) as active_days
        FROM station_hourly_flow
        WHERE flow_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY station_id
        HAVING AVG(bikes_departed + bikes_arrived) >= {min_daily_activity}
        AND COUNT(DISTINCT flow_date) >= 30
        ORDER BY avg_daily_activity DESC
        """
        
        result = self.db.read_query(query)
        return result['station_id'].tolist()
    
    def _generate_date_ranges(self, start_date, end_date, freq='M'):
        """Generate date ranges for chunking"""
        ranges = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        while current < end:
            if freq == 'M':
                chunk_end = current + pd.offsets.MonthEnd(0)
            else:  # Weekly
                chunk_end = current + timedelta(days=6)
                
            if chunk_end > end:
                chunk_end = end
                
            ranges.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            
            current = chunk_end + timedelta(days=1)
            
        return ranges
    
    def _load_data_chunk(self, start_date, end_date, station_list):
        """Load a chunk of data from PostgreSQL"""
        # Convert station list to SQL format
        stations_sql = "','".join(station_list)
        
        query = f"""
        SELECT 
            station_id,
            flow_date,
            flow_hour,
            day_of_week,
            is_weekend,
            season,
            bikes_departed,
            bikes_arrived,
            bikes_arrived - bikes_departed as net_flow,
            bikes_departed + bikes_arrived as total_activity,
            avg_trip_duration_min,
            avg_trip_distance_m
        FROM station_hourly_flow
        WHERE flow_date BETWEEN '{start_date}' AND '{end_date}'
        AND station_id IN ('{stations_sql}')
        ORDER BY station_id, flow_date, flow_hour
        """
        
        return self.db.read_query(query)
    
    def _create_features(self, df):
        """Create all features for NetFlow prediction"""
        logger.info("Creating features...")
        
        # Sort data
        df = df.sort_values(['station_id', 'flow_date', 'flow_hour'])
        
        feature_dfs = []
        
        # Process each station separately (more memory efficient)
        for station_id in tqdm(df['station_id'].unique(), desc="Processing stations"):
            station_df = df[df['station_id'] == station_id].copy()
            
            # Skip if too little data
            if len(station_df) < 48:  # Less than 2 days
                continue
            
            # 1. Lag features (most important for time series)
            lag_hours = [1, 2, 3, 6, 12, 24, 48, 24*7]  # Including weekly lag
            
            for lag in lag_hours:
                station_df[f'net_flow_lag_{lag}h'] = station_df['net_flow'].shift(lag)
                station_df[f'activity_lag_{lag}h'] = station_df['total_activity'].shift(lag)
            
            # 2. Rolling statistics
            windows = [6, 12, 24, 24*7]
            
            for window in windows:
                # Net flow rolling stats
                station_df[f'net_flow_roll_mean_{window}h'] = (
                    station_df['net_flow'].rolling(window, min_periods=1).mean()
                )
                station_df[f'net_flow_roll_std_{window}h'] = (
                    station_df['net_flow'].rolling(window, min_periods=1).std().fillna(0)
                )
                
                # Activity rolling stats
                station_df[f'activity_roll_mean_{window}h'] = (
                    station_df['total_activity'].rolling(window, min_periods=1).mean()
                )
            
            # 3. Time-based features
            station_df['hour_sin'] = np.sin(2 * np.pi * station_df['flow_hour'] / 24)
            station_df['hour_cos'] = np.cos(2 * np.pi * station_df['flow_hour'] / 24)
            station_df['dow_sin'] = np.sin(2 * np.pi * station_df['day_of_week'] / 7)
            station_df['dow_cos'] = np.cos(2 * np.pi * station_df['day_of_week'] / 7)
            
            # 4. Rush hour and time indicators
            station_df['is_morning_rush'] = (
                station_df['flow_hour'].isin([7, 8, 9]) & 
                ~station_df['is_weekend']
            ).astype(int)
            
            station_df['is_evening_rush'] = (
                station_df['flow_hour'].isin([18, 19, 20]) & 
                ~station_df['is_weekend']
            ).astype(int)
            
            station_df['is_late_night'] = (
                (station_df['flow_hour'] >= 23) | 
                (station_df['flow_hour'] <= 5)
            ).astype(int)
            
            # 5. Station-specific features (calculated over entire period)
            station_avg_net_flow = station_df['net_flow'].mean()
            station_std_net_flow = station_df['net_flow'].std()
            station_avg_activity = station_df['total_activity'].mean()
            
            station_df['station_avg_net_flow'] = station_avg_net_flow
            station_df['station_std_net_flow'] = station_std_net_flow
            station_df['station_avg_activity'] = station_avg_activity
            
            # 6. Trend features
            station_df['net_flow_trend_3h'] = (
                station_df['net_flow_lag_1h'] - station_df['net_flow_lag_3h']
            )
            station_df['net_flow_trend_6h'] = (
                station_df['net_flow_lag_1h'] - station_df['net_flow_lag_6h']
            )
            
            # 7. Same time yesterday/last week
            station_df['net_flow_yesterday'] = station_df['net_flow'].shift(24)
            station_df['net_flow_last_week'] = station_df['net_flow'].shift(24 * 7)
            
            # 8. Create target variable (what we want to predict)
            station_df['target_net_flow_2h'] = station_df['net_flow'].shift(-2)
            
            feature_dfs.append(station_df)
        
        # Combine all stations
        features_df = pd.concat(feature_dfs, ignore_index=True)
        
        logger.info(f"Created features for {len(features_df):,} samples")
        
        return features_df
    
    def _combine_and_split(self, chunk_files, output_prefix):
        """Combine chunks and create train/test split"""
        
        # Load all chunks
        all_data = []
        for file in chunk_files:
            logger.info(f"Loading {file}")
            chunk = pd.read_parquet(file)
            all_data.append(chunk)
        
        # Combine
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data shape: {combined.shape}")
        
        # Remove rows with NaN in target
        combined = combined.dropna(subset=['target_net_flow_2h'])
        
        # Remove rows with too many NaN features
        feature_cols = [col for col in combined.columns 
                       if col not in ['station_id', 'flow_date', 'flow_hour', 
                                     'target_net_flow_2h', 'net_flow', 'total_activity',
                                     'bikes_departed', 'bikes_arrived']]
        
        nan_threshold = len(feature_cols) * 0.3  # Allow up to 30% NaN
        combined = combined[combined[feature_cols].isna().sum(axis=1) < nan_threshold]
        
        # Sort by date for proper time series split
        combined = combined.sort_values(['flow_date', 'flow_hour'])
        
        # Create time-based train/test split (last 2 weeks for test)
        split_date = combined['flow_date'].max() - timedelta(days=14)
        
        train_df = combined[combined['flow_date'] <= split_date]
        test_df = combined[combined['flow_date'] > split_date]
        
        logger.info(f"\nTrain: {len(train_df):,} samples ({train_df['flow_date'].min()} to {train_df['flow_date'].max()})")
        logger.info(f"Test: {len(test_df):,} samples ({test_df['flow_date'].min()} to {test_df['flow_date'].max()})")
        
        # Save final datasets
        train_file = f'{output_prefix}_train.csv'
        test_file = f'{output_prefix}_test.csv'
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Also save as parquet for faster loading in Colab
        train_df.to_parquet(f'{output_prefix}_train.parquet', compression='snappy')
        test_df.to_parquet(f'{output_prefix}_test.parquet', compression='snappy')
        
        logger.info(f"\n✅ Saved files:")
        logger.info(f"  - {train_file} ({len(train_df):,} rows)")
        logger.info(f"  - {test_file} ({len(test_df):,} rows)")
        logger.info(f"  - Parquet versions for faster loading")
        
        # Print feature info
        logger.info(f"\n📊 Feature Summary:")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info("Feature categories:")
        logger.info(f"  - Lag features: {len([f for f in feature_cols if 'lag' in f])}")
        logger.info(f"  - Rolling features: {len([f for f in feature_cols if 'roll' in f])}")
        logger.info(f"  - Time features: {len([f for f in feature_cols if any(t in f for t in ['hour', 'dow', 'rush'])])}")
        logger.info(f"  - Station features: {len([f for f in feature_cols if 'station' in f])}")
        
        # Clean up temporary files
        import os
        for file in chunk_files:
            os.remove(file)
            
        return combined


# Main execution
if __name__ == "__main__":
    # Connect to PostgreSQL
    logger.info("🔌 Connecting to PostgreSQL...")
    db = BikeDataDB()
    db.connect()
    
    # Create feature engineer
    engineer = NetFlowFeatureEngineer(db)
    
    # Process 6 months of data
    # Adjust dates based on what you have
    features = engineer.create_features_batch(
        start_date='2024-01-01',
        end_date='2024-12-31',
        output_prefix='netflow_features_6m'
    )
    
    logger.info("\n✅ Feature engineering complete!")
    logger.info("\n📤 Next steps:")
    logger.info("1. Upload these files to Google Drive:")
    logger.info("   - netflow_features_6m_train.parquet")
    logger.info("   - netflow_features_6m_test.parquet")
    logger.info("2. Run the training script in Google Colab")
    logger.info("3. Download the trained model")
    logger.info("4. Use with real-time API for predictions!")