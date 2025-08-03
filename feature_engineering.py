"""
Feature Engineering Pipeline for Seoul Bike Demand Prediction
Creates ML-ready features from raw hourly flow data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from db_connection import BikeDataDB
import logging
from tqdm import tqdm

class BikeFeatureEngineer:
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def create_lag_features(self, df, lags=[1, 2, 3, 6, 12, 24, 48]):
        """Create time-lagged features"""
        # Sort by time
        df = df.sort_values(['station_id', 'flow_date', 'flow_hour'])
        
        # For each station separately
        lag_features = []
        
        for station_id in tqdm(df['station_id'].unique(), desc="Creating lag features"):
            station_df = df[df['station_id'] == station_id].copy()
            
            # Create lags
            for lag in lags:
                station_df[f'demand_lag_{lag}h'] = station_df['total_demand'].shift(lag)
                station_df[f'net_flow_lag_{lag}h'] = station_df['net_flow'].shift(lag)
            
            # Rolling statistics
            for window in [6, 12, 24]:
                station_df[f'demand_rolling_mean_{window}h'] = (
                    station_df['total_demand'].rolling(window, min_periods=1).mean()
                )
                station_df[f'demand_rolling_std_{window}h'] = (
                    station_df['total_demand'].rolling(window, min_periods=1).std()
                )
            
            # Same hour yesterday and last week
            station_df['demand_same_hour_yesterday'] = station_df['total_demand'].shift(24)
            station_df['demand_same_hour_last_week'] = station_df['total_demand'].shift(24 * 7)
            
            # Trend features
            station_df['demand_trend_3h'] = (
                station_df['total_demand'].shift(1) - station_df['total_demand'].shift(3)
            )
            station_df['demand_trend_6h'] = (
                station_df['total_demand'].shift(1) - station_df['total_demand'].shift(6)
            )
            
            lag_features.append(station_df)
        
        return pd.concat(lag_features, ignore_index=True)
    
    def create_cyclical_features(self, df):
        """Convert hour and day to cyclical features"""
        # Hour of day - sine/cosine encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['flow_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['flow_hour'] / 24)
        
        # Day of week - sine/cosine encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (if you have multiple months)
        df['month'] = pd.to_datetime(df['flow_date']).dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_station_features(self, df):
        """Add station-specific features from the fixed profiles"""
        # Load station profiles
        station_profiles = pd.read_csv('station_profiles_fixed.csv')
        
        # Select important features
        station_features = [
            'station_id', 'avg_total_activity', 'morning_dep', 'morning_arr',
            'evening_dep', 'evening_arr', 'weekend_activity', 'weekday_activity',
            'station_type'
        ]
        
        # Merge
        df = df.merge(
            station_profiles[station_features],
            on='station_id',
            how='left'
        )
        
        # One-hot encode station type
        station_type_dummies = pd.get_dummies(df['station_type'], prefix='station_type')
        df = pd.concat([df, station_type_dummies], axis=1)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        # Rush hour indicators
        df['is_morning_rush'] = (
            (df['flow_hour'].between(7, 9)) & (~df['is_weekend'])
        ).astype(int)
        
        df['is_evening_rush'] = (
            (df['flow_hour'].between(18, 20)) & (~df['is_weekend'])
        ).astype(int)
        
        # Late night indicator
        df['is_late_night'] = (
            (df['flow_hour'] >= 23) | (df['flow_hour'] <= 5)
        ).astype(int)
        
        # Interactions
        df['weekend_hour_interaction'] = df['is_weekend'].astype(int) * df['flow_hour']
        df['rush_hour_demand'] = df['is_morning_rush'] * df['total_demand']
        
        # Station type interactions
        if 'station_type_Residential' in df.columns:
            df['residential_morning_rush'] = (
                df['station_type_Residential'] * df['is_morning_rush']
            )
            df['office_evening_rush'] = (
                df.get('station_type_Office', 0) * df['is_evening_rush']
            )
        
        return df
    
    def create_weather_features(self, df):
        """Placeholder for weather features"""
        # TODO: Add weather data when available
        # For now, create dummy weather features
        self.logger.info("Weather features not yet implemented - using placeholders")
        
        # These would come from weather API
        df['temperature'] = 20  # Celsius
        df['precipitation'] = 0  # mm
        df['humidity'] = 60  # %
        df['wind_speed'] = 5  # km/h
        
        # Derived weather features
        df['is_raining'] = (df['precipitation'] > 0).astype(int)
        df['is_extreme_temp'] = ((df['temperature'] < 5) | (df['temperature'] > 30)).astype(int)
        
        return df
    
    def create_target_variables(self, df, horizons=[1, 2, 3]):
        """Create target variables for different prediction horizons"""
        df = df.sort_values(['station_id', 'flow_date', 'flow_hour'])
        
        targets = []
        
        for station_id in tqdm(df['station_id'].unique(), desc="Creating targets"):
            station_df = df[df['station_id'] == station_id].copy()
            
            for h in horizons:
                station_df[f'target_demand_{h}h'] = station_df['total_demand'].shift(-h)
                station_df[f'target_net_flow_{h}h'] = station_df['net_flow'].shift(-h)
            
            targets.append(station_df)
        
        return pd.concat(targets, ignore_index=True)
    
    def prepare_ml_dataset(self, start_date=None, end_date=None, min_station_activity=10):
        """Prepare complete ML dataset"""
        
        self.logger.info("Preparing ML dataset...")
        
        # Get base data
        query = """
            SELECT 
                station_id,
                flow_date,
                flow_hour,
                day_of_week,
                is_weekend,
                season,
                bikes_departed,
                bikes_arrived,
                net_flow,
                avg_trip_duration_min,
                avg_trip_distance_m
            FROM station_hourly_flow
            WHERE 1=1
        """
        
        if start_date:
            query += f" AND flow_date >= '{start_date}'"
        if end_date:
            query += f" AND flow_date <= '{end_date}'"
            
        query += " ORDER BY station_id, flow_date, flow_hour"
        
        df = self.db.read_query(query)
        
        # Create total demand feature
        df['total_demand'] = df['bikes_departed'] + df['bikes_arrived']
        
        # Filter low-activity stations
        station_activity = df.groupby('station_id')['total_demand'].mean()
        active_stations = station_activity[station_activity >= min_station_activity].index
        df = df[df['station_id'].isin(active_stations)]
        
        self.logger.info(f"Processing {len(active_stations)} active stations...")
        
        # Apply feature engineering
        df = self.create_cyclical_features(df)
        df = self.create_station_features(df)
        df = self.create_interaction_features(df)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df)
        df = self.create_target_variables(df)
        
        # Drop rows with NaN in critical features
        critical_features = [
            'demand_lag_1h', 'demand_lag_2h', 'demand_lag_3h',
            'target_demand_2h'  # Our main target
        ]
        
        df_clean = df.dropna(subset=critical_features)
        
        self.logger.info(f"Created {len(df_clean)} training examples")
        self.logger.info(f"Features: {len(df_clean.columns)} columns")
        
        # Split features and identify types
        self._identify_feature_columns(df_clean)
        
        return df_clean
    
    def _identify_feature_columns(self, df):
        """Identify different types of features"""
        self.feature_cols = {
            'temporal': [
                'flow_hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos',
                'dow_sin', 'dow_cos', 'is_morning_rush', 'is_evening_rush'
            ],
            'lag': [col for col in df.columns if 'lag_' in col],
            'rolling': [col for col in df.columns if 'rolling_' in col],
            'station': [
                'avg_total_activity', 'morning_dep', 'morning_arr',
                'evening_dep', 'evening_arr', 'weekend_activity', 'weekday_activity'
            ] + [col for col in df.columns if 'station_type_' in col],
            'weather': [
                'temperature', 'precipitation', 'humidity', 'wind_speed',
                'is_raining', 'is_extreme_temp'
            ],
            'interaction': [
                'weekend_hour_interaction', 'rush_hour_demand',
                'residential_morning_rush', 'office_evening_rush'
            ],
            'target': [col for col in df.columns if 'target_' in col]
        }
        
        # All features except targets and identifiers
        self.ml_features = []
        for feature_type in ['temporal', 'lag', 'rolling', 'station', 'weather', 'interaction']:
            self.ml_features.extend([f for f in self.feature_cols[feature_type] if f in df.columns])
        
        self.logger.info(f"Identified {len(self.ml_features)} ML features")
        
    def get_train_test_split(self, df, test_days=1):
        """Time-based train/test split"""
        # Sort by date
        df = df.sort_values(['flow_date', 'flow_hour'])
        
        # Split by date
        split_date = df['flow_date'].max() - timedelta(days=test_days)
        
        train_df = df[df['flow_date'] <= split_date]
        test_df = df[df['flow_date'] > split_date]
        
        self.logger.info(f"Train: {len(train_df)} examples ({train_df['flow_date'].min()} to {train_df['flow_date'].max()})")
        self.logger.info(f"Test: {len(test_df)} examples ({test_df['flow_date'].min()} to {test_df['flow_date'].max()})")
        
        return train_df, test_df


# Example usage
if __name__ == "__main__":
    # Connect to database
    db = BikeDataDB()
    db.connect()
    
    # Initialize feature engineer
    engineer = BikeFeatureEngineer(db)
    
    # Create features
    print("🔧 Creating features for ML...")
    ml_data = engineer.prepare_ml_dataset()
    
    # Save features
    ml_data.to_csv('bike_ml_features_v2.csv', index=False)
    print(f"\n✅ Saved {len(ml_data)} examples to 'bike_ml_features_v2.csv'")
    
    # Get train/test split
    train_df, test_df = engineer.get_train_test_split(ml_data, test_days=1)
    
    # Save splits
    train_df.to_csv('bike_features_train.csv', index=False)
    test_df.to_csv('bike_features_test.csv', index=False)
    
    print("\n📊 Feature Summary:")
    print(f"- Temporal features: {len(engineer.feature_cols['temporal'])}")
    print(f"- Lag features: {len(engineer.feature_cols['lag'])}")
    print(f"- Station features: {len(engineer.feature_cols['station'])}")
    print(f"- Total ML features: {len(engineer.ml_features)}")
    
    # Show feature importance preview
    print("\n🎯 Key features created:")
    print("- demand_lag_1h, demand_lag_2h (most important!)")
    print("- hour_sin/cos (cyclical time)")
    print("- station type indicators")
    print("- rush hour flags")
    print("- rolling averages")
    print("\nReady for XGBoost training!")