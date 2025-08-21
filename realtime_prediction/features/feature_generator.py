"""
Feature generator for LightGBM stockout prediction model
Generates all 110 features required by the model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from collectors.realtime_bike_collector import RealtimeBikeCollector
from collectors.realtime_weather_collector import RealtimeWeatherCollector
from collectors.historical_data_loader import HistoricalDataLoader

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Generates all 110 features required by the LightGBM model"""
    
    def __init__(self):
        self.bike_collector = RealtimeBikeCollector()
        self.weather_collector = RealtimeWeatherCollector()
        self.history_loader = HistoricalDataLoader()
        
        # Load station profiles
        self.station_profiles = self.load_station_profiles()
        
        # Feature list from model config
        self.required_features = self.load_required_features()
        
    def load_required_features(self) -> List[str]:
        """Load the list of required features from model config"""
        try:
            with open(Config.MODEL_THRESHOLDS_PATH, 'r') as f:
                config = json.load(f)
                return config['model_info']['features']
        except Exception as e:
            logger.error(f"Error loading model features: {e}")
            return []
    
    def load_station_profiles(self) -> Dict:
        """Load pre-calculated station profiles"""
        profile_path = Config.PROJECT_ROOT / "cache" / "station_profiles.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading station profiles: {e}")
        
        logger.warning("Station profiles not found, will use defaults")
        return {}
    
    def generate_features(self, realtime_only: bool = False) -> pd.DataFrame:
        """Generate all features for prediction"""
        
        # 1. Get real-time bike availability
        logger.info("Fetching real-time bike availability...")
        bike_df = self.bike_collector.fetch_all_stations()
        
        if bike_df.empty:
            logger.error("No bike data available")
            return pd.DataFrame()
        
        # 2. Get current weather
        logger.info("Fetching current weather...")
        weather_data = self.weather_collector.fetch_current_weather()
        
        # Add weather features to all stations
        for key, value in weather_data.items():
            if key != 'timestamp' and key != 'source':
                bike_df[key] = value
        
        # 3. Add temporal features
        logger.info("Adding temporal features...")
        bike_df = self.add_temporal_features(bike_df)
        
        # 4. Add derived features
        logger.info("Adding derived features...")
        bike_df = self.add_derived_features(bike_df)
        
        if not realtime_only:
            # 5. Add lag features from historical data
            logger.info("Adding lag features...")
            bike_df = self.history_loader.calculate_lag_features(bike_df, Config.LAG_HOURS)
            
            # 6. Add rolling features
            logger.info("Adding rolling features...")
            bike_df = self.history_loader.calculate_rolling_features(bike_df, Config.ROLLING_WINDOWS)
            
            # 7. Add station profile features
            logger.info("Adding station profile features...")
            bike_df = self.add_station_profile_features(bike_df)
        
        # 8. Ensure all required features are present
        bike_df = self.ensure_all_features(bike_df)
        
        logger.info(f"Generated {len(bike_df.columns)} features for {len(bike_df)} stations")
        
        return bike_df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal encoding features"""
        now = datetime.now()
        
        # Basic temporal
        df['hour'] = now.hour
        df['day_of_week'] = now.weekday()
        df['month'] = now.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time period flags
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_rush'] = df['hour'].isin([17, 18, 19]).astype(int)
        df['is_late_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        
        # Weekend is already added by bike collector
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from current state"""
        
        # Net flow estimation (from availability change if historical not available)
        if 'net_flow' not in df.columns:
            df['net_flow'] = 0  # Will be updated by net flow estimator
        
        if 'bikes_arrived' not in df.columns:
            df['bikes_arrived'] = 0
        
        if 'bikes_departed' not in df.columns:
            df['bikes_departed'] = 0
        
        # Total activity
        df['total_activity'] = df['bikes_arrived'] + df['bikes_departed']
        
        # Trip features (use defaults if not available)
        if 'avg_trip_duration_min' not in df.columns:
            df['avg_trip_duration_min'] = 15.0  # Default 15 minutes
        
        if 'avg_trip_distance_m' not in df.columns:
            df['avg_trip_distance_m'] = 2000.0  # Default 2km
        
        return df
    
    def add_station_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add station-specific historical profile features"""
        
        profile_features = [
            'station_available_bikes_mean',
            'station_available_bikes_std',
            'station_net_flow_mean',
            'station_net_flow_std',
            'station_total_activity_mean',
            'station_total_activity_std',
            'station_station_capacity_mean',
            'station_is_stockout_mean',
            'station_is_nearly_empty_mean'
        ]
        
        # Initialize with defaults
        for feature in profile_features:
            df[feature] = np.nan
        
        # Apply station-specific profiles
        for idx, row in df.iterrows():
            station_id = row['station_id']
            
            if station_id in self.station_profiles:
                profile = self.station_profiles[station_id]
                for feature in profile_features:
                    if feature in profile:
                        df.at[idx, feature] = profile[feature]
        
        # Fill missing with global averages
        df['station_available_bikes_mean'].fillna(10.0, inplace=True)
        df['station_available_bikes_std'].fillna(5.0, inplace=True)
        df['station_net_flow_mean'].fillna(0.0, inplace=True)
        df['station_net_flow_std'].fillna(2.0, inplace=True)
        df['station_total_activity_mean'].fillna(20.0, inplace=True)
        df['station_total_activity_std'].fillna(10.0, inplace=True)
        df['station_station_capacity_mean'].fillna(df['station_capacity'].mean(), inplace=True)
        df['station_is_stockout_mean'].fillna(0.1, inplace=True)
        df['station_is_nearly_empty_mean'].fillna(0.2, inplace=True)
        
        return df
    
    def ensure_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present"""
        
        # Check for missing features
        missing_features = []
        for feature in self.required_features:
            if feature not in df.columns:
                missing_features.append(feature)
                # Add with default value
                df[feature] = np.nan
        
        if missing_features:
            logger.warning(f"Added {len(missing_features)} missing features with NaN: {missing_features[:10]}...")
        
        # Fill NaN values with appropriate defaults
        df = self.fill_missing_values(df)
        
        # Select only required features in correct order
        df = df[self.required_features]
        
        return df
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults"""
        
        # Lag features - forward fill then use current value
        lag_cols = [col for col in df.columns if '_lag_' in col]
        for col in lag_cols:
            base_col = col.split('_lag_')[0]
            if base_col in df.columns:
                df[col].fillna(df[base_col], inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        
        # Rolling features - use current value
        roll_cols = [col for col in df.columns if '_roll_' in col]
        for col in roll_cols:
            if 'mean' in col:
                base_col = col.split('_roll_')[0]
                if base_col in df.columns:
                    df[col].fillna(df[base_col], inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
            elif 'std' in col:
                df[col].fillna(0, inplace=True)
        
        # Weather features
        df['temperature'].fillna(15.0, inplace=True)
        df['humidity'].fillna(60.0, inplace=True)
        df['precipitation'].fillna(0.0, inplace=True)
        df['wind_speed'].fillna(2.0, inplace=True)
        df['feels_like'].fillna(15.0, inplace=True)
        
        # Binary features
        binary_cols = [col for col in df.columns if col.startswith('is_')]
        for col in binary_cols:
            df[col].fillna(0, inplace=True)
        
        # Other features
        df['net_flow'].fillna(0, inplace=True)
        df['bikes_arrived'].fillna(0, inplace=True)
        df['bikes_departed'].fillna(0, inplace=True)
        df['total_activity'].fillna(0, inplace=True)
        df['avg_trip_duration_min'].fillna(15.0, inplace=True)
        df['avg_trip_distance_m'].fillna(2000.0, inplace=True)
        
        return df
    
    def generate_features_for_station(self, station_id: str) -> Optional[pd.Series]:
        """Generate features for a specific station"""
        
        # Get data for specific station
        station_data = self.bike_collector.fetch_station(station_id)
        
        if not station_data:
            logger.error(f"No data for station {station_id}")
            return None
        
        # Convert to DataFrame for processing
        df = pd.DataFrame([station_data])
        
        # Generate all features
        df = self.generate_features()
        
        if not df.empty:
            return df.iloc[0]
        
        return None


if __name__ == "__main__":
    # Test feature generation
    logging.basicConfig(level=logging.INFO)
    
    generator = FeatureGenerator()
    
    print("Generating features for all stations...")
    features_df = generator.generate_features(realtime_only=True)
    
    if not features_df.empty:
        print(f"\nGenerated features shape: {features_df.shape}")
        print(f"Sample features for first station:")
        print(features_df.iloc[0].head(20))
        
        # Check for required features
        print(f"\nRequired features: {len(generator.required_features)}")
        print(f"Generated features: {len(features_df.columns)}")
        
        missing = set(generator.required_features) - set(features_df.columns)
        if missing:
            print(f"Missing features: {missing}")
        else:
            print("All required features present!")
    else:
        print("Failed to generate features")