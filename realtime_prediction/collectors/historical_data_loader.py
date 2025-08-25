"""
Historical data loader from PostgreSQL database
Loads 14 days of historical data for lag feature calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from sqlalchemy import create_engine, text
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from db_connection import BikeDataDB

logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    """Loads historical data from PostgreSQL for feature engineering"""
    
    def __init__(self):
        self.db = BikeDataDB()
        self.db.connect()
        self.window_hours = Config.HISTORICAL_WINDOW_HOURS  # 168 hours (7 days)
        self.cache = {}
        
    def load_availability_history(self, 
                                 station_ids: Optional[List[str]] = None,
                                 hours: int = None) -> pd.DataFrame:
        """Load historical availability data with intelligent fallback"""
        if hours is None:
            hours = self.window_hours
        
        current_date = datetime.now()
        fallback_strategies = [
            # Strategy 1: Last 7 days (most recent)
            {
                'name': 'Last 7 days',
                'start_date': current_date - timedelta(hours=hours),
                'end_date': current_date,
                'quality': 1.0
            },
            # Strategy 2: 2 weeks ago, same day
            {
                'name': '2 weeks ago',
                'start_date': current_date - timedelta(days=14, hours=hours),
                'end_date': current_date - timedelta(days=14),
                'quality': 0.85
            },
            # Strategy 3: 4 weeks ago, same day
            {
                'name': '4 weeks ago',
                'start_date': current_date - timedelta(days=28, hours=hours),
                'end_date': current_date - timedelta(days=28),
                'quality': 0.7
            },
            # Strategy 4: Same period last year
            {
                'name': 'Last year same period',
                'start_date': current_date.replace(year=current_date.year-1) - timedelta(hours=hours),
                'end_date': current_date.replace(year=current_date.year-1),
                'quality': 0.5
            }
        ]
        
        df = pd.DataFrame()
        used_strategy = None
        
        for strategy in fallback_strategies:
            try:
                query_base = f"""
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
                    WHERE date >= '{strategy['start_date'].date()}'
                    AND date <= '{strategy['end_date'].date()}'
                """
                
                if station_ids:
                    station_list = "', '".join(station_ids)
                    query_base += f" AND station_id IN ('{station_list}')"
                
                query_base += " ORDER BY station_id, date, hour"
                
                df = self.db.read_query(text(query_base))
                
                if not df.empty:
                    used_strategy = strategy
                    logger.info(f"Using fallback strategy: {strategy['name']} "
                              f"(quality: {strategy['quality']:.1%})")
                    logger.info(f"Date range: {strategy['start_date'].date()} to {strategy['end_date'].date()}")
                    break
                    
            except Exception as e:
                logger.debug(f"Strategy '{strategy['name']}' failed: {e}")
                continue
        
        if not df.empty:
            # Create datetime column
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                           df['hour'].astype(str) + ':00:00')
            
            # Adjust datetime to current period if using historical data
            if used_strategy and used_strategy['quality'] < 1.0:
                # Calculate time shift needed
                time_shift = current_date - used_strategy['end_date']
                df['datetime'] = df['datetime'] + time_shift
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                logger.info(f"Adjusted timestamps by {time_shift.days} days to match current period")
            
            # Calculate derived features
            df['utilization_rate'] = df['available_bikes'] / df['station_capacity']
            df['utilization_rate'] = df['utilization_rate'].fillna(0)
            df['capacity_pressure'] = 1 - (df['available_racks'] / df['station_capacity'])
            df['capacity_pressure'] = df['capacity_pressure'].fillna(1)
            
            # Add data quality indicator
            df['data_quality'] = used_strategy['quality'] if used_strategy else 1.0
            
            logger.info(f"Loaded {len(df)} availability records for {df['station_id'].nunique()} stations")
        else:
            logger.warning("No availability history found with any fallback strategy")
            # Return empty dataframe with expected columns
            df = pd.DataFrame(columns=['station_id', 'date', 'hour', 'datetime',
                                      'available_bikes', 'station_capacity', 'available_racks',
                                      'is_stockout', 'is_nearly_empty', 'is_nearly_full',
                                      'utilization_rate', 'capacity_pressure', 'data_quality'])
        
        return df
    
    def load_netflow_history(self,
                           station_ids: Optional[List[str]] = None,
                           hours: int = None) -> pd.DataFrame:
        """Load historical net flow data with intelligent fallback"""
        if hours is None:
            hours = self.window_hours
        
        current_date = datetime.now()
        fallback_strategies = [
            # Strategy 1: Last 7 days (most recent)
            {
                'name': 'Last 7 days',
                'start_date': current_date - timedelta(hours=hours),
                'end_date': current_date,
                'quality': 1.0
            },
            # Strategy 2: 2 weeks ago, same day
            {
                'name': '2 weeks ago',
                'start_date': current_date - timedelta(days=14, hours=hours),
                'end_date': current_date - timedelta(days=14),
                'quality': 0.85
            },
            # Strategy 3: 4 weeks ago, same day
            {
                'name': '4 weeks ago',
                'start_date': current_date - timedelta(days=28, hours=hours),
                'end_date': current_date - timedelta(days=28),
                'quality': 0.7
            },
            # Strategy 4: Same period last year
            {
                'name': 'Last year same period',
                'start_date': current_date.replace(year=current_date.year-1) - timedelta(hours=hours),
                'end_date': current_date.replace(year=current_date.year-1),
                'quality': 0.5
            }
        ]
        
        df = pd.DataFrame()
        used_strategy = None
        
        for strategy in fallback_strategies:
            try:
                query_base = f"""
                    SELECT 
                        station_id,
                        flow_date as date,
                        flow_hour as hour,
                        bikes_departed,
                        bikes_arrived,
                        net_flow,
                        day_of_week,
                        is_weekend,
                        avg_trip_duration_min,
                        avg_trip_distance_m
                    FROM station_hourly_flow
                    WHERE flow_date >= '{strategy['start_date'].date()}'
                    AND flow_date <= '{strategy['end_date'].date()}'
                """
                
                if station_ids:
                    station_list = "', '".join(station_ids)
                    query_base += f" AND station_id IN ('{station_list}')"
                
                query_base += " ORDER BY station_id, flow_date, flow_hour"
                
                df = self.db.read_query(text(query_base))
                
                if not df.empty:
                    used_strategy = strategy
                    logger.info(f"Using fallback strategy: {strategy['name']} "
                              f"(quality: {strategy['quality']:.1%})")
                    logger.info(f"Date range: {strategy['start_date'].date()} to {strategy['end_date'].date()}")
                    break
                    
            except Exception as e:
                logger.debug(f"Strategy '{strategy['name']}' failed: {e}")
                continue
        
        if not df.empty:
            # Create datetime column
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                           df['hour'].astype(str) + ':00:00')
            
            # Adjust datetime to current period if using historical data
            if used_strategy and used_strategy['quality'] < 1.0:
                # Calculate time shift needed
                time_shift = current_date - used_strategy['end_date']
                df['datetime'] = df['datetime'] + time_shift
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                logger.info(f"Adjusted timestamps by {time_shift.days} days to match current period")
            
            # Calculate total activity
            df['total_activity'] = df['bikes_departed'] + df['bikes_arrived']
            
            # Add data quality indicator
            df['data_quality'] = used_strategy['quality'] if used_strategy else 1.0
            
            logger.info(f"Loaded {len(df)} netflow records for {df['station_id'].nunique()} stations")
        else:
            logger.warning("No netflow history found with any fallback strategy")
            # Return empty dataframe with expected columns
            df = pd.DataFrame(columns=['station_id', 'date', 'hour', 'datetime',
                                      'bikes_departed', 'bikes_arrived', 'net_flow',
                                      'total_activity', 'data_quality'])
        
        return df
    
    def load_weather_history(self, hours: int = None) -> pd.DataFrame:
        """Load historical weather data with intelligent fallback"""
        if hours is None:
            hours = self.window_hours
        
        current_date = datetime.now()
        fallback_strategies = [
            # Strategy 1: Last 7 days (most recent)
            {
                'name': 'Last 7 days',
                'start_date': current_date - timedelta(hours=hours),
                'end_date': current_date,
                'quality': 1.0
            },
            # Strategy 2: 2 weeks ago
            {
                'name': '2 weeks ago',
                'start_date': current_date - timedelta(days=14, hours=hours),
                'end_date': current_date - timedelta(days=14),
                'quality': 0.8
            },
            # Strategy 3: Same period last year (for seasonal patterns)
            {
                'name': 'Last year same period',
                'start_date': current_date.replace(year=current_date.year-1) - timedelta(hours=hours),
                'end_date': current_date.replace(year=current_date.year-1),
                'quality': 0.6
            }
        ]
        
        df = pd.DataFrame()
        used_strategy = None
        
        for strategy in fallback_strategies:
            try:
                query = text(f"""
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
                    WHERE date >= '{strategy['start_date'].date()}'
                    AND date <= '{strategy['end_date'].date()}'
                    ORDER BY date, hour
                """)
                
                df = self.db.read_query(query)
                
                if not df.empty:
                    used_strategy = strategy
                    logger.info(f"Using weather fallback: {strategy['name']} "
                              f"(quality: {strategy['quality']:.1%})")
                    break
                    
            except Exception as e:
                logger.debug(f"Weather strategy '{strategy['name']}' failed: {e}")
                continue
        
        if not df.empty:
            # Create datetime column
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                           df['hour'].astype(str) + ':00:00')
            
            # Adjust datetime to current period if using historical data
            if used_strategy and used_strategy['quality'] < 1.0:
                time_shift = current_date - used_strategy['end_date']
                df['datetime'] = df['datetime'] + time_shift
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                logger.info(f"Adjusted weather timestamps by {time_shift.days} days")
            
            logger.info(f"Loaded {len(df)} weather records")
        else:
            logger.warning("No weather history found with any fallback strategy")
            # Return empty dataframe with expected columns
            df = pd.DataFrame(columns=['date', 'hour', 'datetime', 'temperature',
                                      'humidity', 'precipitation', 'wind_speed',
                                      'feels_like', 'is_raining', 'is_snowing',
                                      'weather_severity'])
        
        return df
    
    def load_combined_history(self,
                            station_ids: Optional[List[str]] = None,
                            hours: int = None) -> pd.DataFrame:
        """Load all historical data combined"""
        if hours is None:
            hours = self.window_hours
        
        # Load individual datasets
        availability_df = self.load_availability_history(station_ids, hours)
        netflow_df = self.load_netflow_history(station_ids, hours)
        weather_df = self.load_weather_history(hours)
        
        if availability_df.empty:
            logger.warning("No historical data available")
            return pd.DataFrame()
        
        # Start with availability as base
        combined_df = availability_df.copy()
        
        # Merge netflow data if available
        if not netflow_df.empty:
            netflow_cols = ['station_id', 'datetime', 'bikes_departed', 'bikes_arrived', 
                          'net_flow', 'total_activity', 'avg_trip_duration_min', 'avg_trip_distance_m']
            combined_df = pd.merge(
                combined_df,
                netflow_df[netflow_cols],
                on=['station_id', 'datetime'],
                how='left'
            )
            
            # Fill missing netflow values
            combined_df['bikes_departed'] = combined_df['bikes_departed'].fillna(0)
            combined_df['bikes_arrived'] = combined_df['bikes_arrived'].fillna(0)
            combined_df['net_flow'] = combined_df['net_flow'].fillna(0)
            combined_df['total_activity'] = combined_df['total_activity'].fillna(0)
        else:
            # Create empty netflow columns
            combined_df['bikes_departed'] = 0
            combined_df['bikes_arrived'] = 0
            combined_df['net_flow'] = 0
            combined_df['total_activity'] = 0
        
        # Merge weather data if available
        if not weather_df.empty:
            weather_cols = ['datetime', 'temperature', 'humidity', 'precipitation', 
                          'wind_speed', 'feels_like', 'is_raining', 'is_snowing', 'weather_severity']
            combined_df = pd.merge(
                combined_df,
                weather_df[weather_cols],
                on='datetime',
                how='left'
            )
            
            # Fill missing weather values with interpolation or defaults
            combined_df['temperature'] = combined_df['temperature'].interpolate().fillna(15)
            combined_df['humidity'] = combined_df['humidity'].interpolate().fillna(60)
            combined_df['precipitation'] = combined_df['precipitation'].fillna(0)
            combined_df['wind_speed'] = combined_df['wind_speed'].interpolate().fillna(2)
            combined_df['feels_like'] = combined_df['feels_like'].interpolate().fillna(15)
            combined_df['is_raining'] = combined_df['is_raining'].fillna(0)
            combined_df['is_snowing'] = combined_df['is_snowing'].fillna(0)
            combined_df['weather_severity'] = combined_df['weather_severity'].fillna(0)
        
        # Sort by station and time
        combined_df = combined_df.sort_values(['station_id', 'datetime'])
        
        logger.info(f"Combined history: {len(combined_df)} records, {combined_df['station_id'].nunique()} stations")
        
        return combined_df
    
    def get_station_history(self, station_id: str, hours: int = None) -> pd.DataFrame:
        """Get history for a specific station"""
        return self.load_combined_history([station_id], hours)
    
    def calculate_lag_features(self, 
                              current_data: pd.DataFrame,
                              lag_hours: List[int] = None) -> pd.DataFrame:
        """Calculate lag features for current data using historical data - OPTIMIZED"""
        if lag_hours is None:
            lag_hours = Config.LAG_HOURS
        
        # Load historical data for ALL stations at once
        station_ids = current_data['station_id'].unique().tolist()
        history_df = self.load_combined_history(station_ids, max(lag_hours) + 1)
        
        if history_df.empty:
            logger.warning("No historical data for lag features")
            # Return current data with null lag features
            for lag in lag_hours:
                for col in ['available_bikes', 'net_flow', 'total_activity', 
                          'is_stockout', 'utilization_rate']:
                    current_data[f'{col}_lag_{lag}h'] = np.nan
            return current_data
        
        # Process lag features for each station
        result_dfs = []
        current_time = pd.Timestamp.now()
        
        for _, row in current_data.iterrows():
            station_id = row['station_id']
            
            # Get station history
            station_history = history_df[history_df['station_id'] == station_id].copy()
            
            if station_history.empty:
                result_dfs.append(row.to_frame().T)
                continue
            
            # Calculate lag features
            lag_features = {}
            for lag in lag_hours:
                lag_time = current_time - timedelta(hours=lag)
                
                # Find closest historical record
                time_diff = abs(station_history['datetime'] - lag_time)
                if len(time_diff) > 0:
                    closest_idx = time_diff.idxmin()
                    
                    # Only use if within 1 hour of target time
                    if time_diff.loc[closest_idx] <= timedelta(hours=1):
                        lag_record = station_history.loc[closest_idx]
                        
                        lag_features[f'available_bikes_lag_{lag}h'] = lag_record['available_bikes']
                        lag_features[f'net_flow_lag_{lag}h'] = lag_record['net_flow']
                        lag_features[f'total_activity_lag_{lag}h'] = lag_record['total_activity']
                        lag_features[f'is_stockout_lag_{lag}h'] = lag_record['is_stockout']
                        lag_features[f'utilization_rate_lag_{lag}h'] = lag_record['utilization_rate']
            
            # Add lag features to current row
            for key, value in lag_features.items():
                row[key] = value
            
            result_dfs.append(row.to_frame().T)
        
        # Combine all rows
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        return result_df
    
    def calculate_rolling_features(self,
                                  current_data: pd.DataFrame,
                                  rolling_windows: List[int] = None) -> pd.DataFrame:
        """Calculate rolling statistics features"""
        if rolling_windows is None:
            rolling_windows = Config.ROLLING_WINDOWS
        
        # Load historical data
        station_ids = current_data['station_id'].unique().tolist()
        history_df = self.load_combined_history(station_ids, max(rolling_windows))
        
        if history_df.empty:
            logger.warning("No historical data for rolling features")
            # Return current data with null rolling features
            for window in rolling_windows:
                for metric in ['available_bikes', 'net_flow', 'total_activity', 
                             'temperature', 'feels_like']:
                    current_data[f'{metric}_roll_mean_{window}h'] = np.nan
                for metric in ['net_flow', 'total_activity']:
                    current_data[f'{metric}_roll_std_{window}h'] = np.nan
            return current_data
        
        # Calculate rolling features for each station
        result_dfs = []
        
        for _, row in current_data.iterrows():
            station_id = row['station_id']
            current_time = pd.Timestamp.now()
            
            # Get station history
            station_history = history_df[history_df['station_id'] == station_id].copy()
            
            if station_history.empty:
                result_dfs.append(row.to_frame().T)
                continue
            
            # Calculate rolling features
            rolling_features = {}
            for window in rolling_windows:
                cutoff_time = current_time - timedelta(hours=window)
                window_data = station_history[station_history['datetime'] > cutoff_time]
                
                if not window_data.empty:
                    # Mean features
                    rolling_features[f'available_bikes_roll_mean_{window}h'] = window_data['available_bikes'].mean()
                    rolling_features[f'net_flow_roll_mean_{window}h'] = window_data['net_flow'].mean()
                    rolling_features[f'total_activity_roll_mean_{window}h'] = window_data['total_activity'].mean()
                    rolling_features[f'temperature_roll_mean_{window}h'] = window_data['temperature'].mean()
                    rolling_features[f'feels_like_roll_mean_{window}h'] = window_data['feels_like'].mean()
                    
                    # Std features
                    rolling_features[f'net_flow_roll_std_{window}h'] = window_data['net_flow'].std()
                    rolling_features[f'total_activity_roll_std_{window}h'] = window_data['total_activity'].std()
            
            # Add rolling features to current row
            for key, value in rolling_features.items():
                row[key] = value
            
            result_dfs.append(row.to_frame().T)
        
        # Combine all rows
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        return result_df


if __name__ == "__main__":
    # Test the historical data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = HistoricalDataLoader()
    
    # Test loading different data types
    print("Testing availability history...")
    availability = loader.load_availability_history(hours=24)
    print(f"Loaded {len(availability)} availability records")
    
    print("\nTesting netflow history...")
    netflow = loader.load_netflow_history(hours=24)
    print(f"Loaded {len(netflow)} netflow records")
    
    print("\nTesting weather history...")
    weather = loader.load_weather_history(hours=24)
    print(f"Loaded {len(weather)} weather records")
    
    print("\nTesting combined history...")
    combined = loader.load_combined_history(hours=24)
    if not combined.empty:
        print(f"Combined data shape: {combined.shape}")
        print(f"Columns: {combined.columns.tolist()}")