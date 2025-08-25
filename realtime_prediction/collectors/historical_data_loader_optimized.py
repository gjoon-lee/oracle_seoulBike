"""
Optimized Historical Data Loader for Real-time Predictions
Bulk processes all stations at once instead of iterating
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from sqlalchemy import text
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from db_connection import BikeDataDB

logger = logging.getLogger(__name__)

class OptimizedHistoricalDataLoader:
    """Optimized loader for historical data with bulk processing"""
    
    def __init__(self, window_hours: int = 168):
        self.window_hours = window_hours
        self.db = BikeDataDB()
        self.db.connect()
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    def load_combined_history_bulk(self,
                                   station_ids: Optional[List[str]] = None,
                                   hours: int = None) -> pd.DataFrame:
        """Load historical data for all stations in one query"""
        if hours is None:
            hours = self.window_hours
        
        # Check cache first
        cache_key = f"history_{hours}_{len(station_ids) if station_ids else 'all'}"
        if self._cache_time and (datetime.now() - self._cache_time).seconds < self._cache_ttl:
            if cache_key in self._cache:
                logger.info(f"Using cached historical data")
                return self._cache[cache_key]
        
        current_date = datetime.now()
        start_date = current_date - timedelta(hours=hours)
        
        # Build optimized query for all stations at once
        query = text("""
            WITH combined_data AS (
                SELECT 
                    a.station_id,
                    a.date,
                    a.hour,
                    a.available_bikes,
                    a.station_capacity,
                    a.is_stockout,
                    COALESCE(n.bikes_arrived, 0) as bikes_arrived,
                    COALESCE(n.bikes_departed, 0) as bikes_departed,
                    COALESCE(n.net_flow, 0) as net_flow,
                    COALESCE(n.bikes_arrived + n.bikes_departed, 0) as total_activity
                FROM bike_availability_hourly a
                LEFT JOIN station_hourly_flow n
                    ON a.station_id = n.station_id 
                    AND a.date = n.flow_date 
                    AND a.hour = n.flow_hour
                WHERE a.date >= :start_date
                    AND a.date <= :end_date
                    AND (:station_list IS NULL OR a.station_id = ANY(:station_list))
            )
            SELECT * FROM combined_data
            ORDER BY station_id, date DESC, hour DESC
        """)
        
        params = {
            'start_date': start_date.date(),
            'end_date': current_date.date(),
            'station_list': station_ids if station_ids else None
        }
        
        try:
            df = self.db.read_query(query, params)
            
            if not df.empty:
                # Create datetime column
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                               df['hour'].astype(str) + ':00:00')
                
                # Calculate utilization rate
                df['utilization_rate'] = (df['available_bikes'] / df['station_capacity']).fillna(0)
                
                # Cache the result
                self._cache[cache_key] = df
                self._cache_time = datetime.now()
                
                logger.info(f"Loaded {len(df)} historical records for {df['station_id'].nunique()} stations in bulk")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def calculate_lag_features_optimized(self, 
                                        current_data: pd.DataFrame,
                                        lag_hours: List[int] = None) -> pd.DataFrame:
        """Calculate lag features using vectorized operations"""
        if lag_hours is None:
            lag_hours = Config.LAG_HOURS
        
        # Load all historical data at once
        station_ids = current_data['station_id'].unique().tolist()
        history_df = self.load_combined_history_bulk(station_ids, max(lag_hours) + 1)
        
        if history_df.empty:
            logger.warning("No historical data for lag features")
            # Add empty lag columns
            for lag in lag_hours:
                for col in ['available_bikes', 'net_flow', 'total_activity', 
                          'is_stockout', 'utilization_rate']:
                    current_data[f'{col}_lag_{lag}h'] = np.nan
            return current_data
        
        current_time = pd.Timestamp.now()
        
        # Process all lag hours at once using vectorized operations
        for lag in lag_hours:
            lag_time = current_time - pd.Timedelta(hours=lag)
            
            # Get data for this lag hour (with 1-hour tolerance)
            mask = (
                (history_df['datetime'] >= lag_time - pd.Timedelta(hours=1)) &
                (history_df['datetime'] <= lag_time + pd.Timedelta(hours=1))
            )
            lag_data = history_df[mask].copy()
            
            if not lag_data.empty:
                # Group by station and take the most recent value
                lag_grouped = lag_data.sort_values('datetime').groupby('station_id').last()
                
                # Merge with current data
                for col in ['available_bikes', 'net_flow', 'total_activity', 
                           'is_stockout', 'utilization_rate']:
                    if col in lag_grouped.columns:
                        lag_col_name = f'{col}_lag_{lag}h'
                        current_data = current_data.merge(
                            lag_grouped[[col]].rename(columns={col: lag_col_name}),
                            left_on='station_id',
                            right_index=True,
                            how='left'
                        )
                    else:
                        current_data[f'{col}_lag_{lag}h'] = np.nan
            else:
                # No data for this lag hour
                for col in ['available_bikes', 'net_flow', 'total_activity', 
                           'is_stockout', 'utilization_rate']:
                    current_data[f'{col}_lag_{lag}h'] = np.nan
        
        logger.info(f"Calculated lag features for {len(current_data)} stations using vectorized operations")
        return current_data
    
    def calculate_rolling_features_optimized(self,
                                            current_data: pd.DataFrame,
                                            rolling_windows: List[int] = None) -> pd.DataFrame:
        """Calculate rolling features using vectorized operations"""
        if rolling_windows is None:
            rolling_windows = Config.ROLLING_WINDOWS
        
        # Load historical data
        station_ids = current_data['station_id'].unique().tolist()
        history_df = self.load_combined_history_bulk(station_ids, max(rolling_windows))
        
        if history_df.empty:
            logger.warning("No historical data for rolling features")
            # Add empty rolling columns
            for window in rolling_windows:
                for metric in ['available_bikes', 'net_flow', 'total_activity']:
                    current_data[f'{metric}_roll_mean_{window}h'] = np.nan
                    current_data[f'{metric}_roll_std_{window}h'] = np.nan
            return current_data
        
        current_time = pd.Timestamp.now()
        
        # Calculate rolling statistics for each window
        for window in rolling_windows:
            window_start = current_time - pd.Timedelta(hours=window)
            
            # Filter data for this window
            window_data = history_df[history_df['datetime'] >= window_start].copy()
            
            if not window_data.empty:
                # Group by station and calculate statistics
                grouped = window_data.groupby('station_id')
                
                # Calculate means
                for metric in ['available_bikes', 'net_flow', 'total_activity']:
                    if metric in window_data.columns:
                        mean_col = f'{metric}_roll_mean_{window}h'
                        std_col = f'{metric}_roll_std_{window}h'
                        
                        stats = grouped[metric].agg(['mean', 'std'])
                        
                        # Merge with current data
                        current_data = current_data.merge(
                            stats[['mean']].rename(columns={'mean': mean_col}),
                            left_on='station_id',
                            right_index=True,
                            how='left'
                        )
                        
                        current_data = current_data.merge(
                            stats[['std']].rename(columns={'std': std_col}),
                            left_on='station_id',
                            right_index=True,
                            how='left'
                        )
            else:
                # No data for this window
                for metric in ['available_bikes', 'net_flow', 'total_activity']:
                    current_data[f'{metric}_roll_mean_{window}h'] = np.nan
                    current_data[f'{metric}_roll_std_{window}h'] = np.nan
        
        logger.info(f"Calculated rolling features for {len(current_data)} stations using vectorized operations")
        return current_data