"""
Fast Historical Data Loader using vectorized operations
Processes all stations in bulk instead of one-by-one
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

class FastHistoricalDataLoader:
    """Optimized historical data loader with bulk processing"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    def load_combined_history_bulk(self, station_ids: List[str], hours: int = 168) -> pd.DataFrame:
        """Load all historical data in ONE query"""
        
        # Check cache first
        cache_key = f"history_{hours}_{len(station_ids)}"
        if self._cache_time and (datetime.now() - self._cache_time).seconds < self._cache_ttl:
            if cache_key in self._cache:
                logger.info(f"Using cached historical data")
                return self._cache[cache_key]
        
        current_date = datetime.now()
        start_date = current_date - timedelta(hours=hours)
        
        # Build ONE optimized query for ALL stations
        # Format station IDs properly for SQL IN clause
        if station_ids:
            station_list = "','".join(station_ids)
            station_filter = f"AND a.station_id IN ('{station_list}')"
        else:
            station_filter = ""  # No filter if no station IDs
        
        # Query will be built below
        
        try:
            logger.info(f"Loading {hours}h history for {len(station_ids)} stations in bulk...")
            # BikeDataDB.read_query only takes the query, not params separately
            # So we need to format the query with the dates
            query_str = f"""
            WITH combined_data AS (
                SELECT 
                    a.station_id,
                    a.date,
                    a.hour,
                    a.available_bikes,
                    a.station_capacity,
                    COALESCE(a.available_bikes / NULLIF(a.station_capacity, 0), 0) as utilization_rate,
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
                WHERE a.date >= '{start_date.date()}'
                    AND a.date <= '{current_date.date()}'
                    {station_filter}
            )
            SELECT * FROM combined_data
            ORDER BY station_id, date DESC, hour DESC
            """
            df = self.db.read_query(text(query_str))
            
            if not df.empty:
                # Create datetime column
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                               df['hour'].astype(str).str.zfill(2) + ':00:00')
                
                # Cache the result
                self._cache[cache_key] = df
                self._cache_time = datetime.now()
                
                logger.info(f"Loaded {len(df)} records for {df['station_id'].nunique()} stations in ONE query")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading bulk history: {e}")
            # Fallback to empty dataframe
            return pd.DataFrame()
    
    def calculate_lag_features_vectorized(self, 
                                         current_data: pd.DataFrame,
                                         lag_hours: List[int] = None) -> pd.DataFrame:
        """Calculate lag features using pandas vectorized operations - SUPER FAST"""
        
        if lag_hours is None:
            lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
        
        logger.info(f"Calculating lag features for {len(current_data)} stations...")
        
        # Get all station IDs
        station_ids = current_data['station_id'].unique().tolist()
        
        # Load ALL historical data at once
        history_df = self.load_combined_history_bulk(station_ids, max(lag_hours) + 24)
        
        if history_df.empty:
            logger.warning("No historical data available for lag features")
            # Add empty lag columns
            for lag in lag_hours:
                for col in ['available_bikes', 'net_flow', 'total_activity', 
                          'is_stockout', 'utilization_rate']:
                    current_data[f'{col}_lag_{lag}h'] = np.nan
            return current_data
        
        # VECTORIZED APPROACH - Process all stations at once!
        current_time = pd.Timestamp.now()
        
        # Set index for faster lookups
        history_df = history_df.set_index(['station_id', 'datetime'])
        
        # Process each lag hour
        for lag in lag_hours:
            lag_time = current_time - pd.Timedelta(hours=lag)
            
            logger.debug(f"Processing lag {lag}h...")
            
            # Create lag column names
            lag_cols = {
                'available_bikes': f'available_bikes_lag_{lag}h',
                'net_flow': f'net_flow_lag_{lag}h',
                'total_activity': f'total_activity_lag_{lag}h',
                'is_stockout': f'is_stockout_lag_{lag}h',
                'utilization_rate': f'utilization_rate_lag_{lag}h'
            }
            
            # For each station, find the closest historical record to lag_time
            for station_id in station_ids:
                try:
                    # Get this station's history
                    station_mask = history_df.index.get_level_values('station_id') == station_id
                    station_history = history_df[station_mask]
                    
                    if not station_history.empty:
                        # Find closest time to lag_time
                        time_diffs = abs(station_history.index.get_level_values('datetime') - lag_time)
                        
                        # Get record within 1 hour of target
                        within_hour = time_diffs <= pd.Timedelta(hours=1)
                        if within_hour.any():
                            closest_idx = time_diffs[within_hour].argmin()
                            closest_record = station_history.iloc[closest_idx]
                            
                            # Update current_data for this station
                            station_mask_current = current_data['station_id'] == station_id
                            for col, lag_col in lag_cols.items():
                                if col in closest_record.index:
                                    current_data.loc[station_mask_current, lag_col] = closest_record[col]
                
                except Exception as e:
                    logger.debug(f"Error processing lag for station {station_id}: {e}")
                    continue
        
        # Fill any remaining NaN values
        for lag in lag_hours:
            for col in ['available_bikes', 'net_flow', 'total_activity', 
                       'is_stockout', 'utilization_rate']:
                lag_col = f'{col}_lag_{lag}h'
                if lag_col not in current_data.columns:
                    current_data[lag_col] = np.nan
        
        logger.info(f"Lag features calculated successfully for {len(current_data)} stations")
        return current_data
    
    def calculate_rolling_features_vectorized(self,
                                             current_data: pd.DataFrame,
                                             rolling_windows: List[int] = None) -> pd.DataFrame:
        """Calculate rolling features using vectorized operations"""
        
        if rolling_windows is None:
            rolling_windows = [6, 12, 24, 168]
        
        logger.info(f"Calculating rolling features for {len(current_data)} stations...")
        
        # Get all station IDs
        station_ids = current_data['station_id'].unique().tolist()
        
        # Load historical data
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
        
        # Group by station for efficient processing
        grouped = history_df.groupby('station_id')
        
        # Calculate rolling statistics for each window
        for window in rolling_windows:
            window_start = current_time - pd.Timedelta(hours=window)
            
            logger.debug(f"Processing rolling window {window}h...")
            
            # Calculate statistics for each station
            for station_id in station_ids:
                try:
                    # Get station data within window
                    if station_id in grouped.groups:
                        station_data = grouped.get_group(station_id)
                        window_data = station_data[station_data['datetime'] >= window_start]
                        
                        if not window_data.empty:
                            # Calculate statistics
                            station_mask = current_data['station_id'] == station_id
                            
                            for metric in ['available_bikes', 'net_flow', 'total_activity']:
                                if metric in window_data.columns:
                                    mean_val = window_data[metric].mean()
                                    std_val = window_data[metric].std()
                                    
                                    current_data.loc[station_mask, f'{metric}_roll_mean_{window}h'] = mean_val
                                    current_data.loc[station_mask, f'{metric}_roll_std_{window}h'] = std_val
                
                except Exception as e:
                    logger.debug(f"Error processing rolling for station {station_id}: {e}")
                    continue
        
        # Fill any remaining NaN values for missing columns
        for window in rolling_windows:
            for metric in ['available_bikes', 'net_flow', 'total_activity']:
                mean_col = f'{metric}_roll_mean_{window}h'
                std_col = f'{metric}_roll_std_{window}h'
                
                if mean_col not in current_data.columns:
                    current_data[mean_col] = np.nan
                if std_col not in current_data.columns:
                    current_data[std_col] = np.nan
        
        logger.info(f"Rolling features calculated successfully for {len(current_data)} stations")
        return current_data