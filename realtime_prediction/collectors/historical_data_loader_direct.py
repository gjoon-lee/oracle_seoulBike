"""
Direct Historical Data Loader using targeted SQL queries
No massive data loading - just fetch exactly what we need
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from sqlalchemy import text
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from db_connection import BikeDataDB

logger = logging.getLogger(__name__)

class DirectHistoricalDataLoader:
    """Efficient historical data loader using direct SQL queries"""
    
    def __init__(self):
        self.db = BikeDataDB()
        self.db.connect()
        logger.info("DirectHistoricalDataLoader initialized")
    
    def calculate_lag_features_direct(self, 
                                     current_data: pd.DataFrame,
                                     lag_hours: List[int] = None) -> pd.DataFrame:
        """Calculate lag features using direct SQL queries - SUPER FAST"""
        
        if lag_hours is None:
            lag_hours = [1, 6, 24]  # Default to essential lags only
        
        logger.info(f"Calculating lag features for {len(current_data)} stations using DIRECT SQL...")
        
        # Get current time
        current_time = datetime.now()
        
        # SHIFT BACK 7 DAYS to use last week's same day as proxy
        # Today is Aug 25 (Mon) -> Use Aug 18 (Mon) data
        reference_time = current_time - timedelta(days=7)
        logger.info(f"Using reference date: {reference_time.date()} (7 days ago) as proxy for {current_time.date()}")
        
        # Build SQL query for exact time points
        time_conditions = []
        lag_mapping = {}
        
        for lag in lag_hours:
            # Calculate lag from REFERENCE time, not current time
            lag_time = reference_time - timedelta(hours=lag)
            lag_date = lag_time.date()
            lag_hour = lag_time.hour
            
            time_conditions.append(f"(a.date = '{lag_date}' AND a.hour = {lag_hour})")
            lag_mapping[f"{lag_date}_{lag_hour}"] = lag
        
        # Build query for FLOW data only (we have August 2025 flow data)
        # For availability, we'll use current values as smart fill-in
        query = f"""
        SELECT 
            station_id,
            flow_date as date,
            flow_hour as hour,
            COALESCE(net_flow, 0) as net_flow,
            COALESCE(bikes_arrived + bikes_departed, 0) as total_activity
        FROM station_hourly_flow
        WHERE ({' OR '.join(time_conditions).replace('a.date', 'flow_date').replace('a.hour', 'flow_hour')})
        ORDER BY station_id, flow_date, flow_hour
        """
        
        try:
            # Execute the query
            logger.info("Executing direct SQL query for lag features...")
            lag_df = self.db.read_query(text(query))
            
            if lag_df.empty:
                logger.warning("No historical flow data found, using smart fill-in")
                # Use current values as smart fill-in for availability
                for lag in lag_hours:
                    # Use current availability as proxy for lag
                    current_data[f'available_bikes_lag_{lag}h'] = current_data['available_bikes']
                    current_data[f'is_stockout_lag_{lag}h'] = current_data['is_stockout']
                    current_data[f'utilization_rate_lag_{lag}h'] = current_data['utilization_rate']
                    # Flow data gets 0 if not available
                    current_data[f'net_flow_lag_{lag}h'] = 0
                    current_data[f'total_activity_lag_{lag}h'] = 0
                return current_data
            
            # Create a lookup dictionary for fast access
            logger.info(f"Processing {len(lag_df)} lag records...")
            lag_lookup = {}
            for _, row in lag_df.iterrows():
                key = f"{row['station_id']}_{row['date']}_{row['hour']}"
                lag_lookup[key] = row
            
            # Apply lag features to current data
            for lag in lag_hours:
                # Use REFERENCE time for lag calculations
                lag_time = reference_time - timedelta(hours=lag)
                lag_date = lag_time.date()
                lag_hour = lag_time.hour
                
                # For FLOW data - use historical values from August 2025
                for col in ['net_flow', 'total_activity']:
                    col_name = f'{col}_lag_{lag}h'
                    current_data[col_name] = current_data['station_id'].apply(
                        lambda sid: lag_lookup.get(f"{sid}_{lag_date}_{lag_hour}", {}).get(col, 0)
                    )
                
                # For AVAILABILITY data - use current values as smart fill-in
                # (since we don't have historical availability for August 2025)
                current_data[f'available_bikes_lag_{lag}h'] = current_data['available_bikes']
                current_data[f'is_stockout_lag_{lag}h'] = current_data['is_stockout']
                current_data[f'utilization_rate_lag_{lag}h'] = current_data['utilization_rate']
            
            logger.info(f"Lag features calculated successfully in SECONDS!")
            return current_data
            
        except Exception as e:
            logger.error(f"Error calculating lag features: {e}")
            # Use smart fill-in on error
            for lag in lag_hours:
                # Use current values for availability
                current_data[f'available_bikes_lag_{lag}h'] = current_data['available_bikes']
                current_data[f'is_stockout_lag_{lag}h'] = current_data['is_stockout']
                current_data[f'utilization_rate_lag_{lag}h'] = current_data['utilization_rate']
                # Zero for flow if error
                current_data[f'net_flow_lag_{lag}h'] = 0
                current_data[f'total_activity_lag_{lag}h'] = 0
            return current_data
    
    def calculate_rolling_features_direct(self,
                                         current_data: pd.DataFrame,
                                         rolling_windows: List[int] = None) -> pd.DataFrame:
        """Calculate rolling features using optimized SQL"""
        
        if rolling_windows is None:
            rolling_windows = [6, 24]  # Only essential windows
        
        logger.info(f"Calculating rolling features for {len(current_data)} stations...")
        
        current_time = datetime.now()
        # Use same reference time shift (7 days back)
        reference_time = current_time - timedelta(days=7)
        logger.info(f"Using reference date {reference_time.date()} for rolling features")
        
        # Build query for each window
        for window in rolling_windows:
            window_start = reference_time - timedelta(hours=window)
            
            query = f"""
            SELECT 
                a.station_id,
                AVG(a.available_bikes) as available_bikes_mean,
                STDDEV(a.available_bikes) as available_bikes_std,
                AVG(COALESCE(n.net_flow, 0)) as net_flow_mean,
                STDDEV(COALESCE(n.net_flow, 0)) as net_flow_std,
                AVG(COALESCE(n.bikes_arrived + n.bikes_departed, 0)) as total_activity_mean,
                STDDEV(COALESCE(n.bikes_arrived + n.bikes_departed, 0)) as total_activity_std
            FROM bike_availability_hourly a
            LEFT JOIN station_hourly_flow n
                ON a.station_id = n.station_id 
                AND a.date = n.flow_date 
                AND a.hour = n.flow_hour
            WHERE a.date >= '{window_start.date()}'
                AND (a.date > '{window_start.date()}' OR a.hour >= {window_start.hour})
            GROUP BY a.station_id
            """
            
            try:
                roll_df = self.db.read_query(text(query))
                
                if not roll_df.empty:
                    # Merge with current data
                    roll_df = roll_df.set_index('station_id')
                    
                    for metric in ['available_bikes', 'net_flow', 'total_activity']:
                        mean_col = f'{metric}_roll_mean_{window}h'
                        std_col = f'{metric}_roll_std_{window}h'
                        
                        current_data[mean_col] = current_data['station_id'].map(
                            roll_df[f'{metric}_mean'].to_dict()
                        ).fillna(0)
                        
                        current_data[std_col] = current_data['station_id'].map(
                            roll_df[f'{metric}_std'].to_dict()
                        ).fillna(0)
                else:
                    # Add empty columns
                    for metric in ['available_bikes', 'net_flow', 'total_activity']:
                        current_data[f'{metric}_roll_mean_{window}h'] = 0
                        current_data[f'{metric}_roll_std_{window}h'] = 0
                        
            except Exception as e:
                logger.error(f"Error calculating rolling features for {window}h window: {e}")
                # Add empty columns on error
                for metric in ['available_bikes', 'net_flow', 'total_activity']:
                    current_data[f'{metric}_roll_mean_{window}h'] = 0
                    current_data[f'{metric}_roll_std_{window}h'] = 0
        
        logger.info("Rolling features calculated successfully!")
        return current_data