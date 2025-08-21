"""
Real-time bike availability collector from Seoul Open API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

logger = logging.getLogger(__name__)

class RealtimeBikeCollector:
    """Fetches real-time bike availability from Seoul Open API"""
    
    def __init__(self):
        self.api_key = Config.BIKE_API_KEY
        self.base_url = Config.BIKE_API_BASE_URL
        self.service = Config.BIKE_API_SERVICE
        self.max_records = Config.BIKE_API_MAX_RECORDS
        
    def _build_url(self, start_idx: int, end_idx: int) -> str:
        """Build API URL for given index range"""
        return f"{self.base_url}/{self.api_key}/json/{self.service}/{start_idx}/{end_idx}/"
    
    def _parse_station_data(self, station: Dict) -> Dict:
        """Parse single station data from API response"""
        try:
            # Extract station ID and ensure ST- prefix
            station_id = str(station.get('stationId', ''))
            if not station_id.startswith('ST-'):
                # Handle different formats (e.g., "ST-4" or just "4")
                if station_id.isdigit():
                    station_id = f"ST-{int(station_id):03d}"
                else:
                    station_id = f"ST-{station_id}"
            
            # Calculate availability metrics
            total_racks = int(station.get('rackTotCnt', 0))
            available_bikes = int(station.get('parkingBikeTotCnt', 0))
            available_racks = total_racks - available_bikes
            
            # Calculate status flags
            is_stockout = 1 if available_bikes <= 2 else 0
            is_nearly_empty = 1 if available_bikes <= 5 else 0
            is_nearly_full = 1 if available_racks <= 5 else 0
            
            # Calculate utilization metrics
            utilization_rate = available_bikes / total_racks if total_racks > 0 else 0
            capacity_pressure = 1 - (available_racks / total_racks) if total_racks > 0 else 1
            
            return {
                'station_id': station_id,
                'station_name': station.get('stationName', ''),
                'latitude': float(station.get('stationLatitude', 0)),
                'longitude': float(station.get('stationLongitude', 0)),
                'available_bikes': available_bikes,
                'station_capacity': total_racks,
                'available_racks': available_racks,
                'is_stockout': is_stockout,
                'is_nearly_empty': is_nearly_empty,
                'is_nearly_full': is_nearly_full,
                'utilization_rate': utilization_rate,
                'capacity_pressure': capacity_pressure,
                'shared_bikes': int(station.get('shared', 0)),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error parsing station data: {e}")
            return None
    
    def fetch_all_stations(self) -> pd.DataFrame:
        """Fetch data for all stations with pagination"""
        all_stations = []
        start_idx = 1
        total_count = None
        
        while True:
            end_idx = min(start_idx + self.max_records - 1, total_count or float('inf'))
            
            try:
                url = self._build_url(start_idx, int(end_idx))
                logger.info(f"Fetching stations {start_idx} to {end_idx}")
                
                response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if 'rentBikeStatus' not in data:
                    logger.error(f"Invalid API response structure: {data}")
                    break
                
                bike_data = data['rentBikeStatus']
                
                # Get total count on first request
                if total_count is None:
                    total_count = int(bike_data.get('list_total_count', 0))
                    logger.info(f"Total stations to fetch: {total_count}")
                
                # Check for API-level errors
                result_code = bike_data.get('RESULT', {}).get('CODE', '')
                if result_code != 'INFO-000':
                    logger.error(f"API error: {bike_data.get('RESULT', {})}")
                    break
                
                # Parse station data
                stations = bike_data.get('row', [])
                for station in stations:
                    parsed = self._parse_station_data(station)
                    if parsed:
                        all_stations.append(parsed)
                
                # Check if we've fetched all stations
                if end_idx >= total_count:
                    break
                
                start_idx = end_idx + 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                if Config.FALLBACK_TO_CACHE:
                    logger.info("Will fallback to cached data")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        # Convert to DataFrame
        if all_stations:
            df = pd.DataFrame(all_stations)
            logger.info(f"Successfully fetched {len(df)} stations")
            
            # Add datetime components for feature engineering
            df['fetch_date'] = df['timestamp'].dt.date
            df['fetch_hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            return df
        else:
            logger.warning("No station data fetched")
            return pd.DataFrame()
    
    def fetch_station(self, station_id: str) -> Optional[Dict]:
        """Fetch data for a specific station"""
        df = self.fetch_all_stations()
        if not df.empty:
            station_data = df[df['station_id'] == station_id]
            if not station_data.empty:
                return station_data.iloc[0].to_dict()
        return None
    
    def get_high_risk_stations(self, threshold: float = 0.8) -> pd.DataFrame:
        """Get stations with high utilization (potential stockout risk)"""
        df = self.fetch_all_stations()
        if not df.empty:
            high_risk = df[df['utilization_rate'] >= threshold].copy()
            high_risk = high_risk.sort_values('utilization_rate', ascending=False)
            return high_risk[['station_id', 'station_name', 'available_bikes', 
                            'utilization_rate', 'is_stockout']]
        return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean fetched data"""
        if df.empty:
            return df
        
        # Remove invalid stations
        df = df[df['station_capacity'] > 0].copy()
        
        # Fix any negative values
        df['available_bikes'] = df['available_bikes'].clip(lower=0)
        df['available_racks'] = df['available_racks'].clip(lower=0)
        
        # Ensure consistency
        df['available_racks'] = df['station_capacity'] - df['available_bikes']
        
        # Log any anomalies
        anomalies = df[df['available_bikes'] > df['station_capacity']]
        if not anomalies.empty:
            logger.warning(f"Found {len(anomalies)} stations with bikes > capacity")
        
        return df


if __name__ == "__main__":
    # Test the collector
    logging.basicConfig(level=logging.INFO)
    
    collector = RealtimeBikeCollector()
    
    # Fetch all stations
    print("Fetching all stations...")
    df = collector.fetch_all_stations()
    
    if not df.empty:
        print(f"\nFetched {len(df)} stations")
        print(f"Stockout stations: {df['is_stockout'].sum()}")
        print(f"Nearly empty stations: {df['is_nearly_empty'].sum()}")
        print(f"Average utilization: {df['utilization_rate'].mean():.2%}")
        
        # Show high risk stations
        print("\nHigh risk stations:")
        high_risk = collector.get_high_risk_stations(threshold=0.7)
        if not high_risk.empty:
            print(high_risk.head(10))
    else:
        print("Failed to fetch station data")