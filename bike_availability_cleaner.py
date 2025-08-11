# bike_availability_cleaner.py
"""
Process bike availability data using existing station master
No API calls - just mapping and processing
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from sqlalchemy import text

class BikeAvailabilityProcessor:
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = self._setup_logger()
        self.station_mapping = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('AvailabilityProcessor')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler('availability_processing.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_station_mapping(self, station_info_file):
        """Build 대여소번호 → station_id mapping"""
        
        # Check cached mapping first
        try:
            cached = self.db.read_query("SELECT rental_number, station_id FROM rental_station_mapping")
            if len(cached) > 0:
                self.station_mapping = dict(zip(cached['rental_number'].astype(str), cached['station_id']))
                self.logger.info(f"Loaded {len(self.station_mapping)} cached mappings")
                return
        except:
            pass
        
        self.logger.info("Building station mapping...")
        
        # Create mapping table
        self.db.execute_query(text("""
            CREATE TABLE IF NOT EXISTS rental_station_mapping (
                rental_number VARCHAR(20) PRIMARY KEY,
                station_id VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Load station info Excel
        station_info = pd.read_excel(station_info_file, sheet_name='Sheet')
        station_info = station_info[['대여소번호', '위도', '경도']].copy()
        station_info['lat_5dp'] = station_info['위도'].round(5).astype(str)
        station_info['lon_5dp'] = station_info['경도'].round(5).astype(str)
        
        # Load station master from DB
        master = self.db.read_query("""
            SELECT station_id, lat_5dp, lon_5dp 
            FROM station_master
        """)
        
        if len(master) == 0:
            raise Exception("No station master data! Run update_station_master.py first")
        
        # Create lookup dict
        coord_to_id = {}
        for _, row in master.iterrows():
            key = (row['lat_5dp'], row['lon_5dp'])
            coord_to_id[key] = row['station_id']
        
        # Match stations
        matched = 0
        for _, row in station_info.iterrows():
            rental_num = str(row['대여소번호'])
            key = (row['lat_5dp'], row['lon_5dp'])
            
            if key in coord_to_id:
                station_id = coord_to_id[key]
                self.station_mapping[rental_num] = station_id
                
                # Save to DB
                self.db.execute_query(
                    text("INSERT INTO rental_station_mapping VALUES (:rental_num, :station_id) ON CONFLICT DO NOTHING"),
                    {'rental_num': rental_num, 'station_id': station_id}
                )
                matched += 1
        
        self.logger.info(f"Matched {matched}/{len(station_info)} stations")
    
    def process_file(self, filepath, station_info_file):
        """Process availability data file"""
        
        start_time = time.time()
        
        # Load mapping
        if not self.station_mapping:
            self.load_station_mapping(station_info_file)
        
        # Read data
        df = pd.read_excel(filepath)
        initial_rows = len(df)
        
        # Convert rental number to string
        df['대여소번호'] = df['대여소번호'].astype(str)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['일시'])
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        
        # Map to station IDs
        df['station_id'] = df['대여소번호'].map(self.station_mapping)
        
        # Keep only mapped
        df = df[df['station_id'].notna()]
        
        # Process
        df['available_bikes'] = df['거치대수']
        
        # Calculate capacity
        station_capacity = df.groupby('station_id')['available_bikes'].max() * 1.2
        df['station_capacity'] = df['station_id'].map(station_capacity).fillna(30).astype(int)
        df['available_racks'] = df['station_capacity'] - df['available_bikes']
        
        # Targets
        df['is_stockout'] = (df['available_bikes'] <= 2).astype(int)
        df['is_nearly_empty'] = (df['available_bikes'] <= 5).astype(int)
        df['is_nearly_full'] = (df['available_racks'] <= 5).astype(int)
        
        # Aggregate hourly
        hourly = df.groupby(['station_id', 'date', 'hour']).agg({
            'available_bikes': 'mean',
            'station_capacity': 'first',
            'available_racks': 'mean',
            'is_stockout': 'max',
            'is_nearly_empty': 'max',
            'is_nearly_full': 'max'
        }).round(1).reset_index()
        
        # Create table
        self.db.execute_query(text("""
            CREATE TABLE IF NOT EXISTS bike_availability_hourly (
                station_id VARCHAR(20),
                date DATE,
                hour INTEGER,
                available_bikes REAL,
                station_capacity INTEGER,
                available_racks REAL,
                is_stockout INTEGER,
                is_nearly_empty INTEGER,
                is_nearly_full INTEGER,
                PRIMARY KEY (station_id, date, hour)
            )
        """))
        
        # Save
        self.db.insert_dataframe(hourly, 'bike_availability_hourly')
        
        self.logger.info(f"""
        ✅ Processed {os.path.basename(filepath)}:
        - Rows: {initial_rows:,} → {len(hourly):,}
        - Stations: {len(hourly['station_id'].unique())}
        - Stockouts: {hourly['is_stockout'].sum():,}
        - Time: {time.time() - start_time:.1f}s
        """)


if __name__ == "__main__":
    from db_connection import BikeDataDB
    
    db = BikeDataDB()
    db.connect()
    
    processor = BikeAvailabilityProcessor(db)
    processor.process_file(
        filepath='availability_data/data_2412.xlsx',
        station_info_file='availability_data/station_info.xlsx'
    )