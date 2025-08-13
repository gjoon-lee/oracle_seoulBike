# weather_data_processor.py
"""
Weather data processor
Cleans weather data for Seoul weather station
Loads to PostgreSQL Database
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import logging
import time

class WeatherDataProcessor:
    """
    Weather data processor 
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = self._setup_logger()
        
        # Column mapping for weather data
        self.column_mapping = {
            '일시': 'datetime',
            '기온(°C)': 'temperature',
            '강수량(mm)': 'precipitation', 
            '풍속(m/s)': 'wind_speed',
            '습도(%)': 'humidity',
            '적설(cm)': 'snow_depth'
        }
        
    def _setup_logger(self):
        """Same logging setup as bike cleaner"""
        logger = logging.getLogger('WeatherProcessor')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler('weather_processing.log')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def read_weather_csv(self, filepath):
        """Read weather CSV with Korean encoding"""
        try:
            df = pd.read_csv(filepath, encoding='cp949')
            self.logger.info(f"✅ Successfully read {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            raise
    
    def process_weather_data(self, df):
        """
        Process weather data
        """
        start_time = time.time()
        initial_rows = len(df)
        
        # 1. Drop station ID and station name columns (first two columns)
        # These columns are '지점' (station ID) and '지점명' (station name)
        columns_to_drop = []
        if '지점' in df.columns:
            columns_to_drop.append('지점')
            df = df.drop(columns=['지점'])
        if '지점명' in df.columns:
            columns_to_drop.append('지점명')
            df = df.drop(columns=['지점명'])
        
        if columns_to_drop:
            self.logger.info(f"Dropped columns: {columns_to_drop}")
        
        # 2. Rename columns
        df = df.rename(columns=self.column_mapping)
        
        # 3. Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        
        # 4. Handle missing values
        df['precipitation'] = df['precipitation'].fillna(0)
        df['snow_depth'] = df['snow_depth'].fillna(0)
        
        # For temperature/humidity/wind, use forward fill then backward fill
        df['temperature'] = df['temperature'].ffill().bfill()
        df['humidity'] = df['humidity'].ffill().bfill()
        df['wind_speed'] = df['wind_speed'].ffill().bfill()
        
        # 5. Remove any remaining rows with missing critical data
        df = df.dropna(subset=['temperature', 'humidity'])
        
        # 6. Create simple derived features
        df['is_raining'] = (df['precipitation'] > 0).astype(int)
        df['is_snowing'] = (df['snow_depth'] > 0).astype(int)
        df['is_freezing'] = (df['temperature'] < 0).astype(int)
        
        # Comfort conditions for biking
        df['is_comfortable'] = (
            (df['temperature'].between(15, 25)) & 
            (df['precipitation'] == 0) & 
            (df['wind_speed'] < 5)
        ).astype(int)
        
        # Weather severity (simple 0-3 scale)
        df['weather_severity'] = (
            (df['temperature'] < 0).astype(int) +
            (df['temperature'] > 30).astype(int) + 
            (df['precipitation'] > 5).astype(int) +
            (df['wind_speed'] > 10).astype(int)
        )
        
        # 7. Simple feels-like temperature
        df['feels_like'] = df['temperature'].copy()
        # Wind chill when cold
        cold_mask = df['temperature'] < 10
        df.loc[cold_mask, 'feels_like'] = df.loc[cold_mask, 'temperature'] - (df.loc[cold_mask, 'wind_speed'] * 0.7)
        
        # 8. Drop the original datetime column (we have date and hour separately)
        df = df.drop(columns=['datetime'])
        
        # Remove duplicates (in case same hour appears twice)
        df = df.drop_duplicates(subset=['date', 'hour'], keep='first')
        
        # Log processing stats
        final_rows = len(df)
        processing_time = time.time() - start_time
        
        self.logger.info(f"""
        Weather processing complete:
        - Initial rows: {initial_rows:,}
        - Final rows: {final_rows:,}
        - Removed: {initial_rows - final_rows:,}
        - Processing time: {processing_time:.2f} seconds
        """)
        
        return df
    
    def process_file(self, filepath):
        """Process a single weather file"""
        filename = os.path.basename(filepath)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {filename}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 1. Read CSV
            df = self.read_weather_csv(filepath)
            
            # 2. Process the data
            df_processed = self.process_weather_data(df)
            
            # 3. Load to PostgreSQL
            self.logger.info("Loading weather data to PostgreSQL...")
            rows = self.db.insert_dataframe(df_processed, 'weather_hourly')
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Successfully processed {filename} in {processing_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {filename}: {e}")
            return False
    
    def process_directory(self, directory_path, pattern='*.csv'):
        """Process all weather files in directory"""
        files = glob.glob(os.path.join(directory_path, pattern))
        files.sort()
        
        self.logger.info(f"Found {len(files)} weather files to process")
        
        success_count = 0
        for filepath in files:
            if self.process_file(filepath):
                success_count += 1
        
        self.logger.info(f"\nProcessing complete: {success_count}/{len(files)} files successful")


# Create the weather table in PostgreSQL
def create_weather_table(db_connection):
    """Create weather_hourly table"""
    from sqlalchemy import text
    
    create_sql = text("""
    CREATE TABLE IF NOT EXISTS weather_hourly (
        date DATE NOT NULL,
        hour INTEGER NOT NULL,
        temperature REAL,
        precipitation REAL,
        wind_speed REAL,
        humidity REAL,
        snow_depth REAL,
        feels_like REAL,
        is_raining INTEGER,
        is_snowing INTEGER,
        is_freezing INTEGER,
        is_comfortable INTEGER,
        weather_severity INTEGER,
        PRIMARY KEY (date, hour)
    );
    """)
    
    create_index_date = text("CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_hourly(date);")
    create_index_comfortable = text("CREATE INDEX IF NOT EXISTS idx_weather_comfortable ON weather_hourly(is_comfortable);")
    
    db_connection.execute_query(create_sql)
    db_connection.execute_query(create_index_date)
    db_connection.execute_query(create_index_comfortable)
    print("Weather table created successfully")


# Example usage
if __name__ == "__main__":
    from db_connection import BikeDataDB
    
    # Initialize database
    db = BikeDataDB()
    db.connect()
    
    # Create table
    create_weather_table(db)
    
    # Initialize processor
    processor = WeatherDataProcessor(db)
    
    # Process weather files
    processor.process_directory('weather_data/')