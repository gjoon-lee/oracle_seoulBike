import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import logging
from pathlib import Path
import time

class BikeDataCleaner:
    """
    Cleaning pipeline for Seoul bike trip data
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = self._setup_logger()
        
        # Column name mapping (garbled Korean -> English)
        self.column_mapping = {
            '기준_날짜': 'record_date',
            '집계_기준': 'aggregation_type',  
            '기준_시간대': 'time_slot',
            '시작_대여소_ID': 'start_station_id',
            '시작_대여소명': 'start_station_name',
            '종료_대여소_ID': 'end_station_id',
            '종료_대여소명': 'end_station_name',
            '전체_건수': 'trip_count',
            '전체_이용_분': 'total_duration_min',
            '전체_이용_거리': 'total_distance_m'
        }
        
        # Aggregation type mapping
        self.agg_type_mapping = {
            '출발시간': 'departure',
            '도착시간': 'arrival'
        }
        
    def _setup_logger(self):
        """Configure logging"""
        logger = logging.getLogger('BikeDataCleaner')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('bike_cleaning.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def read_csv_with_encoding(self, filepath):
        """
        Read CSV with Korean encoding (cp949 or euc-kr)
        Tries multiple encodings to handle different file formats
        """
        encodings = ['cp949', 'euc-kr', 'utf-8-sig', 'utf-8']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                self.logger.info(f"✅ Successfully read {filepath} with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error with {encoding}: {e}")
                
        raise ValueError(f"Could not read {filepath} with any encoding")
    
    def extract_date_from_filename(self, filename):
        """Extract date from filename format: tpss_bcycl_od_statnhm_YYYYMMDD.csv"""
        try:
            date_str = filename.split('_')[-1].replace('.csv', '')
            return datetime.strptime(date_str, '%Y%m%d').date()
        except:
            self.logger.error(f"Could not extract date from {filename}")
            return None
    
    def clean_dataframe(self, df, file_date):
        """
        Main cleaning function
        - Rename columns to English
        - Convert data types
        - Handle nulls and invalid values
        - Add derived columns
        """
        start_time = time.time()
        initial_rows = len(df)
        
        # 1. Rename columns
        df = df.rename(columns=self.column_mapping)
        
        # 2. Convert date column
        df['record_date'] = pd.to_datetime(df['record_date'], format='%Y%m%d').dt.date
        
        # 3. Map aggregation types
        df['aggregation_type'] = df['aggregation_type'].map(self.agg_type_mapping)
        
        # 4. Convert time slot to hour (e.g., 935 -> 9, 1415 -> 14)
        df['hour'] = df['time_slot'] // 100
        df['minute'] = df['time_slot'] % 100
        
        # 5. Handle missing values
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # For numeric columns, fill NaN with 0
        numeric_cols = ['trip_count', 'total_duration_min', 'total_distance_m']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 6. Data type conversions
        df['trip_count'] = pd.to_numeric(df['trip_count'], errors='coerce').fillna(0).astype(int)
        df['total_duration_min'] = pd.to_numeric(df['total_duration_min'], errors='coerce').fillna(0)
        df['total_distance_m'] = pd.to_numeric(df['total_distance_m'], errors='coerce').fillna(0)
        
        # 7. Remove invalid records
        # Remove records with invalid station IDs
        df = df.dropna(subset=['start_station_id', 'end_station_id'])
        
        # Remove records with zero trips (likely data errors)
        df = df[df['trip_count'] > 0]
        
        # 8. Add derived columns
        df['day_of_week'] = pd.to_datetime(df['record_date']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['season'] = pd.to_datetime(df['record_date']).dt.month.map(self._get_season)
        
        # 9. Create full timestamp
        df['timestamp'] = pd.to_datetime(df['record_date'].astype(str) + ' ' + 
                                       df['hour'].astype(str).str.zfill(2) + ':' +
                                       df['minute'].astype(str).str.zfill(2))
        
        # Log cleaning stats
        final_rows = len(df)
        processing_time = time.time() - start_time
        
        self.logger.info(f"""
        Cleaning complete for {file_date}:
        - Initial rows: {initial_rows:,}
        - Final rows: {final_rows:,}
        - Removed: {initial_rows - final_rows:,} ({(initial_rows - final_rows) / initial_rows * 100:.1f}%)
        - Processing time: {processing_time:.2f} seconds
        """)
        
        return df
    
    def _get_season(self, month):
        """Map month to Korean season"""
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    def aggregate_hourly_flow(self, df):
        """
        Aggregate data to hourly station flow
        Creates records showing bikes arriving/departing each hour
        """
        hourly_flows = []
        
        # Group by station, date, hour
        for (station_id, date, hour), group in df.groupby(['start_station_id', 'record_date', 'hour']):
            # Calculate departures and arrivals
            departures = group[group['aggregation_type'] == 'departure']['trip_count'].sum()
            
            # For arrivals, we need to look at end_station_id
            arrivals_df = df[(df['end_station_id'] == station_id) & 
                            (df['record_date'] == date) & 
                            (df['hour'] == hour) & 
                            (df['aggregation_type'] == 'arrival')]
            arrivals = arrivals_df['trip_count'].sum()
            
            # Get other metrics
            day_of_week = group['day_of_week'].iloc[0]
            is_weekend = group['is_weekend'].iloc[0]
            season = group['season'].iloc[0]
            
            # Average trip metrics (for departures only)
            departure_data = group[group['aggregation_type'] == 'departure']
            if len(departure_data) > 0 and departures > 0:
                avg_duration = departure_data['total_duration_min'].sum() / departures
                avg_distance = departure_data['total_distance_m'].sum() / departures
            else:
                avg_duration = 0
                avg_distance = 0
            
            hourly_flows.append({
                'station_id': station_id,
                'flow_date': date,
                'flow_hour': hour,
                'bikes_departed': int(departures),
                'bikes_arrived': int(arrivals),
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'season': season,
                'avg_trip_duration_min': round(avg_duration, 2),
                'avg_trip_distance_m': round(avg_distance, 2)
            })
        
        return pd.DataFrame(hourly_flows)
    
    def validate_data(self, df):
        """Run data quality checks"""
        issues = []
        
        # Check for duplicate records
        duplicates = df.duplicated(subset=['record_date', 'time_slot', 'start_station_id', 
                                         'end_station_id', 'aggregation_type']).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate records")
        
        # Check for unrealistic values
        long_trips = df[df['total_duration_min'] > 480].shape[0]  # > 8 hours
        if long_trips > 0:
            issues.append(f"Found {long_trips} trips longer than 8 hours")
        
        # Check for missing critical data
        missing_stations = df[df['start_station_id'].isna() | df['end_station_id'].isna()].shape[0]
        if missing_stations > 0:
            issues.append(f"Found {missing_stations} records with missing station IDs")
        
        return issues
    
    def process_file(self, filepath):
        """Process a single CSV file through the entire pipeline"""
        filename = os.path.basename(filepath)
        file_date = self.extract_date_from_filename(filename)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {filename}")
        self.logger.info(f"{'='*60}")
        
        try:
            # 1. Read CSV with Korean encoding
            df = self.read_csv_with_encoding(filepath)
            
            # 2. Clean the data
            df_clean = self.clean_dataframe(df, file_date)
            
            # 3. Validate
            issues = self.validate_data(df_clean)
            if issues:
                self.logger.warning(f"Data quality issues: {', '.join(issues)}")
            
            # 4. Load raw data to PostgreSQL
            self.logger.info("Loading raw trip data to PostgreSQL...")
            rows_inserted = self.db.insert_dataframe(df_clean, 'raw_bike_trips')
            
            # 5. Create hourly aggregations
            self.logger.info("Creating hourly flow aggregations...")
            hourly_flow = self.aggregate_hourly_flow(df_clean)
            
            # 6. Load aggregated data
            flow_rows = self.db.insert_dataframe(hourly_flow, 'station_hourly_flow')
            
            # 7. Log processing stats
            stats = {
                'file_name': filename,
                'record_date': file_date,
                'total_rows': len(df),
                'valid_rows': len(df_clean),
                'duplicate_rows': len(df) - len(df_clean),
                'encoding_errors': 0,
                'processing_time': 0,  # Will be calculated
                'status': 'SUCCESS'
            }
            
            self.db.log_data_quality(filename, stats)
            
            self.logger.info(f"✅ Successfully processed {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {filename}: {e}")
            return False
    
    def process_directory(self, directory_path, pattern='tpss_bcycl_od_statnhm_*.csv'):
        """Process all CSV files in a directory"""
        files = glob.glob(os.path.join(directory_path, pattern))
        files.sort()  # Process in chronological order
        
        self.logger.info(f"Found {len(files)} files to process")
        
        success_count = 0
        for filepath in files:
            if self.process_file(filepath):
                success_count += 1
        
        self.logger.info(f"\nProcessing complete: {success_count}/{len(files)} files successful")
        
    def get_processing_summary(self):
        """Get summary of all processed files"""
        query = """
        SELECT 
            status,
            COUNT(*) as file_count,
            SUM(total_rows) as total_rows_processed,
            AVG(processing_time_sec) as avg_processing_time
        FROM data_quality_log
        GROUP BY status
        """
        return self.db.read_query(query)


# Example usage
if __name__ == "__main__":
    from db_connection import BikeDataDB
    
    # Initialize database connection
    db = BikeDataDB()
    db.connect()
    
    # Initialize cleaner
    cleaner = BikeDataCleaner(db)
    
    # Process a single file
    # cleaner.process_file('data/tpss_bcycl_od_statnhm_20250601.csv')
    
    # Or process all files in a directory
    # cleaner.process_directory('data/')
    
    # Get summary
    # summary = cleaner.get_processing_summary()
    # print(summary)