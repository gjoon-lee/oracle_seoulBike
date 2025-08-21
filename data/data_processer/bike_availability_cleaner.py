# bike_availability_cleaner.py
"""
Optimized bike availability processor with cached mapping dictionary.
Uses pre-computed 대여소번호 → station_id mappings from rental_station_mapping table.
Achieves 99.96% station coverage (2779/2780 stations) via multi-strategy matching.
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
        self.rental_to_station = {}  # Cached mapping dictionary
        
    def _setup_logger(self):
        logger = logging.getLogger('AvailabilityProcessor')
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        fh = logging.FileHandler('availability_processing.log')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_station_mapping(self):
        """
        Load pre-computed 대여소번호 → station_id mapping from database.
        This mapping was created using multi-strategy matching:
        1. Address-based matching (exact + partial)  
        2. Coordinate-based matching (6-2 decimal precision)
        Achieves 99.96% coverage (2779/2780 stations).
        """
        try:
            cached = self.db.read_query(text("""
                SELECT rental_number, station_id 
                FROM rental_station_mapping
            """))
            
            if len(cached) > 0:
                self.rental_to_station = dict(zip(
                    cached['rental_number'].astype(str), 
                    cached['station_id']
                ))
                self.logger.info(f"Loaded {len(self.rental_to_station)} station mappings from database")
                self.logger.info(f"Coverage: {len(self.rental_to_station)/2780*100:.2f}% of all stations")
                return True
            else:
                self.logger.error("No mappings found in rental_station_mapping table!")
                self.logger.error("Run station mapping generation first.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading station mapping: {e}")
            return False
    
    def process_file(self, filepath):
        """Process availability data CSV file using cached mappings"""
        
        start_time = time.time()
        filename = os.path.basename(filepath)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {filename}")
        self.logger.info(f"{'='*60}")
        
        # Load mapping if not already loaded
        if not self.rental_to_station:
            if not self.load_station_mapping():
                return False
        
        # Read availability data with Korean encoding
        try:
            df = pd.read_csv(filepath, encoding='cp949')
        except UnicodeDecodeError:
            self.logger.warning(f"CP949 decoding failed for {filename}, trying UTF-8")
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except:
                self.logger.error(f"Failed to read {filename} with any encoding")
                return False
        
        initial_rows = len(df)
        self.logger.info(f"Read {initial_rows:,} rows from {filename}")
        
        # Convert rental number to string for mapping
        df['rental_number'] = df['대여소번호'].astype(str).str.zfill(5)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['일시'])
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['시간대'].astype(int)
        
        # Map to station IDs using cached dictionary (FAST!)
        df['station_id'] = df['rental_number'].map(self.rental_to_station)
        
        # Check mapping coverage
        unmapped_count = df['station_id'].isna().sum()
        mapped_count = len(df) - unmapped_count
        coverage = mapped_count / len(df) * 100
        
        self.logger.info(f"Mapping results: {mapped_count:,}/{len(df):,} records ({coverage:.1f}%)")
        
        if unmapped_count > 0:
            unmapped_rentals = df[df['station_id'].isna()]['rental_number'].unique()
            self.logger.warning(f"Unmapped rental numbers: {len(unmapped_rentals)} unique stations")
            
            # Save unmapped data for reference
            df_unmapped = df[df['station_id'].isna()].copy()
            unmapped_filename = f"unmapped_{filename}"
            df_unmapped.to_csv(unmapped_filename, index=False, encoding='utf-8')
            self.logger.info(f"Saved unmapped data to {unmapped_filename}")
        
        # Keep only mapped stations
        df_mapped = df[df['station_id'].notna()].copy()
        
        if len(df_mapped) == 0:
            self.logger.error("No rows could be mapped to station IDs!")
            return False
        
        # Process availability data
        df_mapped['available_bikes'] = df_mapped['거치대수량'].astype(int)
        df_mapped['station_name_kr'] = df_mapped['대여소명']
        
        # Calculate dynamic capacity (max observed + 20% buffer)
        station_capacity = df_mapped.groupby('station_id')['available_bikes'].max()
        station_capacity = (station_capacity * 1.2).fillna(30).astype(int)
        df_mapped['station_capacity'] = df_mapped['station_id'].map(station_capacity)
        df_mapped['available_racks'] = (df_mapped['station_capacity'] - df_mapped['available_bikes']).clip(lower=0)
        
        # Create classification targets
        df_mapped['is_stockout'] = (df_mapped['available_bikes'] <= 2).astype(int)
        df_mapped['is_nearly_empty'] = (df_mapped['available_bikes'] <= 5).astype(int)
        df_mapped['is_nearly_full'] = (df_mapped['available_racks'] <= 5).astype(int)
        
        # Aggregate to hourly level
        hourly = df_mapped.groupby(['station_id', 'date', 'hour']).agg({
            'available_bikes': 'mean',
            'station_capacity': 'first',
            'available_racks': 'mean',
            'is_stockout': 'max',
            'is_nearly_empty': 'max',
            'is_nearly_full': 'max'
        }).round(1).reset_index()
        
        # Create table if not exists
        self._create_availability_table()
        
        # Save to database
        self.db.insert_dataframe(hourly, 'bike_availability_hourly')
        
        # Final statistics
        unique_stations = len(hourly['station_id'].unique())
        stockout_hours = hourly['is_stockout'].sum()
        processing_time = time.time() - start_time
        
        self.logger.info(f"""
        Processing complete:
        - Original rows: {initial_rows:,}
        - Mapped rows: {len(df_mapped):,} ({coverage:.1f}%)
        - Unique stations: {unique_stations:,}
        - Hourly records: {len(hourly):,}
        - Stockout hours: {stockout_hours:,}
        - Processing time: {processing_time:.1f}s
        """)
        
        return True
    
    def _create_availability_table(self):
        """Create bike_availability_hourly table with indexes"""
        
        create_table = text("""
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
            );
        """)
        
        # Create indexes for common queries
        create_date_idx = text("CREATE INDEX IF NOT EXISTS idx_availability_date ON bike_availability_hourly(date);")
        create_stockout_idx = text("CREATE INDEX IF NOT EXISTS idx_availability_stockout ON bike_availability_hourly(is_stockout);")
        create_station_idx = text("CREATE INDEX IF NOT EXISTS idx_availability_station ON bike_availability_hourly(station_id);")
        
        self.db.execute_query(create_table)
        self.db.execute_query(create_date_idx)
        self.db.execute_query(create_stockout_idx)
        self.db.execute_query(create_station_idx)
    
    def process_directory(self, directory_path, pattern='*.csv', limit_files=None):
        """Process all availability files in directory"""
        import glob
        
        files = glob.glob(os.path.join(directory_path, pattern))
        files.sort()
        
        if limit_files:
            files = files[:limit_files]
        
        if not files:
            self.logger.error(f"No files found matching pattern {pattern} in {directory_path}")
            return
        
        self.logger.info(f"Found {len(files)} availability files to process")
        
        # Load mapping once before processing all files
        if not self.load_station_mapping():
            self.logger.error("Cannot proceed without station mapping")
            return
        
        success_count = 0
        total_start_time = time.time()
        
        for i, filepath in enumerate(files, 1):
            try:
                self.logger.info(f"Processing file {i}/{len(files)}: {os.path.basename(filepath)}")
                if self.process_file(filepath):
                    success_count += 1
                else:
                    self.logger.error(f"Failed to process {filepath}")
            except Exception as e:
                self.logger.error(f"Exception processing {filepath}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        total_time = time.time() - total_start_time
        
        self.logger.info(f"""
        ========================================
        BATCH PROCESSING COMPLETE
        ========================================
        Files processed: {success_count}/{len(files)} successful
        Total time: {total_time:.1f}s
        Average time per file: {total_time/len(files):.1f}s
        """)
        
        # Final summary
        try:
            result = self.db.read_query(text("""
                SELECT 
                    COUNT(DISTINCT station_id) as unique_stations,
                    COUNT(*) as total_records,
                    SUM(is_stockout) as stockout_hours,
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM bike_availability_hourly
            """))
            
            if len(result) > 0:
                stats = result.iloc[0]
                self.logger.info(f"""
                FINAL DATABASE STATISTICS:
                - Unique stations: {stats['unique_stations']:,}
                - Total hourly records: {stats['total_records']:,}
                - Stockout hours: {stats['stockout_hours']:,}
                - Date range: {stats['start_date']} to {stats['end_date']}
                """)
        except Exception as e:
            self.logger.error(f"Error generating final statistics: {e}")


# Usage examples
if __name__ == "__main__":
    from db_connection import BikeDataDB
    
    # Connect to database
    db = BikeDataDB()
    db.connect()
    
    # Create processor
    processor = BikeAvailabilityProcessor(db)
    
    # Example 1: Process single file
    # processor.process_file('availability_data/data_2401.csv')
    
    # Example 2: Process all files in directory
    processor.process_directory('availability_data/')
    
    # Example 3: Process limited files for testing
    # processor.process_directory('availability_data/', limit_files=2)