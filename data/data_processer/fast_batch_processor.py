# fast_batch_processor.py - Optimized for speed
import pandas as pd
import time
import logging
from sqlalchemy import text
from db_connection import BikeDataDB
import glob
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info('=== FAST BATCH PROCESSOR FOR 12 MONTHS ===')
print('=== FAST BATCH PROCESSOR FOR 12 MONTHS ===')

# Setup
db = BikeDataDB()
db.connect()

# Load mapping once
logger.info('Loading station mappings...')
mapping = db.read_query(text("SELECT rental_number, station_id FROM rental_station_mapping"))
rental_to_station = dict(zip(mapping['rental_number'], mapping['station_id']))
logger.info(f'Loaded {len(rental_to_station):,} station mappings')

# Get all files
files = sorted(glob.glob('availability_data/data_*.csv'))
logger.info(f'Found {len(files)} files to process')

# Create table once
db.execute_query(text("""
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
"""))

total_start_time = time.time()
all_hourly_data = []

for i, filepath in enumerate(files, 1):
    file_start_time = time.time()
    filename = os.path.basename(filepath)
    print(f'\n[{i}/12] Processing {filename}...')
    
    # Read file
    df = pd.read_csv(filepath, encoding='cp949')
    print(f'  Read {len(df):,} rows')
    
    # Quick processing
    df['rental_number'] = df['대여소번호'].astype(str).str.zfill(5)
    df['station_id'] = df['rental_number'].map(rental_to_station)
    
    # Count coverage
    mapped_count = df['station_id'].notna().sum()
    coverage = mapped_count / len(df) * 100
    print(f'  Mapped: {mapped_count:,}/{len(df):,} ({coverage:.1f}%)')
    
    # Keep only mapped
    df = df[df['station_id'].notna()].copy()
    
    if len(df) == 0:
        print('  No mapped data!')
        continue
    
    # Process quickly
    df['datetime'] = pd.to_datetime(df['일시'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['시간대'].astype(int)
    df['available_bikes'] = df['거치대수량'].astype(int)
    
    # Simple capacity calculation
    station_capacity = df.groupby('station_id')['available_bikes'].max() * 1.2
    station_capacity = station_capacity.fillna(30).astype(int)
    df['station_capacity'] = df['station_id'].map(station_capacity)
    df['available_racks'] = (df['station_capacity'] - df['available_bikes']).clip(lower=0)
    
    # Quick targets
    df['is_stockout'] = (df['available_bikes'] <= 2).astype(int)
    df['is_nearly_empty'] = (df['available_bikes'] <= 5).astype(int)
    df['is_nearly_full'] = (df['available_racks'] <= 5).astype(int)
    
    # Aggregate
    hourly = df.groupby(['station_id', 'date', 'hour']).agg({
        'available_bikes': 'mean',
        'station_capacity': 'first',
        'available_racks': 'mean',
        'is_stockout': 'max',
        'is_nearly_empty': 'max',
        'is_nearly_full': 'max'
    }).round(1).reset_index()
    
    print(f'  Generated {len(hourly):,} hourly records')
    
    # Collect for batch insert
    all_hourly_data.append(hourly)
    
    file_time = time.time() - file_start_time
    print(f'  Time: {file_time:.1f}s')

# Batch insert all data
print(f'\n=== BATCH DATABASE INSERT ===')
insert_start = time.time()

if all_hourly_data:
    combined_data = pd.concat(all_hourly_data, ignore_index=True)
    print(f'Combined {len(combined_data):,} total records')
    
    # Insert in chunks for memory efficiency
    chunk_size = 50000
    for i in range(0, len(combined_data), chunk_size):
        chunk = combined_data.iloc[i:i+chunk_size]
        db.insert_dataframe(chunk, 'bike_availability_hourly')
        print(f'  Inserted chunk {i//chunk_size + 1}: {len(chunk):,} records')

insert_time = time.time() - insert_start
total_time = time.time() - total_start_time

print(f'\n=== FINAL RESULTS ===')
print(f'Processing time: {total_time:.1f}s')
print(f'Database insert time: {insert_time:.1f}s')

# Final stats
final_stats = db.read_query(text("""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT station_id) as stations,
        MIN(date) as start_date,
        MAX(date) as end_date,
        SUM(is_stockout) as stockout_hours
    FROM bike_availability_hourly
"""))

if len(final_stats) > 0:
    stats = final_stats.iloc[0]
    print(f'Final database:')
    print(f'  Total records: {stats["total_records"]:,}')
    print(f'  Unique stations: {stats["stations"]:,}')
    print(f'  Date range: {stats["start_date"]} to {stats["end_date"]}')
    print(f'  Stockout hours: {stats["stockout_hours"]:,}')
    
print('\nAll 12 months processed successfully!')