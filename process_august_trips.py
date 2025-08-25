#!/usr/bin/env python3
"""
Process August 2025 Seoul bike trip data and upload to PostgreSQL
Handles the 5-7 day data delay from Seoul Open API
"""

import os
import sys
import glob
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_connection import BikeDataDB
from data.data_processer.bike_data_cleaner import BikeDataCleaner
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('august_trip_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_net_flow_column(db):
    """Ensure net_flow column exists in station_hourly_flow table"""
    try:
        # Check if net_flow column exists
        check_query = text("""
            SELECT column_name, generation_expression
            FROM information_schema.columns 
            WHERE table_name='station_hourly_flow' 
            AND column_name='net_flow'
        """)
        result = db.read_query(check_query)
        
        if result.empty:
            logger.info("Adding net_flow as GENERATED column...")
            # Add net_flow as a generated column
            alter_query = text("""
                ALTER TABLE station_hourly_flow 
                ADD COLUMN IF NOT EXISTS net_flow INTEGER 
                GENERATED ALWAYS AS (bikes_arrived - bikes_departed) STORED
            """)
            db.execute_query(alter_query)
            logger.info("[SUCCESS] net_flow column added as GENERATED column")
        else:
            # Check if it's already a generated column
            if not result['generation_expression'].isna().all():
                logger.info("[SUCCESS] net_flow is already a GENERATED column (auto-calculated)")
            else:
                logger.info("[INFO] net_flow column exists but is not GENERATED")
                # Try to convert to generated column
                try:
                    # First drop the existing column
                    drop_query = text("ALTER TABLE station_hourly_flow DROP COLUMN net_flow")
                    db.execute_query(drop_query)
                    # Then add as generated
                    add_query = text("""
                        ALTER TABLE station_hourly_flow 
                        ADD COLUMN net_flow INTEGER 
                        GENERATED ALWAYS AS (bikes_arrived - bikes_departed) STORED
                    """)
                    db.execute_query(add_query)
                    logger.info("[SUCCESS] Converted net_flow to GENERATED column")
                except:
                    logger.info("[INFO] Could not convert to GENERATED, column will remain as-is")
            
    except Exception as e:
        logger.error(f"Error checking net_flow column: {e}")
        # Continue anyway - the column likely exists and works

def process_august_data():
    """Process August 2025 trip data"""
    
    logger.info("="*60)
    logger.info("AUGUST 2025 TRIP DATA PROCESSOR")
    logger.info("="*60)
    
    # Initialize database connection
    logger.info("\n1. Connecting to PostgreSQL...")
    db = BikeDataDB()
    db.connect()
    
    if not db.test_connection():
        logger.error("[ERROR] Database connection failed!")
        return False
    
    # Ensure net_flow column exists (as GENERATED column)
    try:
        ensure_net_flow_column(db)
    except Exception as e:
        logger.warning(f"Note about net_flow column: {e}")
        logger.info("Continuing with processing...")
    
    # Initialize cleaner
    logger.info("\n2. Initializing data cleaner...")
    cleaner = BikeDataCleaner(db)
    
    # Look for August data files
    data_paths = [
        'data/raw_data/bike_historical_data/2025_08/',
        'bike_historical_data/Y2025/2025_08/',
        'bike_historical_data/202508/',
        '.'  # Current directory as fallback
    ]
    
    files_found = []
    for path in data_paths:
        pattern = os.path.join(path, 'tpss_bcycl_od_statnhm_202508*.csv')
        files = glob.glob(pattern)
        files_found.extend(files)
    
    if not files_found:
        logger.warning("\n[WARNING] No August 2025 files found!")
        logger.info("Please place your files in one of these locations:")
        for path in data_paths:
            logger.info(f"  - {path}")
        logger.info("Files should match pattern: tpss_bcycl_od_statnhm_202508*.csv")
        return False
    
    # Remove duplicates and sort
    files_found = sorted(list(set(files_found)))
    
    logger.info(f"\n3. Found {len(files_found)} files to process:")
    for file in files_found:
        logger.info(f"  - {os.path.basename(file)}")
    
    # Check existing data
    logger.info("\n4. Checking existing data in database...")
    existing_check = db.read_query(text("""
        SELECT 
            COUNT(*) as record_count,
            MIN(flow_date) as earliest_date,
            MAX(flow_date) as latest_date,
            COUNT(DISTINCT station_id) as station_count
        FROM station_hourly_flow
        WHERE flow_date >= '2025-08-01'
    """))
    
    if not existing_check.empty and existing_check.iloc[0]['record_count'] > 0:
        logger.info(f"  Existing August 2025 records: {existing_check.iloc[0]['record_count']:,}")
        logger.info(f"  Date range: {existing_check.iloc[0]['earliest_date']} to {existing_check.iloc[0]['latest_date']}")
        logger.info(f"  Stations: {existing_check.iloc[0]['station_count']:,}")
    else:
        logger.info("  No existing August 2025 data found")
    
    # Process each file
    logger.info("\n5. Processing files...")
    success_count = 0
    failed_files = []
    
    for i, filepath in enumerate(files_found, 1):
        filename = os.path.basename(filepath)
        logger.info(f"\n[{i}/{len(files_found)}] Processing {filename}...")
        
        try:
            if cleaner.process_file(filepath):
                success_count += 1
                logger.info(f"  [SUCCESS] Successfully processed {filename}")
            else:
                failed_files.append(filename)
                logger.warning(f"  [WARNING] Failed to process {filename}")
        except Exception as e:
            logger.error(f"  [ERROR] Error processing {filename}: {e}")
            failed_files.append(filename)
    
    # net_flow is automatically calculated as a GENERATED column
    logger.info("\n6. net_flow values are auto-calculated (GENERATED column)")
    
    # Final verification
    logger.info("\n7. Verifying processed data...")
    final_check = db.read_query(text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT station_id) as unique_stations,
            COUNT(DISTINCT flow_date) as days_covered,
            MIN(flow_date) as start_date,
            MAX(flow_date) as end_date,
            SUM(bikes_departed) as total_departures,
            SUM(bikes_arrived) as total_arrivals,
            AVG(net_flow) as avg_net_flow,
            SUM(CASE WHEN net_flow IS NULL THEN 1 ELSE 0 END) as null_netflow_count
        FROM station_hourly_flow
        WHERE flow_date >= '2025-08-01'
    """))
    
    if not final_check.empty:
        stats = final_check.iloc[0]
        logger.info("\n" + "="*60)
        logger.info("AUGUST 2025 DATA STATISTICS")
        logger.info("="*60)
        logger.info(f"Total records:     {stats['total_records']:,}")
        logger.info(f"Unique stations:   {stats['unique_stations']:,}")
        logger.info(f"Days covered:      {stats['days_covered']}")
        logger.info(f"Date range:        {stats['start_date']} to {stats['end_date']}")
        logger.info(f"Total departures:  {stats['total_departures']:,}")
        logger.info(f"Total arrivals:    {stats['total_arrivals']:,}")
        if stats['avg_net_flow'] is not None:
            logger.info(f"Avg net flow:      {stats['avg_net_flow']:.2f}")
        logger.info(f"Null net_flow:     {stats['null_netflow_count']}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("[COMPLETE] PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Successfully processed: {success_count}/{len(files_found)} files")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    
    # Check data quality
    logger.info("\n8. Data quality check...")
    quality_check = db.read_query(text("""
        SELECT 
            flow_date,
            COUNT(*) as records,
            COUNT(DISTINCT station_id) as stations,
            AVG(bikes_departed) as avg_departures,
            AVG(bikes_arrived) as avg_arrivals,
            AVG(net_flow) as avg_net_flow
        FROM station_hourly_flow
        WHERE flow_date >= '2025-08-13' 
        AND flow_date <= '2025-08-19'
        GROUP BY flow_date
        ORDER BY flow_date
    """))
    
    if not quality_check.empty:
        logger.info("\nDaily summary (Aug 13-19):")
        for _, row in quality_check.iterrows():
            if row['avg_net_flow'] is not None:
                logger.info(f"  {row['flow_date']}: {row['records']:,} records, "
                           f"{row['stations']} stations, "
                           f"net_flow avg: {row['avg_net_flow']:.1f}")
            else:
                logger.info(f"  {row['flow_date']}: {row['records']:,} records, "
                           f"{row['stations']} stations")
    
    logger.info("\n[READY] August data ready for predictions!")
    logger.info("Next steps:")
    logger.info("1. Restart the prediction API: cd realtime_prediction && python main.py")
    logger.info("2. Test XGBoost predictions: GET http://localhost:8000/predict/all")
    logger.info("3. Verify net_flow values are not 0")
    
    return True

if __name__ == "__main__":
    success = process_august_data()
    sys.exit(0 if success else 1)