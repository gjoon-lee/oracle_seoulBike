#!/usr/bin/env python3
"""
Weekly Seoul bike trip data update pipeline
Handles the 5-7 day data delay from Seoul Open API
Run this every Monday to process the previous week's data
"""

import os
import sys
import glob
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_connection import BikeDataDB
from data.data_processer.bike_data_cleaner import BikeDataCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyDataUpdater:
    """Manages weekly updates of Seoul bike trip data"""
    
    def __init__(self):
        self.db = BikeDataDB()
        self.cleaner = None
        
    def connect(self):
        """Initialize database connection"""
        self.db.connect()
        if not self.db.test_connection():
            raise ConnectionError("Failed to connect to database")
        self.cleaner = BikeDataCleaner(self.db)
        logger.info("✅ Database connected")
    
    def get_date_range(self, week='previous'):
        """
        Get date range for processing
        
        Args:
            week: 'previous' for last week, or specific date string 'YYYY-MM-DD'
        
        Returns:
            tuple: (start_date, end_date)
        """
        if week == 'previous':
            # Account for 5-7 day delay
            today = datetime.now()
            days_back = 7  # Start from 7 days ago
            
            # Get last Monday to Sunday
            end_date = today - timedelta(days=days_back)
            # Find last Sunday
            while end_date.weekday() != 6:  # 6 = Sunday
                end_date -= timedelta(days=1)
            
            # Start from Monday
            start_date = end_date - timedelta(days=6)
            
        else:
            # Parse specific date
            start_date = datetime.strptime(week, '%Y-%m-%d')
            end_date = start_date + timedelta(days=6)
        
        return start_date.date(), end_date.date()
    
    def check_existing_data(self, start_date, end_date):
        """Check what data already exists in database"""
        query = f"""
            SELECT 
                flow_date,
                COUNT(*) as records,
                COUNT(DISTINCT station_id) as stations
            FROM station_hourly_flow
            WHERE flow_date >= '{start_date}' 
            AND flow_date <= '{end_date}'
            GROUP BY flow_date
            ORDER BY flow_date
        """
        
        existing = self.db.read_query(query)
        
        if not existing.empty:
            logger.info(f"Found existing data for {len(existing)} days:")
            for _, row in existing.iterrows():
                logger.info(f"  {row['flow_date']}: {row['records']:,} records, {row['stations']} stations")
            return existing
        else:
            logger.info("No existing data found for this period")
            return None
    
    def find_trip_files(self, start_date, end_date):
        """Find trip CSV files for the date range"""
        files_to_process = []
        
        # Generate expected filenames
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            filename_pattern = f"tpss_bcycl_od_statnhm_{date_str}.csv"
            
            # Search in multiple locations
            search_paths = [
                f"bike_historical_data/{current_date.strftime('%Y_%m')}/",
                f"bike_historical_data/Y{current_date.year}/{current_date.strftime('%Y_%m')}/",
                f"bike_historical_data/{current_date.strftime('%Y%m')}/",
                ".",  # Current directory
                "data/"  # Data directory
            ]
            
            file_found = False
            for path in search_paths:
                filepath = os.path.join(path, filename_pattern)
                if os.path.exists(filepath):
                    files_to_process.append(filepath)
                    file_found = True
                    break
            
            if not file_found:
                logger.warning(f"⚠️  File not found for {current_date}: {filename_pattern}")
            
            current_date += timedelta(days=1)
        
        return files_to_process
    
    def process_week_data(self, start_date, end_date):
        """Process a week of trip data"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📅 Processing week: {start_date} to {end_date}")
        logger.info(f"{'='*60}")
        
        # Check existing data
        existing_data = self.check_existing_data(start_date, end_date)
        
        # Find files to process
        files = self.find_trip_files(start_date, end_date)
        
        if not files:
            logger.error("❌ No files found for this week!")
            logger.info("\nTo process this week's data:")
            logger.info("1. Download the CSV files from Seoul Open Data")
            logger.info("2. Place them in: bike_historical_data/YYYY_MM/")
            logger.info("3. Run this script again")
            return False
        
        logger.info(f"\n📁 Found {len(files)} files to process")
        
        # Process each file
        success_count = 0
        failed_files = []
        
        for i, filepath in enumerate(files, 1):
            filename = os.path.basename(filepath)
            logger.info(f"\n[{i}/{len(files)}] Processing {filename}...")
            
            try:
                if self.cleaner.process_file(filepath):
                    success_count += 1
                    logger.info(f"  ✅ Success")
                else:
                    failed_files.append(filename)
                    logger.warning(f"  ⚠️  Failed")
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                failed_files.append(filename)
        
        # Update net_flow values
        logger.info("\n📊 Updating net_flow calculations...")
        self.db.execute_query(f"""
            UPDATE station_hourly_flow 
            SET net_flow = bikes_arrived - bikes_departed
            WHERE flow_date >= '{start_date}' 
            AND flow_date <= '{end_date}'
            AND net_flow IS NULL
        """)
        
        # Generate summary
        summary = self.generate_weekly_summary(start_date, end_date)
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ WEEKLY UPDATE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Processed: {success_count}/{len(files)} files")
        
        if failed_files:
            logger.warning(f"Failed files: {', '.join(failed_files)}")
        
        return success_count == len(files)
    
    def generate_weekly_summary(self, start_date, end_date):
        """Generate summary statistics for the week"""
        summary_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT station_id) as unique_stations,
                COUNT(DISTINCT flow_date) as days_covered,
                SUM(bikes_departed) as total_departures,
                SUM(bikes_arrived) as total_arrivals,
                AVG(net_flow) as avg_net_flow,
                MIN(net_flow) as min_net_flow,
                MAX(net_flow) as max_net_flow
            FROM station_hourly_flow
            WHERE flow_date >= '{start_date}' 
            AND flow_date <= '{end_date}'
        """
        
        summary = self.db.read_query(summary_query)
        
        if not summary.empty:
            stats = summary.iloc[0]
            logger.info("\n📊 WEEKLY STATISTICS")
            logger.info("-" * 40)
            logger.info(f"Total records:     {stats['total_records']:,}")
            logger.info(f"Unique stations:   {stats['unique_stations']:,}")
            logger.info(f"Days covered:      {stats['days_covered']}")
            logger.info(f"Total departures:  {stats['total_departures']:,}")
            logger.info(f"Total arrivals:    {stats['total_arrivals']:,}")
            logger.info(f"Avg net flow:      {stats['avg_net_flow']:.2f}")
            logger.info(f"Net flow range:    {stats['min_net_flow']} to {stats['max_net_flow']}")
        
        return summary
    
    def check_data_quality(self, start_date, end_date):
        """Check data quality and completeness"""
        quality_query = f"""
            SELECT 
                flow_date,
                flow_hour,
                COUNT(DISTINCT station_id) as stations,
                AVG(bikes_departed) as avg_departures,
                AVG(bikes_arrived) as avg_arrivals,
                SUM(CASE WHEN net_flow IS NULL THEN 1 ELSE 0 END) as null_netflow
            FROM station_hourly_flow
            WHERE flow_date >= '{start_date}' 
            AND flow_date <= '{end_date}'
            GROUP BY flow_date, flow_hour
            HAVING null_netflow > 0 OR stations < 100
            ORDER BY flow_date, flow_hour
        """
        
        issues = self.db.read_query(quality_query)
        
        if not issues.empty:
            logger.warning(f"\n⚠️  Data quality issues found: {len(issues)} hours with problems")
            for _, row in issues.head(5).iterrows():
                logger.warning(f"  {row['flow_date']} {row['flow_hour']:02d}:00 - "
                             f"{row['stations']} stations, {row['null_netflow']} null net_flows")
        else:
            logger.info("\n✅ Data quality check passed")
        
        return issues

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Weekly Seoul bike data update')
    parser.add_argument('--week', default='previous', 
                       help='Week to process: "previous" or date "YYYY-MM-DD"')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check existing data, don\'t process')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = WeeklyDataUpdater()
    
    try:
        # Connect to database
        updater.connect()
        
        # Determine date range
        if args.start and args.end:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        else:
            start_date, end_date = updater.get_date_range(args.week)
        
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        if args.check_only:
            # Just check existing data
            updater.check_existing_data(start_date, end_date)
            updater.check_data_quality(start_date, end_date)
        else:
            # Process the week's data
            success = updater.process_week_data(start_date, end_date)
            
            # Check data quality
            updater.check_data_quality(start_date, end_date)
            
            if success:
                logger.info("\n✅ Weekly update successful!")
                logger.info("\nNext steps:")
                logger.info("1. Restart prediction API if running")
                logger.info("2. Monitor prediction accuracy")
                logger.info("3. Schedule next week's update")
            else:
                logger.error("\n❌ Weekly update had errors")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()