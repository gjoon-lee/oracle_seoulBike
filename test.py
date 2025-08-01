"""
Test script for the Seoul Bike cleaning pipeline
Run this to process your June data files
"""

from db_connection import BikeDataDB
from bike_data_cleaner import BikeDataCleaner
import os

def test_single_file():
    """Test with one file first"""
    print("🚴 Seoul Bike Data Cleaning Pipeline Test")
    print("="*50)
    
    # 1. Connect to database
    print("\n1. Connecting to PostgreSQL...")
    db = BikeDataDB()
    db.connect()
    
    if not db.test_connection():
        print("❌ Database connection failed!")
        return
    
    # 2. Initialize cleaner
    print("\n2. Initializing data cleaner...")
    cleaner = BikeDataCleaner(db)
    
    # 3. Process June 1 file
    print("\n3. Processing June 1, 2025 data...")
    success = cleaner.process_file('bike_historical_data/2025_06/tpss_bcycl_od_statnhm_20250601.csv')
    
    if success:
        print("\n✅ File processed successfully!")
        
        # 4. Check what was loaded
        print("\n4. Checking loaded data...")
        
        # Check raw trips
        raw_count = db.read_query("""
            SELECT COUNT(*) as count, 
                   MIN(record_date) as first_date,
                   MAX(record_date) as last_date
            FROM raw_bike_trips
        """)
        print(f"\nRaw trips table: {raw_count.iloc[0]['count']:,} records")
        
        # Check hourly flow
        flow_count = db.read_query("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT station_id) as stations,
                   SUM(bikes_departed) as total_departures,
                   SUM(bikes_arrived) as total_arrivals
            FROM station_hourly_flow
        """)
        print(f"\nHourly flow table:")
        print(f"  - Records: {flow_count.iloc[0]['count']:,}")
        print(f"  - Stations: {flow_count.iloc[0]['stations']:,}")
        print(f"  - Total departures: {flow_count.iloc[0]['total_departures']:,}")
        print(f"  - Total arrivals: {flow_count.iloc[0]['total_arrivals']:,}")
        
        # 5. Sample analysis
        print("\n5. Sample analysis - Top 10 busiest stations:")
        busy_stations = db.read_query("""
            SELECT 
                station_id,
                SUM(bikes_departed + bikes_arrived) as total_activity,
                AVG(bikes_departed) as avg_hourly_departures,
                AVG(bikes_arrived) as avg_hourly_arrivals
            FROM station_hourly_flow
            GROUP BY station_id
            ORDER BY total_activity DESC
            LIMIT 10
        """)
        print(busy_stations)
        
        # 6. Hourly pattern
        print("\n6. Hourly usage pattern:")
        hourly_pattern = db.read_query("""
            SELECT 
                flow_hour,
                SUM(bikes_departed) as departures,
                SUM(bikes_arrived) as arrivals,
                SUM(bikes_departed) - SUM(bikes_arrived) as net_flow
            FROM station_hourly_flow
            GROUP BY flow_hour
            ORDER BY flow_hour
        """)
        print(hourly_pattern.head(10))

def process_week_data():
    """Process a full week of data"""
    print("\n\n🚴 Processing Full Week (June 1-7, 2025)")
    print("="*50)
    
    db = BikeDataDB()
    db.connect()
    cleaner = BikeDataCleaner(db)
    
    # List of files to process
    files = [
        'tpss_bcycl_od_statnhm_20250601.csv',
        'tpss_bcycl_od_statnhm_20250602.csv',
        'tpss_bcycl_od_statnhm_20250603.csv',
        'tpss_bcycl_od_statnhm_20250604.csv',
        'tpss_bcycl_od_statnhm_20250605.csv',
        'tpss_bcycl_od_statnhm_20250606.csv',
        'tpss_bcycl_od_statnhm_20250607.csv',
    ]
    
    # Process each file
    for file in files:
        if os.path.exists(file):
            print(f"\nProcessing {file}...")
            cleaner.process_file(file)
        else:
            print(f"\n⚠️  File not found: {file}")
    
    # Summary
    summary = cleaner.get_processing_summary()
    print("\n\nProcessing Summary:")
    print(summary)

if __name__ == "__main__":
    # First test with single file
    test_single_file()
    
    # Then process full week (uncomment when ready)
    # process_week_data()