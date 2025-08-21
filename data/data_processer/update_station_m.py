# update_station_master.py
"""
Update station master data from API
Run this manually when new stations are added (maybe once a year)
"""

import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import text

def update_station_master(db_connection):
    """Fetch latest station data from API and update DB"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    load_dotenv()
    API_KEY = os.getenv("KEY_BIKE_STATION_MASTER")
    
    if not API_KEY:
        logger.error("No API key found!")
        return False
    
    logger.info(f"Starting station master update at {datetime.now()}")
    
    # Create table and index together
    create_sql = text("""
    CREATE TABLE IF NOT EXISTS station_master (
        station_id VARCHAR(20) PRIMARY KEY,
        station_name VARCHAR(200),
        station_address VARCHAR(500),
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        lat_5dp VARCHAR(10),
        lon_5dp VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_station_coords ON station_master(lat_5dp, lon_5dp);
    """)
    
    with db_connection.engine.connect() as conn:
        conn.execute(create_sql)
        conn.commit()
    
    # Fetch all stations
    all_stations = []
    start = 1
    batch_size = 1000
    
    while True:
        url = f'http://openapi.seoul.go.kr:8088/{API_KEY}/json/bikeStationMaster/{start}/{start + batch_size - 1}/'
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'bikeStationMaster' not in data:
                break
                
            stations = data['bikeStationMaster']['row']
            all_stations.extend(stations)
            
            total = data['bikeStationMaster']['list_total_count']
            logger.info(f"Fetched {len(all_stations)}/{total} stations...")
            
            if start + batch_size > total:
                break
                
            start += batch_size
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"API error: {e}")
            break
    
    # Process and save
    if all_stations:
        df = pd.DataFrame(all_stations)
        logger.info(f"API response columns: {list(df.columns)}")
        logger.info(f"Sample row: {df.iloc[0].to_dict()}")
        
        # Prepare data
        for _, row in df.iterrows():
            lat = float(row['LAT'])
            lon = float(row['LOT'])
            
            sql = text("""
            INSERT INTO station_master 
            (station_id, station_name, station_address, latitude, longitude, lat_5dp, lon_5dp)
            VALUES (:station_id, :station_name, :station_address, :latitude, :longitude, :lat_5dp, :lon_5dp)
            ON CONFLICT (station_id) DO UPDATE SET
                station_name = EXCLUDED.station_name,
                station_address = EXCLUDED.station_address,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                lat_5dp = EXCLUDED.lat_5dp,
                lon_5dp = EXCLUDED.lon_5dp,
                updated_at = CURRENT_TIMESTAMP
            """)
            
            # Use ADDR2 as station name if available, otherwise use station_id
            station_name = row.get('ADDR2', row['RNTLS_ID']) if row.get('ADDR2') else row['RNTLS_ID']
            
            db_connection.execute_query(sql, {
                'station_id': row['RNTLS_ID'],
                'station_name': station_name,
                'station_address': row['ADDR1'],
                'latitude': lat,
                'longitude': lon,
                'lat_5dp': f"{lat:.5f}",
                'lon_5dp': f"{lon:.5f}"
            })
        
        logger.info(f"✅ Updated {len(all_stations)} stations in DB")
        
        # Show summary
        summary = db_connection.read_query("""
            SELECT 
                COUNT(*) as total_stations,
                COUNT(CASE WHEN updated_at > created_at THEN 1 END) as updated_stations,
                COUNT(CASE WHEN DATE(created_at) = CURRENT_DATE THEN 1 END) as new_stations
            FROM station_master
        """)
        
        logger.info(f"Summary: {summary.iloc[0].to_dict()}")
        return True
    
    return False


if __name__ == "__main__":
    from db_connection import BikeDataDB
    
    db = BikeDataDB()
    db.connect()
    
    update_station_master(db)