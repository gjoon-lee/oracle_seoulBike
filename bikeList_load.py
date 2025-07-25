import requests
import os
import json
import time
import datetime
import sqlite3
import schedule
import logging
from dotenv import load_dotenv

# Loading API key
load_dotenv()
API_KEY_LIST = os.getenv("KEY_BIKE_LIST")

# Setting up basic logging
logging.basicConfig(
    filename="bike_fetch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_bikeList_data():
    """Function to fetch bikeList data for all stations"""
    all_stations = []
    batch_size = 999
    start = 1
    keep_going = True

    # Fetching all stations
    while keep_going:

        logging.info(f"Fetching stations {start} to {start + batch_size}")

        # Make HTTP request and add it to all_stations
        url = f'http://openapi.seoul.go.kr:8088/{API_KEY_LIST}/json/bikeList/{start}/{start+999}/' 
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logging.error(f"❌ API error at batch {start}: {e}")
            return
        
        size_response = data['rentBikeStatus']['list_total_count']
        all_stations.extend(data['rentBikeStatus']['row'])
        
        # Check how many stations are obtained
        if size_response < batch_size:
            keep_going = False
        else:
            start+=batch_size+1
        
        # Adding 0.5s delay between api requests
        time.sleep(0.5)

    # Saving data to seoul_bike.db
    conn = sqlite3.connect('seoul_bike.db')
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="minutes")

    for station in all_stations:
        cursor.execute(
            '''
            INSERT INTO bike_availability 
            (station_id, timestamp, available_bikes, available_racks)
            VALUES (?, ?, ?, ?)
            ''', 
            (
                station['stationId'],
                timestamp,
                int(station['parkingBikeTotCnt']),
                int(station['rackTotCnt']) - int(station['parkingBikeTotCnt'])
            )
        )
    conn.commit()
    check_data_health(timestamp)
    conn.close()
    logging.info(f"✅ Inserted {len(all_stations)} stations at {timestamp}")

def check_data_health(timestamp):
    """Function to check basic data health"""

    conn = sqlite3.connect('seoul_bike.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(DISTINCT station_id)
        FROM bike_availability
        WHERE timestamp = ?
    """, (timestamp,))
    count = cursor.fetchone()[0]
    conn.close()

    if count >= 2700:
            logging.info(f"Data health OK: {count} stations stored for {timestamp}")
    else:
        logging.warning(f"Data health issue: only {count} stations collected at {timestamp}")

# Schedule fetching cycle every 5 minutes
schedule.every(5).minutes.do(fetch_bikeList_data)


if __name__ == "__main__":
    logging.info("📡 Bike data collection started...")
    fetch_bikeList_data()  

    while True:
        schedule.run_pending()
        time.sleep(1)