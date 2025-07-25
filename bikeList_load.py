import requests
import os
import json
import time
import datetime
from dotenv import load_dotenv

# Loading API key
load_dotenv()
API_KEY_LIST = os.getenv("KEY_BIKE_LIST")

all_stations = []
batch_size = 999
start = 1
keep_going = True

# Fetching all stations
while keep_going:

    print(f"Fetching stations {start} to {start+batch_size}")

    # Make HTTP request and add it to all_stations
    url = f'http://openapi.seoul.go.kr:8088/{API_KEY_LIST}/json/bikeList/{start}/{start+999}/' 
    response = requests.get(url)
    data = response.json()
    size_response = data['rentBikeStatus']['list_total_count']
    all_stations.extend(data['rentBikeStatus']['row'])
    
    # Check how many stations are obtained
    if size_response < batch_size:
        keep_going = False
    else:
        start+=batch_size
    
    # Adding 0.5s delay between api requests
    time.sleep(0.5)

print(f"Total stations collected: {len(all_stations)}")

# Saving the data after fetching API
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"bike_stations_{timestamp}.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_stations, f, ensure_ascii=False, indent=2)

print(f"Data saved to {filename}")


