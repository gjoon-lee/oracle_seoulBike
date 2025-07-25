import requests
import os
import json
from dotenv import load_dotenv

# fetching url for bike station location
load_dotenv()
API_KEY_STATION = os.getenv("KEY_BIKE_STATION_MASTER")
url_station = f'http://openapi.seoul.go.kr:8088/{API_KEY_STATION}/json/bikeStationMaster/1/5/'

# Print sample response for bikeStationMaster API request
response = requests.get(url_station)
if response.status_code == 200:
    data = response.json()
    print("Printing json response:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

print("\n")
print()


# fetching url for bike list and rent status
API_KEY_LIST = os.getenv("KEY_BIKE_LIST")
url_list = f'http://openapi.seoul.go.kr:8088/{API_KEY_LIST}/json/bikeList/1/5/' # note: can only call up to 1000 stations in one request

# Print response for bikeList API request
response = requests.get(url_list)
if response.status_code == 200:
    data = response.json()
    print("Printing json response:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
