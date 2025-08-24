"""
Real-time weather data collector from Korea Meteorological Administration API
Falls back to OpenWeatherMap if KMA API is unavailable
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

logger = logging.getLogger(__name__)

class RealtimeWeatherCollector:
    """Fetches real-time weather data for Seoul"""
    
    def __init__(self):
        self.kma_api_key = Config.WEATHER_API_KEY
        self.kma_base_url = Config.WEATHER_API_BASE_URL
        self.station_code = Config.WEATHER_STATION_CODE
        
        # Fallback: OpenWeatherMap (free tier)
        self.owm_api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.owm_base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.seoul_coords = {"lat": 37.5665, "lon": 126.9780}
        
        # Cache for recent weather data
        self.weather_cache = []
        self.cache_hours = 24
        
    def fetch_kma_weather(self) -> Optional[Dict]:
        """Fetch weather from Korea Meteorological Administration"""
        try:
            # Format current date and hour
            now = datetime.now()
            base_date = now.strftime("%Y%m%d")
            base_time = now.strftime("%H00")
            
            params = {
                "serviceKey": self.kma_api_key,
                "numOfRows": 1,
                "pageNo": 1,
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": base_date,
                "startHh": base_time[:2],
                "endDt": base_date,
                "endHh": base_time[:2],
                "stnIds": self.station_code
            }
            
            response = requests.get(
                f"{self.kma_base_url}/getWthrDataList",
                params=params,
                timeout=Config.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                
                if items:
                    weather = items[0]
                    return self._parse_kma_data(weather)
            
            logger.warning("KMA API returned no data, falling back to OpenWeatherMap")
            return None
            
        except Exception as e:
            logger.error(f"KMA API error: {e}")
            return None
    
    def fetch_openweather(self) -> Optional[Dict]:
        """Fallback: Fetch weather from OpenWeatherMap"""
        if not self.owm_api_key:
            logger.warning("No OpenWeatherMap API key configured")
            return None
        
        try:
            params = {
                "lat": self.seoul_coords["lat"],
                "lon": self.seoul_coords["lon"],
                "appid": self.owm_api_key,
                "units": "metric"
            }
            
            response = requests.get(self.owm_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_openweather_data(data)
            
            logger.error(f"OpenWeatherMap API error: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"OpenWeatherMap error: {e}")
            return None
    
    def _parse_kma_data(self, data: Dict) -> Dict:
        """Parse KMA weather data"""
        try:
            # Extract and convert weather values
            temp = float(data.get("ta", 0))  # 기온
            humidity = float(data.get("hm", 0))  # 습도
            precipitation = float(data.get("rn", 0) or 0)  # 강수량
            wind_speed = float(data.get("ws", 0))  # 풍속
            
            # Calculate derived features
            feels_like = self._calculate_feels_like(temp, humidity, wind_speed)
            is_raining = 1 if precipitation > 0 else 0
            is_snowing = 1 if temp < 0 and precipitation > 0 else 0
            weather_severity = self._calculate_severity(precipitation, wind_speed, temp)
            
            return {
                "temperature": temp,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "feels_like": feels_like,
                "is_raining": is_raining,
                "is_snowing": is_snowing,
                "weather_severity": weather_severity,
                "timestamp": datetime.now(),
                "source": "KMA"
            }
        except Exception as e:
            logger.error(f"Error parsing KMA data: {e}")
            return None
    
    def _parse_openweather_data(self, data: Dict) -> Dict:
        """Parse OpenWeatherMap data"""
        try:
            main = data.get("main", {})
            wind = data.get("wind", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})
            
            temp = float(main.get("temp", 0))
            humidity = float(main.get("humidity", 0))
            wind_speed = float(wind.get("speed", 0))
            
            # Get precipitation (1h or 3h)
            precipitation = float(rain.get("1h", rain.get("3h", 0))) if rain else 0
            precipitation += float(snow.get("1h", snow.get("3h", 0))) if snow else 0
            
            feels_like = float(main.get("feels_like", temp))
            is_raining = 1 if rain else 0
            is_snowing = 1 if snow else 0
            weather_severity = self._calculate_severity(precipitation, wind_speed, temp)
            
            return {
                "temperature": temp,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "feels_like": feels_like,
                "is_raining": is_raining,
                "is_snowing": is_snowing,
                "weather_severity": weather_severity,
                "timestamp": datetime.now(),
                "source": "OpenWeatherMap"
            }
        except Exception as e:
            logger.error(f"Error parsing OpenWeather data: {e}")
            return None
    
    def _calculate_feels_like(self, temp: float, humidity: float, wind_speed: float) -> float:
        """Calculate feels-like temperature"""
        # Simple heat index / wind chill calculation
        if temp >= 27:  # Heat index
            feels_like = temp + 0.5 * (humidity - 40)
        elif temp <= 10 and wind_speed > 1:  # Wind chill
            feels_like = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        else:
            feels_like = temp
        
        return round(feels_like, 1)
    
    def _calculate_severity(self, precipitation: float, wind_speed: float, temp: float) -> int:
        """Calculate weather severity (0-3 scale)"""
        severity = 0
        
        # Precipitation severity
        if precipitation > 10:
            severity = max(severity, 3)
        elif precipitation > 5:
            severity = max(severity, 2)
        elif precipitation > 1:
            severity = max(severity, 1)
        
        # Wind severity
        if wind_speed > 10:
            severity = max(severity, 2)
        elif wind_speed > 5:
            severity = max(severity, 1)
        
        # Temperature severity
        if temp < -5 or temp > 35:
            severity = max(severity, 2)
        elif temp < 0 or temp > 30:
            severity = max(severity, 1)
        
        return severity
    
    def fetch_current_weather(self) -> Dict:
        """Fetch current weather from available sources"""
        # Try KMA first
        weather = self.fetch_kma_weather()
        
        # Fallback to OpenWeatherMap
        if not weather:
            weather = self.fetch_openweather()
        
        # Last resort: use cached or default values
        if not weather:
            weather = self.get_cached_or_default()
        
        # Add to cache
        if weather:
            self.weather_cache.append(weather)
            # Keep only recent hours
            cutoff_time = datetime.now() - timedelta(hours=self.cache_hours)
            self.weather_cache = [w for w in self.weather_cache 
                                 if w['timestamp'] > cutoff_time]
        
        return weather
    
    def get_cached_or_default(self) -> Dict:
        """Get most recent cached weather or default values"""
        if self.weather_cache:
            logger.info("Using cached weather data")
            return self.weather_cache[-1].copy()
        
        logger.warning("Using default weather values")
        return {
            "temperature": 28.0,  # Seoul summer (August) average
            "humidity": 65.0,     # Higher humidity in summer
            "precipitation": 0.0,
            "wind_speed": 2.5,
            "feels_like": 31.0,   # Feels hotter due to humidity
            "is_raining": 0,
            "is_snowing": 0,
            "weather_severity": 0,
            "timestamp": datetime.now(),
            "source": "default"
        }
    
    def get_weather_history(self, hours: int = 24) -> pd.DataFrame:
        """Get weather history from cache"""
        if not self.weather_cache:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_weather = [w for w in self.weather_cache if w['timestamp'] > cutoff_time]
        
        if recent_weather:
            return pd.DataFrame(recent_weather)
        return pd.DataFrame()
    
    def calculate_rolling_features(self, hours_list: List[int] = [6, 12, 24, 168]) -> Dict:
        """Calculate rolling weather features"""
        df = self.get_weather_history(max(hours_list))
        
        if df.empty:
            return {f"temperature_roll_mean_{h}h": None for h in hours_list}
        
        features = {}
        for hours in hours_list:
            cutoff = datetime.now() - timedelta(hours=hours)
            recent = df[df['timestamp'] > cutoff]
            
            if not recent.empty:
                features[f"temperature_roll_mean_{hours}h"] = recent['temperature'].mean()
                features[f"feels_like_roll_mean_{hours}h"] = recent['feels_like'].mean()
            else:
                features[f"temperature_roll_mean_{hours}h"] = None
                features[f"feels_like_roll_mean_{hours}h"] = None
        
        return features


if __name__ == "__main__":
    # Test the weather collector
    logging.basicConfig(level=logging.INFO)
    
    collector = RealtimeWeatherCollector()
    
    print("Fetching current weather...")
    weather = collector.fetch_current_weather()
    
    if weather:
        print(f"\nWeather source: {weather['source']}")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Feels like: {weather['feels_like']}°C")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Precipitation: {weather['precipitation']}mm")
        print(f"Wind speed: {weather['wind_speed']}m/s")
        print(f"Weather severity: {weather['weather_severity']}/3")
        print(f"Raining: {'Yes' if weather['is_raining'] else 'No'}")
        print(f"Snowing: {'Yes' if weather['is_snowing'] else 'No'}")
    else:
        print("Failed to fetch weather data")