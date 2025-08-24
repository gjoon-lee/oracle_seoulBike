"""
API Client for connecting to FastAPI backend
"""

import requests
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BikeAPIClient:
    """Client for interacting with the bike prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to API at {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_stations_status(self, district: Optional[str] = None) -> Optional[Dict]:
        """Get current status of all stations with pagination"""
        all_stations = []
        offset = 0
        limit = 1000  # Max per request
        
        # Fetch all stations with pagination (up to 3 calls for ~2.8k stations)
        while offset < 3000:  # Safety limit
            params = {'limit': limit, 'offset': offset}
            if district:
                params['district'] = district
            
            response = self._make_request('GET', '/stations/status', params=params)
            
            if not response or 'stations' not in response:
                break
                
            all_stations.extend(response['stations'])
            
            # Check if we got all stations
            total_count = response.get('total_count', 0)
            if offset + limit >= total_count:
                break
                
            offset += limit
        
        if all_stations:
            return {
                'stations': all_stations,
                'count': len(all_stations),
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def get_station_detail(self, station_id: str) -> Optional[Dict]:
        """Get detailed information for a specific station"""
        return self._make_request('GET', f'/predict/{station_id}')
    
    def get_predictions(self, stations: Optional[List[str]] = None) -> Optional[Dict]:
        """Get stockout predictions for stations"""
        # Use /predict/all endpoint for predictions
        return self._make_request('GET', '/predict/all')
    
    def get_high_risk_stations(self, threshold: float = 0.5) -> Optional[Dict]:
        """Get stations with high stockout risk"""
        params = {'threshold': threshold}
        return self._make_request('GET', '/high-risk', params=params)
    
    def get_demand_forecast(self, station_id: str, hours: int = 24) -> Optional[Dict]:
        """Get demand forecast for a specific station"""
        params = {'hours': hours}
        return self._make_request('GET', f'/api/predictions/demand/{station_id}', params=params)
    
    def get_current_weather(self) -> Optional[Dict]:
        """Get current weather data"""
        return self._make_request('GET', '/weather/current')
    
    def get_statistics(self, period: str = 'day') -> Optional[Dict]:
        """Get system statistics"""
        params = {'period': period}
        return self._make_request('GET', '/api/statistics', params=params)
    
    def get_weekly_report(self) -> Optional[Dict]:
        """Get weekly report data"""
        return self._make_request('GET', '/api/reports/weekly')
    
    def trigger_model_update(self, model_type: str = 'lightgbm') -> Optional[Dict]:
        """Trigger model retraining"""
        data = {'model_type': model_type}
        return self._make_request('POST', '/api/models/retrain', json=data)
    
    def get_system_health(self) -> Optional[Dict]:
        """Get system health status"""
        return self._make_request('GET', '/health')
    
    def search_stations(self, query: str) -> Optional[List[Dict]]:
        """Search for stations by name or ID"""
        params = {'q': query}
        return self._make_request('GET', '/api/stations/search', params=params)
    
    def get_realtime_availability(self) -> Optional[Dict]:
        """Get real-time availability from Seoul Open API"""
        return self._make_request('GET', '/api/realtime/availability')
    
    def get_historical_data(self, station_id: str, days: int = 7) -> Optional[Dict]:
        """Get historical data for a station"""
        params = {'days': days}
        return self._make_request('GET', f'/api/stations/{station_id}/history', params=params)
    
    def export_data(self, data_type: str, format: str = 'csv') -> Optional[bytes]:
        """Export data in specified format"""
        params = {'format': format}
        url = f"{self.base_url}/api/export/{data_type}"
        
        try:
            response = self.session.get(url, params=params, stream=True)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def get_alerts(self, level: str = 'all') -> Optional[List[Dict]]:
        """Get current system alerts"""
        params = {'level': level}
        return self._make_request('GET', '/api/alerts', params=params)
    
    def acknowledge_alert(self, alert_id: str) -> Optional[Dict]:
        """Acknowledge an alert"""
        return self._make_request('POST', f'/api/alerts/{alert_id}/acknowledge')
    
    def get_performance_metrics(self) -> Optional[Dict]:
        """Get model performance metrics"""
        return self._make_request('GET', '/api/models/metrics')
    
    def update_station_capacity(self, station_id: str, new_capacity: int) -> Optional[Dict]:
        """Update station capacity"""
        data = {'capacity': new_capacity}
        return self._make_request('PUT', f'/api/stations/{station_id}/capacity', json=data)
    
    def get_redistribution_suggestions(self) -> Optional[List[Dict]]:
        """Get bike redistribution suggestions"""
        return self._make_request('GET', '/api/operations/redistribution')
    
    def mark_station_maintenance(self, station_id: str, status: bool) -> Optional[Dict]:
        """Mark station for maintenance"""
        data = {'under_maintenance': status}
        return self._make_request('PUT', f'/api/stations/{station_id}/maintenance', json=data)