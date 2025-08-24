"""
Utility functions for Streamlit dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        "api": {
            "base_url": "http://localhost:8000",
            "timeout": 30,
            "retry_count": 3
        },
        "dashboard": {
            "refresh_interval": 300,  # 5 minutes
            "max_stations_display": 20,
            "chart_height": 400,
            "map_zoom": 11
        },
        "thresholds": {
            "stockout_warning": 0.5,
            "stockout_critical": 0.8,
            "utilization_high": 0.8,
            "utilization_low": 0.2
        },
        "districts": [
            "강남구", "강동구", "강북구", "강서구", "관악구", "광진구",
            "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구",
            "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
            "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
        ]
    }

def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dashboard metrics from station data"""
    
    metrics = {
        "total_stations": len(df),
        "empty_stations": 0,
        "warning_stations": 0,
        "avg_utilization": 0,
        "total_bikes": 0,
        "total_capacity": 0
    }
    
    if len(df) == 0:
        return metrics
    
    # Calculate metrics
    if 'is_stockout' in df.columns:
        metrics['empty_stations'] = len(df[df['is_stockout'] == 1])
    
    if 'utilization_rate' in df.columns:
        metrics['avg_utilization'] = df['utilization_rate'].mean() * 100
        metrics['warning_stations'] = len(df[df['utilization_rate'] > 0.8])
    
    if 'available_bikes' in df.columns:
        metrics['total_bikes'] = df['available_bikes'].sum()
    
    if 'station_capacity' in df.columns:
        metrics['total_capacity'] = df['station_capacity'].sum()
    
    # Calculate percentages
    if metrics['total_stations'] > 0:
        metrics['empty_percentage'] = (metrics['empty_stations'] / metrics['total_stations']) * 100
        metrics['warning_percentage'] = (metrics['warning_stations'] / metrics['total_stations']) * 100
    
    return metrics

def filter_stations(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to station dataframe"""
    
    filtered_df = df.copy()
    
    # District filter
    if 'district' in filters and filters['district'] != "전체 구":
        if 'district' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['district'] == filters['district']]
    
    # Status filter
    if 'status' in filters:
        if filters['status'] == "비어있음" and 'is_stockout' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_stockout'] == 1]
        elif filters['status'] == "경고" and 'utilization_rate' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['utilization_rate'] > 0.8]
        elif filters['status'] == "정상":
            if 'is_stockout' in filtered_df.columns and 'utilization_rate' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['is_stockout'] == 0) & 
                    (filtered_df['utilization_rate'] <= 0.8)
                ]
    
    # Search filter
    if 'search' in filters and filters['search']:
        search_term = filters['search'].lower()
        if 'station_id' in filtered_df.columns and 'station_name' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['station_id'].str.lower().str.contains(search_term, na=False)) |
                (filtered_df['station_name'].str.lower().str.contains(search_term, na=False))
            ]
    
    return filtered_df

def format_timedelta(td: timedelta) -> str:
    """Format timedelta to Korean string"""
    
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    if hours > 0:
        return f"{hours}시간 {minutes}분"
    else:
        return f"{minutes}분"

def get_time_category(probability: float) -> Tuple[str, str]:
    """Get time category and color based on stockout probability"""
    
    if probability < 0.3:
        return "1시간 미만", "#FFF59D"
    elif probability < 0.5:
        return "1-2시간", "#FFB74D"
    elif probability < 0.7:
        return "2-3시간", "#FF7043"
    else:
        return "3시간 이상", "#EF5350"

def generate_sample_data(num_stations: int = 100) -> pd.DataFrame:
    """Generate sample station data for testing"""
    
    np.random.seed(42)
    
    stations = []
    for i in range(1, num_stations + 1):
        station = {
            'station_id': f'ST-{i:03d}',
            'station_name': f'대여소 {i}',
            'district': np.random.choice([
                "강남구", "강동구", "강북구", "강서구", "관악구"
            ]),
            'available_bikes': np.random.randint(0, 20),
            'station_capacity': 20,
            'utilization_rate': np.random.uniform(0, 1),
            'is_stockout': 0,
            'stockout_probability': np.random.uniform(0, 1)
        }
        
        # Set stockout flag
        if station['available_bikes'] <= 2:
            station['is_stockout'] = 1
        
        # Calculate available racks
        station['available_racks'] = station['station_capacity'] - station['available_bikes']
        
        stations.append(station)
    
    return pd.DataFrame(stations)

def calculate_redistribution_needs(df: pd.DataFrame) -> List[Dict]:
    """Calculate bike redistribution recommendations"""
    
    recommendations = []
    
    # Find empty stations
    empty_stations = df[df['is_stockout'] == 1].copy()
    
    # Find overfull stations
    if 'utilization_rate' in df.columns:
        full_stations = df[df['utilization_rate'] < 0.2].copy()
    else:
        full_stations = pd.DataFrame()
    
    # Create redistribution pairs
    for _, empty in empty_stations.iterrows():
        if len(full_stations) > 0:
            # Find nearest full station (simplified - would use actual coordinates)
            nearest_full = full_stations.iloc[0]
            
            recommendations.append({
                'from_station': nearest_full['station_id'],
                'from_name': nearest_full.get('station_name', ''),
                'to_station': empty['station_id'],
                'to_name': empty.get('station_name', ''),
                'bikes_to_move': min(5, nearest_full.get('available_bikes', 5)),
                'priority': 'high' if empty.get('stockout_probability', 0) > 0.8 else 'medium'
            })
            
            # Remove used station
            full_stations = full_stations.iloc[1:]
            
            if len(recommendations) >= 10:  # Limit recommendations
                break
    
    return recommendations

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string"""
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{decimals}f}%"

def get_status_icon(status: str) -> str:
    """Get icon for status"""
    icons = {
        "정상": "🟢",
        "주의": "🟡",
        "경고": "🟠",
        "위험": "🔴",
        "비어있음": "🔴",
        "오프라인": "⚫"
    }
    return icons.get(status, "⚪")

def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string to datetime object"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, return current time
    logger.warning(f"Could not parse datetime: {dt_str}")
    return datetime.now()

def create_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Create download link for dataframe"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    return f'<a href="data:text/csv;charset=utf-8,{csv}" download="{filename}">📥 다운로드</a>'

def validate_station_id(station_id: str) -> bool:
    """Validate station ID format"""
    import re
    pattern = r'^ST-\d{3,4}$'
    return bool(re.match(pattern, station_id))

def get_peak_hours() -> List[int]:
    """Get peak usage hours"""
    return [7, 8, 9, 18, 19, 20]  # Morning and evening rush hours

def is_peak_time(hour: int = None) -> bool:
    """Check if current time is peak hour"""
    if hour is None:
        hour = datetime.now().hour
    return hour in get_peak_hours()

def calculate_trend(current: float, previous: float) -> Tuple[str, float]:
    """Calculate trend direction and percentage"""
    if previous == 0:
        return "→", 0
    
    change = ((current - previous) / previous) * 100
    
    if change > 5:
        return "↑", change
    elif change < -5:
        return "↓", change
    else:
        return "→", change

def group_stations_by_risk(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group stations by risk level"""
    
    groups = {
        "critical": pd.DataFrame(),
        "high": pd.DataFrame(),
        "medium": pd.DataFrame(),
        "low": pd.DataFrame()
    }
    
    if 'stockout_probability' not in df.columns:
        return groups
    
    groups['critical'] = df[df['stockout_probability'] >= 0.8]
    groups['high'] = df[(df['stockout_probability'] >= 0.6) & (df['stockout_probability'] < 0.8)]
    groups['medium'] = df[(df['stockout_probability'] >= 0.4) & (df['stockout_probability'] < 0.6)]
    groups['low'] = df[df['stockout_probability'] < 0.4]
    
    return groups

def calculate_service_level(metrics: Dict) -> float:
    """Calculate overall service level score (0-100)"""
    
    score = 100
    
    # Deduct points for empty stations
    if 'empty_percentage' in metrics:
        score -= min(30, metrics['empty_percentage'] * 2)
    
    # Deduct points for warning stations
    if 'warning_percentage' in metrics:
        score -= min(20, metrics['warning_percentage'])
    
    # Bonus for good utilization
    if 'avg_utilization' in metrics:
        if 30 <= metrics['avg_utilization'] <= 70:
            score += 10
    
    return max(0, min(100, score))