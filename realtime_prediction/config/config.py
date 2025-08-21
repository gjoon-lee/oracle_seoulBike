"""
Configuration management for real-time prediction system
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the prediction system"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "cache"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Create directories if they don't exist
    CACHE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # API Keys
    BIKE_API_KEY = os.getenv("KEY_BIKE_LIST")
    WEATHER_API_KEY = os.getenv("KEY_WEATHER_API", "")  # KMA API key if required
    
    # Seoul Bike API Configuration
    BIKE_API_BASE_URL = "http://openapi.seoul.go.kr:8088"
    BIKE_API_SERVICE = "bikeList"
    BIKE_API_MAX_RECORDS = 1000  # Max records per request
    
    # Weather API Configuration (KMA)
    WEATHER_API_BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService"
    WEATHER_STATION_CODE = "108"  # Seoul weather station
    
    # Database Configuration
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "bike_data"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    }
    
    # Model Configuration
    MODEL_PATH = MODELS_DIR / "lightgbm_stockout_model_20250819_072922.pkl"
    MODEL_THRESHOLDS_PATH = MODELS_DIR / "model_thresholds_20250819_072922.json"
    
    # Load model thresholds
    @classmethod
    def load_model_thresholds(cls) -> Dict[str, Any]:
        """Load model threshold configuration"""
        with open(cls.MODEL_THRESHOLDS_PATH, 'r') as f:
            return json.load(f)
    
    # Prediction Configuration
    PREDICTION_MODES = {
        "alert": 0.70,  # High recall mode
        "balanced": 0.65  # Balanced mode
    }
    DEFAULT_PREDICTION_MODE = "balanced"
    
    # Cache Configuration
    CACHE_TTL_SECONDS = 300  # 5 minutes
    HISTORICAL_WINDOW_HOURS = 168  # 7 days for lag features
    CACHE_MAX_SIZE_MB = 500  # Maximum cache size in MB
    
    # Scheduler Configuration
    SCHEDULE_INTERVALS = {
        "bike_data": 5 * 60,  # 5 minutes in seconds
        "weather_data": 30 * 60,  # 30 minutes
        "historical_refresh": 60 * 60,  # 1 hour
        "prediction_update": 5 * 60,  # 5 minutes
        "station_profile": 24 * 60 * 60,  # Daily
        "cache_cleanup": 6 * 60 * 60  # Every 6 hours
    }
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "Seoul Bike Stockout Prediction API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Real-time stockout predictions for Seoul bike sharing system"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "prediction_api.log"
    
    # Feature Engineering Configuration
    LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
    ROLLING_WINDOWS = [6, 12, 24, 168]
    
    # Station Mapping
    STATION_ID_PREFIX = "ST-"
    
    # Performance Configuration
    BATCH_SIZE = 100  # Batch size for predictions
    MAX_WORKERS = 4  # Thread pool size
    REQUEST_TIMEOUT = 30  # API request timeout in seconds
    
    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    FALLBACK_TO_CACHE = True
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get PostgreSQL connection URL"""
        cfg = cls.DB_CONFIG
        return f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration"""
        errors = []
        
        if not cls.BIKE_API_KEY:
            errors.append("Missing BIKE_API_KEY in environment")
        
        if not cls.MODEL_PATH.exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        if not cls.MODEL_THRESHOLDS_PATH.exists():
            errors.append(f"Thresholds file not found: {cls.MODEL_THRESHOLDS_PATH}")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True

# Validate configuration on import
if not Config.validate_config():
    print("Warning: Configuration validation failed. Some features may not work.")