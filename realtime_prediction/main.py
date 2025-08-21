"""
FastAPI application for real-time bike stockout predictions
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging
import asyncio
import uvicorn

from config.config import Config
from services.prediction_service import PredictionService
from collectors.realtime_bike_collector import RealtimeBikeCollector
from collectors.realtime_weather_collector import RealtimeWeatherCollector

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_service = PredictionService()
bike_collector = RealtimeBikeCollector()
weather_collector = RealtimeWeatherCollector()

# Cache for predictions
prediction_cache = {
    "data": None,
    "timestamp": None,
    "ttl_seconds": Config.CACHE_TTL_SECONDS
}

# Pydantic models
class PredictionResponse(BaseModel):
    station_id: str
    stockout_probability: float = Field(..., ge=0, le=1)
    is_stockout_predicted: int = Field(..., ge=0, le=1)
    risk_level: str
    prediction_mode: str
    timestamp: datetime
    current_status: Dict

class StationStatus(BaseModel):
    station_id: str
    station_name: str
    available_bikes: int
    station_capacity: int
    utilization_rate: float
    is_stockout: int

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    last_prediction: Optional[datetime]
    cache_status: str

# Helper functions
def get_cached_predictions():
    """Get predictions from cache if valid"""
    if prediction_cache["data"] is not None and prediction_cache["timestamp"] is not None:
        age_seconds = (datetime.now() - prediction_cache["timestamp"]).total_seconds()
        if age_seconds < prediction_cache["ttl_seconds"]:
            return prediction_cache["data"]
    return None

def update_prediction_cache(data):
    """Update prediction cache"""
    prediction_cache["data"] = data
    prediction_cache["timestamp"] = datetime.now()

# API Endpoints
@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Seoul Bike Stockout Prediction API",
        "version": Config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """System health check"""
    try:
        model_loaded = prediction_service.model is not None
        cache_valid = get_cached_predictions() is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now(),
            model_loaded=model_loaded,
            last_prediction=prediction_cache["timestamp"],
            cache_status="valid" if cache_valid else "expired"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/predict/all", tags=["predictions"])
async def predict_all_stations(
    mode: str = Query("balanced", description="Prediction mode: 'alert' or 'balanced'"),
    use_cache: bool = Query(True, description="Use cached predictions if available")
):
    """Generate predictions for all stations"""
    try:
        # Check cache first
        if use_cache:
            cached = get_cached_predictions()
            if cached is not None:
                logger.info("Returning cached predictions")
                return cached
        
        # Generate new predictions
        logger.info(f"Generating predictions for all stations (mode: {mode})")
        predictions = prediction_service.predict_all_stations(mode=mode)
        
        if predictions.empty:
            raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        # Convert to dict for JSON response
        result = {
            "predictions": predictions.to_dict(orient="records"),
            "summary": {
                "total_stations": len(predictions),
                "at_risk_stations": int(predictions["is_stockout_predicted"].sum()),
                "average_risk": float(predictions["stockout_probability"].mean()),
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Update cache
        update_prediction_cache(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{station_id}", response_model=PredictionResponse, tags=["predictions"])
async def predict_station(
    station_id: str,
    mode: str = Query("balanced", description="Prediction mode: 'alert' or 'balanced'")
):
    """Generate prediction for a specific station"""
    try:
        logger.info(f"Generating prediction for station {station_id}")
        prediction = prediction_service.predict_station(station_id, mode)
        
        if not prediction:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        
        return PredictionResponse(**prediction)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for station {station_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/high-risk", tags=["predictions"])
async def get_high_risk_stations(
    threshold: float = Query(0.7, ge=0, le=1, description="Risk threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of stations to return")
):
    """Get stations with high stockout risk"""
    try:
        logger.info(f"Getting high risk stations (threshold: {threshold})")
        high_risk = prediction_service.get_high_risk_stations(threshold)
        
        if high_risk.empty:
            return {
                "high_risk_stations": [],
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
        
        # Limit results
        high_risk = high_risk.head(limit)
        
        return {
            "high_risk_stations": high_risk.to_dict(orient="records"),
            "count": len(high_risk),
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting high risk stations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stations/status", tags=["stations"])
async def get_stations_status(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of stations")
):
    """Get current availability status for all stations"""
    try:
        logger.info("Fetching current station status")
        stations_df = bike_collector.fetch_all_stations()
        
        if stations_df.empty:
            raise HTTPException(status_code=500, detail="Failed to fetch station data")
        
        # Limit results
        stations_df = stations_df.head(limit)
        
        # Select relevant columns
        status_df = stations_df[[
            'station_id', 'station_name', 'available_bikes',
            'station_capacity', 'utilization_rate', 'is_stockout'
        ]]
        
        return {
            "stations": status_df.to_dict(orient="records"),
            "count": len(status_df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching station status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/current", tags=["weather"])
async def get_current_weather():
    """Get current weather conditions"""
    try:
        logger.info("Fetching current weather")
        weather = weather_collector.fetch_current_weather()
        
        return {
            "weather": weather,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["model"])
async def get_model_info():
    """Get model information and configuration"""
    try:
        return prediction_service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh", tags=["admin"])
async def refresh_predictions(background_tasks: BackgroundTasks):
    """Manually refresh prediction cache"""
    try:
        # Clear cache
        prediction_cache["data"] = None
        prediction_cache["timestamp"] = None
        
        # Trigger background refresh
        background_tasks.add_task(refresh_all_predictions)
        
        return {
            "message": "Refresh initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def refresh_all_predictions():
    """Background task to refresh predictions"""
    try:
        logger.info("Background refresh started")
        predictions = prediction_service.predict_all_stations()
        
        if not predictions.empty:
            result = {
                "predictions": predictions.to_dict(orient="records"),
                "summary": {
                    "total_stations": len(predictions),
                    "at_risk_stations": int(predictions["is_stockout_predicted"].sum()),
                    "average_risk": float(predictions["stockout_probability"].mean()),
                    "mode": "balanced",
                    "timestamp": datetime.now().isoformat()
                }
            }
            update_prediction_cache(result)
            logger.info("Background refresh completed")
        
    except Exception as e:
        logger.error(f"Background refresh failed: {e}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Seoul Bike Stockout Prediction API")
    
    # Validate configuration
    if not Config.validate_config():
        logger.error("Configuration validation failed")
    
    # Initial prediction generation
    try:
        await refresh_all_predictions()
    except Exception as e:
        logger.error(f"Failed to generate initial predictions: {e}")
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level=Config.LOG_LEVEL.lower()
    )