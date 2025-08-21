"""
Prediction service using pre-trained LightGBM model
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from features.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for generating stockout predictions"""
    
    def __init__(self):
        self.model = None
        self.thresholds = None
        self.feature_generator = FeatureGenerator()
        self.load_model()
        
    def load_model(self):
        """Load pre-trained LightGBM model and thresholds"""
        try:
            # Load model
            logger.info(f"Loading model from {Config.MODEL_PATH}")
            self.model = joblib.load(Config.MODEL_PATH)
            
            # Load thresholds
            with open(Config.MODEL_THRESHOLDS_PATH, 'r') as f:
                self.thresholds = json.load(f)
            
            logger.info("Model and thresholds loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_all_stations(self, mode: str = "balanced") -> pd.DataFrame:
        """Generate predictions for all stations"""
        
        # Generate features
        logger.info("Generating features for all stations...")
        features_df = self.feature_generator.generate_features()
        
        if features_df.empty:
            logger.error("No features generated")
            return pd.DataFrame()
        
        # Store station IDs (not used in prediction)
        station_ids = features_df.index if 'station_id' not in features_df.columns else features_df['station_id']
        
        # Get feature columns for prediction
        feature_cols = self.thresholds['model_info']['features']
        X = features_df[feature_cols]
        
        # Generate predictions
        logger.info(f"Generating predictions for {len(X)} stations...")
        # Handle both sklearn and native booster models
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            # Native LightGBM Booster
            probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Apply threshold based on mode
        threshold = self.get_threshold(mode)
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'station_id': station_ids,
            'stockout_probability': probabilities,
            'is_stockout_predicted': predictions,
            'risk_level': self.calculate_risk_levels(probabilities),
            'prediction_mode': mode,
            'threshold': threshold,
            'timestamp': datetime.now()
        })
        
        # Add current availability info
        if 'available_bikes' in features_df.columns:
            results['current_available_bikes'] = features_df['available_bikes'].values
        if 'station_capacity' in features_df.columns:
            results['station_capacity'] = features_df['station_capacity'].values
        if 'utilization_rate' in features_df.columns:
            results['current_utilization'] = features_df['utilization_rate'].values
        
        logger.info(f"Predictions complete: {predictions.sum()} stations at risk")
        
        return results
    
    def predict_station(self, station_id: str, mode: str = "balanced") -> Dict:
        """Generate prediction for a specific station"""
        
        # Generate features for station
        features = self.feature_generator.generate_features_for_station(station_id)
        
        if features is None:
            logger.error(f"Could not generate features for station {station_id}")
            return {}
        
        # Prepare features for prediction
        feature_cols = self.thresholds['model_info']['features']
        X = features[feature_cols].values.reshape(1, -1)
        
        # Generate prediction
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X)[0, 1]
        else:
            # Native LightGBM Booster
            probability = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        threshold = self.get_threshold(mode)
        prediction = int(probability >= threshold)
        
        # Create detailed result
        result = {
            'station_id': station_id,
            'stockout_probability': float(probability),
            'is_stockout_predicted': prediction,
            'risk_level': self.calculate_risk_level(probability),
            'prediction_mode': mode,
            'threshold': threshold,
            'confidence': self.calculate_confidence(probability, threshold),
            'timestamp': datetime.now().isoformat(),
            'current_status': {
                'available_bikes': int(features.get('available_bikes', 0)),
                'station_capacity': int(features.get('station_capacity', 0)),
                'utilization_rate': float(features.get('utilization_rate', 0)),
                'is_currently_stockout': bool(features.get('is_stockout', 0))
            },
            'feature_importance': self.get_top_features(features)
        }
        
        return result
    
    def get_high_risk_stations(self, threshold: float = 0.7) -> pd.DataFrame:
        """Get stations with high stockout risk"""
        
        predictions = self.predict_all_stations()
        
        if predictions.empty:
            return pd.DataFrame()
        
        # Filter high risk stations
        high_risk = predictions[predictions['stockout_probability'] >= threshold].copy()
        high_risk = high_risk.sort_values('stockout_probability', ascending=False)
        
        return high_risk
    
    def get_threshold(self, mode: str) -> float:
        """Get prediction threshold for given mode"""
        if mode == "alert":
            return self.thresholds['alert_mode']['threshold']
        elif mode == "balanced":
            return self.thresholds['balanced_mode']['threshold']
        else:
            return Config.PREDICTION_MODES.get(mode, 0.65)
    
    def calculate_risk_levels(self, probabilities: np.ndarray) -> List[str]:
        """Calculate risk levels from probabilities"""
        risk_levels = []
        for prob in probabilities:
            risk_levels.append(self.calculate_risk_level(prob))
        return risk_levels
    
    def calculate_risk_level(self, probability: float) -> str:
        """Calculate risk level from probability"""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def calculate_confidence(self, probability: float, threshold: float) -> float:
        """Calculate prediction confidence"""
        # Distance from threshold indicates confidence
        distance = abs(probability - threshold)
        # Normalize to 0-1 scale
        confidence = min(distance * 2, 1.0)
        return round(confidence, 3)
    
    def get_top_features(self, features: pd.Series, top_n: int = 10) -> Dict:
        """Get top contributing features for a prediction"""
        
        # Get feature importance from model
        if hasattr(self.model, 'feature_importance_'):
            importance = self.model.feature_importance_
            feature_names = self.thresholds['model_info']['features']
            
            # Create importance dict
            importance_dict = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Get top features with their values
            top_features = {}
            for feat_name, feat_importance in sorted_features[:top_n]:
                if feat_name in features.index:
                    top_features[feat_name] = {
                        'value': float(features[feat_name]),
                        'importance': int(feat_importance)
                    }
            
            return top_features
        
        return {}
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': self.thresholds['model_info']['model_type'],
            'target': self.thresholds['model_info']['target'],
            'num_features': self.thresholds['model_info']['num_features'],
            'training_date': self.thresholds['model_info']['training_date'],
            'best_iteration': self.thresholds['model_info']['best_iteration'],
            'modes': {
                'alert': {
                    'threshold': self.thresholds['alert_mode']['threshold'],
                    'recall': self.thresholds['alert_mode']['recall'],
                    'precision': self.thresholds['alert_mode']['precision'],
                    'f1_score': self.thresholds['alert_mode']['f1_score']
                },
                'balanced': {
                    'threshold': self.thresholds['balanced_mode']['threshold'],
                    'recall': self.thresholds['balanced_mode']['recall'],
                    'precision': self.thresholds['balanced_mode']['precision'],
                    'f1_score': self.thresholds['balanced_mode']['f1_score']
                }
            },
            'high_risk_stations': self.thresholds['operational_patterns']['high_risk_stations'],
            'peak_hours': self.thresholds['operational_patterns']['peak_hours']
        }
    
    def batch_predict(self, station_ids: List[str], mode: str = "balanced") -> List[Dict]:
        """Generate predictions for multiple specific stations"""
        results = []
        
        for station_id in station_ids:
            result = self.predict_station(station_id, mode)
            if result:
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Test prediction service
    logging.basicConfig(level=logging.INFO)
    
    service = PredictionService()
    
    # Test model info
    print("Model Information:")
    info = service.get_model_info()
    print(f"Model type: {info['model_type']}")
    print(f"Features: {info['num_features']}")
    print(f"Training date: {info['training_date']}")
    
    # Test predictions
    print("\nGenerating predictions for all stations...")
    predictions = service.predict_all_stations(mode="balanced")
    
    if not predictions.empty:
        print(f"\nPredictions generated for {len(predictions)} stations")
        print(f"Stations at risk: {predictions['is_stockout_predicted'].sum()}")
        print(f"Average risk probability: {predictions['stockout_probability'].mean():.3f}")
        
        # Show high risk stations
        high_risk = service.get_high_risk_stations(threshold=0.6)
        if not high_risk.empty:
            print(f"\nHigh risk stations (>60% probability):")
            print(high_risk[['station_id', 'stockout_probability', 'current_available_bikes']].head())
    else:
        print("Failed to generate predictions")