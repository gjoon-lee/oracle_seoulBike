"""
LightGBM Classifier for Seoul Bike Stockout Prediction
Trains model to predict stockout events 2 hours ahead
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class StockoutPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True  # Handle class imbalance
        }
    
    def load_data(self, train_file='lightgbm_train_2024.parquet', test_file='lightgbm_test_2024.parquet'):
        """Load prepared training and test data"""
        print("📊 Loading training data...")
        train_df = pd.read_parquet(train_file)
        test_df = pd.read_parquet(test_file)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Define feature columns (exclude identifiers and targets)
        exclude_cols = [
            'station_id', 'date', 'hour', 'day_of_week', 'season',
            'target_stockout_2h', 'target_nearly_empty_2h', 
            'target_net_flow_2h', 'target_available_bikes_2h'
        ]
        
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        print(f"Using {len(self.feature_cols)} features")
        
        # Prepare datasets
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_stockout_2h'].astype(int)
        X_test = test_df[self.feature_cols]
        y_test = test_df['target_stockout_2h'].astype(int)
        
        # Store metadata for analysis
        self.test_metadata = test_df[['station_id', 'date', 'hour']].copy()
        
        print(f"\nTarget distribution:")
        print(f"Train: {y_train.mean():.3f} stockout rate ({y_train.sum():,} / {len(y_train):,})")
        print(f"Test: {y_test.mean():.3f} stockout rate ({y_test.sum():,} / {len(y_test):,})")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train LightGBM classifier with early stopping"""
        print("\n🏋️ Training LightGBM model...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best score: {self.model.best_score['valid']['binary_logloss']:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📈 Evaluating model performance...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n🎯 Classification Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"True Neg: {cm[0,0]:,} | False Pos: {cm[0,1]:,}")
        print(f"False Neg: {cm[1,0]:,} | True Pos: {cm[1,1]:,}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        })
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance (LightGBM)')
        plt.xlabel('Importance (Gain)')
        plt.tight_layout()
        plt.savefig('lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def analyze_predictions(self, X_test, y_test):
        """Analyze predictions by time and station"""
        print("\n🔍 Analyzing predictions...")
        
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Create analysis dataframe
        analysis_df = self.test_metadata.copy()
        analysis_df['actual'] = y_test.values
        analysis_df['predicted_proba'] = y_pred_proba
        analysis_df['predicted'] = (y_pred_proba > 0.5).astype(int)
        analysis_df['correct'] = (analysis_df['actual'] == analysis_df['predicted']).astype(int)
        
        # Analysis by hour
        hourly_stats = analysis_df.groupby('hour').agg({
            'actual': 'mean',
            'predicted_proba': 'mean',
            'correct': 'mean'
        }).round(3)
        
        print("\n⏰ Performance by Hour:")
        print(hourly_stats.head(10))
        
        # Analysis by station (top 10 problematic stations)
        station_stats = analysis_df.groupby('station_id').agg({
            'actual': ['count', 'mean'],
            'correct': 'mean'
        })
        station_stats.columns = ['total_predictions', 'stockout_rate', 'accuracy']
        problem_stations = station_stats[station_stats['total_predictions'] >= 100].sort_values('accuracy').head(10)
        
        print("\n🚲 Top 10 Most Challenging Stations:")
        print(problem_stations)
        
        return analysis_df
    
    def save_model(self):
        """Save trained model and metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'models/lightgbm_stockout_model_{timestamp}.pkl'
        joblib.dump(self.model, model_path)
        
        # Save feature columns and metadata
        metadata = {
            'model_type': 'LightGBM Binary Classifier',
            'target': 'stockout_2h_ahead',
            'features': self.feature_cols,
            'num_features': len(self.feature_cols),
            'model_params': self.model_params,
            'best_iteration': self.model.best_iteration,
            'training_date': timestamp
        }
        
        metadata_path = f'models/lightgbm_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Model saved:")
        print(f"Model: {model_path}")
        print(f"Metadata: {metadata_path}")
        
        return model_path, metadata_path
    
    def run_full_training(self):
        """Execute complete training pipeline"""
        print("🚀 Starting LightGBM stockout prediction training")
        print("="*60)
        
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()
        
        # Train model
        self.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Feature importance
        importance_df = self.plot_feature_importance()
        
        # Prediction analysis
        analysis_df = self.analyze_predictions(X_test, y_test)
        
        # Save model
        model_path, metadata_path = self.save_model()
        
        print("\n✅ Training complete!")
        print(f"🎯 Final F1-Score: {metrics['f1_score']:.4f}")
        print(f"🎯 Final ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return self.model, metrics, importance_df

def main():
    """Main training function"""
    predictor = StockoutPredictor()
    model, metrics, importance = predictor.run_full_training()
    return predictor

if __name__ == "__main__":
    predictor = main()