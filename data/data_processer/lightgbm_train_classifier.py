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
from datetime import datetime, timedelta
import joblib
import json
import warnings
import sys
import io

# Set output encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

class StockoutPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.model_params = {
            'objective': 'binary',
            'metric': 'None',  # Will use custom recall metric
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'scale_pos_weight': 3  # Penalize false negatives more heavily
        }
        self.optimal_thresholds = None
        self.operational_metrics = None
    
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
    
    def recall_eval(self, y_pred, y_true):
        """Custom recall metric for LightGBM early stopping"""
        y_true_labels = y_true.get_label()
        y_pred_binary = (y_pred > 0.5).astype(int)
        recall = recall_score(y_true_labels, y_pred_binary)
        return 'recall', recall, True  # Higher is better
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train LightGBM classifier with early stopping"""
        print("\n🏋️ Training LightGBM model...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model with recall optimization
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            feval=self.recall_eval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best recall: {self.model.best_score['valid']['recall']:.4f}")
        
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
    
    def find_optimal_thresholds(self, X_test, y_test):
        """Find optimal thresholds for alert mode (85% recall) and balanced mode (max F1)"""
        print("\n🎯 Finding optimal thresholds...")
        
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        thresholds = np.arange(0.1, 0.91, 0.05)
        metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Count predictions
            num_alerts = y_pred.sum()
            
            metrics.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'num_alerts': num_alerts
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Find alert threshold (closest to 85% recall)
        target_recall = 0.85
        metrics_df['recall_diff'] = abs(metrics_df['recall'] - target_recall)
        alert_idx = metrics_df['recall_diff'].idxmin()
        alert_threshold = metrics_df.loc[alert_idx, 'threshold']
        
        # Find balanced threshold (maximum F1)
        balanced_idx = metrics_df['f1'].idxmax()
        balanced_threshold = metrics_df.loc[balanced_idx, 'threshold']
        
        self.optimal_thresholds = {
            'alert_threshold': alert_threshold,
            'alert_recall': metrics_df.loc[alert_idx, 'recall'],
            'alert_precision': metrics_df.loc[alert_idx, 'precision'],
            'alert_f1': metrics_df.loc[alert_idx, 'f1'],
            'alert_num_predictions': metrics_df.loc[alert_idx, 'num_alerts'],
            'balanced_threshold': balanced_threshold,
            'balanced_recall': metrics_df.loc[balanced_idx, 'recall'],
            'balanced_precision': metrics_df.loc[balanced_idx, 'precision'],
            'balanced_f1': metrics_df.loc[balanced_idx, 'f1'],
            'balanced_num_predictions': metrics_df.loc[balanced_idx, 'num_alerts']
        }
        
        print(f"✅ Alert threshold: {alert_threshold:.2f} (Recall: {self.optimal_thresholds['alert_recall']:.3f})")
        print(f"✅ Balanced threshold: {balanced_threshold:.2f} (F1: {self.optimal_thresholds['balanced_f1']:.3f})")
        
        return self.optimal_thresholds
    
    def evaluate_dual_mode(self, X_test, y_test):
        """Evaluate model performance in both operating modes"""
        print("\n" + "="*60)
        print("📊 DUAL-MODE EVALUATION")
        print("="*60)
        
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Alert Mode Evaluation
        print("\n🚨 ALERT MODE (85% Recall Target)")
        print("-"*40)
        alert_threshold = self.optimal_thresholds['alert_threshold']
        y_pred_alert = (y_pred_proba > alert_threshold).astype(int)
        
        cm_alert = confusion_matrix(y_test, y_pred_alert)
        recall_alert = recall_score(y_test, y_pred_alert)
        precision_alert = precision_score(y_test, y_pred_alert, zero_division=0)
        f1_alert = f1_score(y_test, y_pred_alert, zero_division=0)
        
        print(f"Threshold: {alert_threshold:.2f}")
        print(f"Recall: {recall_alert:.1%} (catching {cm_alert[1,1]:,} of {cm_alert[1,0]+cm_alert[1,1]:,} stockouts)")
        print(f"Precision: {precision_alert:.1%}")
        print(f"F1-Score: {f1_alert:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"True Neg: {cm_alert[0,0]:,} | False Pos: {cm_alert[0,1]:,}")
        print(f"False Neg: {cm_alert[1,0]:,} | True Pos: {cm_alert[1,1]:,}")
        
        # Balanced Mode Evaluation
        print("\n⚖️ BALANCED MODE (Maximum F1)")
        print("-"*40)
        balanced_threshold = self.optimal_thresholds['balanced_threshold']
        y_pred_balanced = (y_pred_proba > balanced_threshold).astype(int)
        
        cm_balanced = confusion_matrix(y_test, y_pred_balanced)
        recall_balanced = recall_score(y_test, y_pred_balanced)
        precision_balanced = precision_score(y_test, y_pred_balanced, zero_division=0)
        f1_balanced = f1_score(y_test, y_pred_balanced, zero_division=0)
        
        print(f"Threshold: {balanced_threshold:.2f}")
        print(f"Recall: {recall_balanced:.1%}")
        print(f"Precision: {precision_balanced:.1%}")
        print(f"F1-Score: {f1_balanced:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"True Neg: {cm_balanced[0,0]:,} | False Pos: {cm_balanced[0,1]:,}")
        print(f"False Neg: {cm_balanced[1,0]:,} | True Pos: {cm_balanced[1,1]:,}")
        
        return {
            'alert_mode': {
                'threshold': alert_threshold,
                'recall': recall_alert,
                'precision': precision_alert,
                'f1': f1_alert,
                'confusion_matrix': cm_alert.tolist()
            },
            'balanced_mode': {
                'threshold': balanced_threshold,
                'recall': recall_balanced,
                'precision': precision_balanced,
                'f1': f1_balanced,
                'confusion_matrix': cm_balanced.tolist()
            }
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
    
    def analyze_operational_patterns(self, X_test, y_test):
        """Analyze operational patterns for resource planning"""
        print("\n📈 OPERATIONAL PATTERNS ANALYSIS")
        print("-"*40)
        
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Create analysis dataframe with metadata
        analysis_df = self.test_metadata.copy()
        analysis_df['actual'] = y_test.values
        analysis_df['predicted_proba'] = y_pred_proba
        
        # Add predictions for both thresholds
        analysis_df['alert_mode'] = (y_pred_proba > self.optimal_thresholds['alert_threshold']).astype(int)
        analysis_df['balanced_mode'] = (y_pred_proba > self.optimal_thresholds['balanced_threshold']).astype(int)
        
        # Add day of week (assuming date column exists)
        analysis_df['day_of_week'] = pd.to_datetime(analysis_df['date']).dt.dayofweek
        analysis_df['is_weekend'] = analysis_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Hourly patterns
        hourly_patterns = analysis_df.groupby('hour').agg({
            'actual': 'mean',
            'alert_mode': 'sum',
            'balanced_mode': 'sum'
        }).round(3)
        
        # Peak vs off-peak
        peak_hours = [7, 8, 9, 17, 18, 19]
        analysis_df['is_peak'] = analysis_df['hour'].isin(peak_hours).astype(int)
        
        peak_stats = analysis_df.groupby('is_peak').agg({
            'actual': 'mean',
            'alert_mode': 'mean',
            'balanced_mode': 'mean'
        })
        
        print("\n⏰ Hourly Alert Distribution:")
        print(f"Peak hours (7-9am, 5-7pm): {hourly_patterns.loc[peak_hours, 'alert_mode'].mean():.0f} alerts/hour")
        print(f"Off-peak hours: {hourly_patterns.loc[~hourly_patterns.index.isin(peak_hours), 'alert_mode'].mean():.0f} alerts/hour")
        
        # Weekend vs weekday
        weekend_stats = analysis_df.groupby('is_weekend').agg({
            'actual': ['count', 'mean'],
            'alert_mode': 'sum',
            'balanced_mode': 'sum'
        })
        
        weekday_alerts = weekend_stats.loc[0, ('alert_mode', 'sum')] / analysis_df[analysis_df['is_weekend']==0]['date'].nunique()
        weekend_alerts = weekend_stats.loc[1, ('alert_mode', 'sum')] / analysis_df[analysis_df['is_weekend']==1]['date'].nunique()
        
        print(f"\n📅 Weekend vs Weekday Patterns:")
        print(f"Weekday average: {weekday_alerts:.0f} alerts/day")
        print(f"Weekend average: {weekend_alerts:.0f} alerts/day")
        print(f"Weekend/Weekday ratio: {weekend_alerts/weekday_alerts:.2f}")
        
        # Station-level analysis
        station_stats = analysis_df.groupby('station_id').agg({
            'actual': ['count', 'mean'],
            'alert_mode': 'sum'
        })
        station_stats.columns = ['total_records', 'stockout_rate', 'total_alerts']
        problem_stations = station_stats[station_stats['total_records'] >= 100].sort_values('stockout_rate', ascending=False).head(10)
        
        print(f"\n🚲 Top 10 High-Risk Stations:")
        for idx, (station_id, row) in enumerate(problem_stations.iterrows(), 1):
            print(f"{idx}. {station_id}: {row['stockout_rate']:.1%} stockout rate, {row['total_alerts']:.0f} alerts")
        
        self.operational_metrics = {
            'peak_hour_alerts': hourly_patterns.loc[peak_hours, 'alert_mode'].mean(),
            'off_peak_alerts': hourly_patterns.loc[~hourly_patterns.index.isin(peak_hours), 'alert_mode'].mean(),
            'weekday_daily_alerts': weekday_alerts,
            'weekend_daily_alerts': weekend_alerts,
            'weekend_weekday_ratio': weekend_alerts/weekday_alerts,
            'high_risk_stations': problem_stations.index.tolist()[:5],
            'peak_hours': peak_hours
        }
        
        return analysis_df
    
    def calculate_business_metrics(self, X_test, y_test):
        """Calculate business impact metrics"""
        print("\n💼 BUSINESS IMPACT SUMMARY")
        print("-"*40)
        
        # Calculate total test period in days
        test_days = self.test_metadata['date'].nunique()
        test_hours = len(X_test) / self.test_metadata['station_id'].nunique()
        
        # Alert mode metrics
        alert_predictions = self.optimal_thresholds['alert_num_predictions']
        alert_daily = alert_predictions / test_days
        alert_recall = self.optimal_thresholds['alert_recall']
        
        # Balanced mode metrics
        balanced_predictions = self.optimal_thresholds['balanced_num_predictions']
        balanced_daily = balanced_predictions / test_days
        balanced_precision = self.optimal_thresholds['balanced_precision']
        
        print(f"\n🚨 Alert Mode: Would generate {alert_daily:.0f} alerts per day")
        print(f"   - Stockout catch rate: {alert_recall:.1%}")
        print(f"   - False positive rate: {(1-self.optimal_thresholds['alert_precision']):.1%}")
        
        print(f"\n⚖️ Balanced Mode: Would generate {balanced_daily:.0f} alerts per day")
        print(f"   - Precision: {balanced_precision:.1%}")
        print(f"   - Stockout catch rate: {self.optimal_thresholds['balanced_recall']:.1%}")
        
        return {
            'alert_mode_daily': alert_daily,
            'balanced_mode_daily': balanced_daily,
            'test_days': test_days
        }
    
    def simulate_production_performance(self, X_test, y_test, lag_days=7):
        """Simulate production performance with lagged features"""
        print("\n🏭 PRODUCTION SIMULATION (1-week lag)")
        print("-"*40)
        
        # Identify lag feature columns
        lag_cols = [col for col in self.feature_cols if 'lag_' in col or 'roll_' in col]
        
        # Create degraded feature set by shifting lag features
        X_test_production = X_test.copy()
        
        # Simulate missing recent data by setting recent lag features to NaN
        # Then forward fill with older values
        for col in lag_cols:
            if 'lag_' in col:
                # Extract lag hours from column name
                lag_hours = int(col.split('lag_')[1].split('h')[0])
                if lag_hours <= lag_days * 24:
                    # Simulate this data being unavailable
                    X_test_production[col] = X_test_production[col].shift(lag_days * 24)
                    X_test_production[col].fillna(X_test_production[col].mean(), inplace=True)
        
        # Make predictions with degraded features
        y_pred_proba_perfect = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred_proba_production = self.model.predict(X_test_production, num_iteration=self.model.best_iteration)
        
        # Evaluate at alert threshold
        threshold = self.optimal_thresholds['alert_threshold']
        y_pred_perfect = (y_pred_proba_perfect > threshold).astype(int)
        y_pred_production = (y_pred_proba_production > threshold).astype(int)
        
        # Calculate metrics
        recall_perfect = recall_score(y_test, y_pred_perfect)
        recall_production = recall_score(y_test, y_pred_production)
        f1_perfect = f1_score(y_test, y_pred_perfect)
        f1_production = f1_score(y_test, y_pred_production)
        
        performance_drop = (recall_perfect - recall_production) / recall_perfect * 100
        
        print(f"\nPerfect Features Performance:")
        print(f"  - Recall: {recall_perfect:.1%}")
        print(f"  - F1: {f1_perfect:.3f}")
        
        print(f"\nProduction ({lag_days}-day lag) Performance:")
        print(f"  - Recall: {recall_production:.1%} ({recall_production-recall_perfect:+.1%})")
        print(f"  - F1: {f1_production:.3f} ({f1_production-f1_perfect:+.3f})")
        print(f"  - Performance Drop: {performance_drop:.1f}%")
        
        return {
            'perfect_recall': recall_perfect,
            'production_recall': recall_production,
            'perfect_f1': f1_perfect,
            'production_f1': f1_production,
            'performance_drop': performance_drop
        }
    
    def save_model_and_config(self, production_metrics=None):
        """Save trained model with comprehensive configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'models/lightgbm_stockout_model_{timestamp}.pkl'
        joblib.dump(self.model, model_path)
        
        # Prepare comprehensive configuration
        config = {
            'model_info': {
                'model_type': 'LightGBM Binary Classifier (Dual-Mode)',
                'target': 'stockout_2h_ahead',
                'features': self.feature_cols,
                'num_features': len(self.feature_cols),
                'model_params': self.model_params,
                'best_iteration': self.model.best_iteration,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'alert_mode': {
                'threshold': float(self.optimal_thresholds['alert_threshold']),
                'recall': float(self.optimal_thresholds['alert_recall']),
                'precision': float(self.optimal_thresholds['alert_precision']),
                'f1_score': float(self.optimal_thresholds['alert_f1']),
                'alerts_per_day': float(self.optimal_thresholds['alert_num_predictions'] / self.test_metadata['date'].nunique())
            },
            'balanced_mode': {
                'threshold': float(self.optimal_thresholds['balanced_threshold']),
                'recall': float(self.optimal_thresholds['balanced_recall']),
                'precision': float(self.optimal_thresholds['balanced_precision']),
                'f1_score': float(self.optimal_thresholds['balanced_f1']),
                'alerts_per_day': float(self.optimal_thresholds['balanced_num_predictions'] / self.test_metadata['date'].nunique())
            }
        }
        
        # Add operational metrics if available
        if self.operational_metrics:
            config['operational_patterns'] = {
                'peak_hour_alerts': float(self.operational_metrics['peak_hour_alerts']),
                'off_peak_alerts': float(self.operational_metrics['off_peak_alerts']),
                'weekday_daily_alerts': float(self.operational_metrics['weekday_daily_alerts']),
                'weekend_daily_alerts': float(self.operational_metrics['weekend_daily_alerts']),
                'weekend_weekday_ratio': float(self.operational_metrics['weekend_weekday_ratio']),
                'high_risk_stations': self.operational_metrics['high_risk_stations'],
                'peak_hours': self.operational_metrics['peak_hours']
            }
        
        # Add production simulation metrics if available
        if production_metrics:
            config['production_simulation'] = {
                'perfect_recall': float(production_metrics['perfect_recall']),
                'production_recall': float(production_metrics['production_recall']),
                'perfect_f1': float(production_metrics['perfect_f1']),
                'production_f1': float(production_metrics['production_f1']),
                'performance_drop': float(production_metrics['performance_drop'])
            }
        
        # Save threshold configuration
        threshold_path = f'models/model_thresholds_{timestamp}.json'
        with open(threshold_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also save a simple threshold file for easy loading
        simple_config = {
            'alert_threshold': float(self.optimal_thresholds['alert_threshold']),
            'balanced_threshold': float(self.optimal_thresholds['balanced_threshold']),
            'alert_recall': float(self.optimal_thresholds['alert_recall']),
            'alert_precision': float(self.optimal_thresholds['alert_precision']),
            'balanced_f1': float(self.optimal_thresholds['balanced_f1']),
            'training_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        simple_path = f'models/thresholds_{timestamp}.json'
        with open(simple_path, 'w') as f:
            json.dump(simple_config, f, indent=2)
        
        print(f"\n💾 Model and configuration saved:")
        print(f"  - Model: {model_path}")
        print(f"  - Full config: {threshold_path}")
        print(f"  - Simple thresholds: {simple_path}")
        
        return model_path, threshold_path
    
    def run_full_training(self):
        """Execute complete training pipeline with dual-mode evaluation"""
        print("="*60)
        print("🚀 LIGHTGBM DUAL-MODE STOCKOUT PREDICTION SYSTEM")
        print("="*60)
        
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()
        
        # Train model with recall optimization
        self.train_model(X_train, y_train, X_test, y_test)
        
        # Find optimal thresholds for both modes
        self.find_optimal_thresholds(X_test, y_test)
        
        # Dual-mode evaluation
        dual_metrics = self.evaluate_dual_mode(X_test, y_test)
        
        # Operational pattern analysis
        analysis_df = self.analyze_operational_patterns(X_test, y_test)
        
        # Business impact metrics
        business_metrics = self.calculate_business_metrics(X_test, y_test)
        
        # Production simulation
        production_metrics = self.simulate_production_performance(X_test, y_test)
        
        # Feature importance
        importance_df = self.plot_feature_importance()
        
        # Save model with comprehensive configuration
        model_path, config_path = self.save_model_and_config(production_metrics)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE - DUAL-MODE SYSTEM READY")
        print("="*60)
        print(f"\n🚨 Alert Mode: {self.optimal_thresholds['alert_recall']:.1%} recall @ threshold {self.optimal_thresholds['alert_threshold']:.2f}")
        print(f"⚖️ Balanced Mode: {self.optimal_thresholds['balanced_f1']:.3f} F1 @ threshold {self.optimal_thresholds['balanced_threshold']:.2f}")
        print(f"\n📊 Expected daily alerts: {business_metrics['alert_mode_daily']:.0f} (alert) / {business_metrics['balanced_mode_daily']:.0f} (balanced)")
        print(f"🏭 Production degradation: {production_metrics['performance_drop']:.1f}% with 1-week lag")
        
        return self.model, dual_metrics, importance_df

def main():
    """Main training function"""
    predictor = StockoutPredictor()
    model, metrics, importance = predictor.run_full_training()
    return predictor

if __name__ == "__main__":
    predictor = main()