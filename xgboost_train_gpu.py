"""
XGBoost training script optimized for Google Colab GPU
Trains model to predict net flow 2 hours ahead
Compatible with existing API infrastructure
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class XGBoostTrainer:
    def __init__(self, use_gpu=True, base_dir='.'):
        self.use_gpu = use_gpu
        self.base_dir = base_dir
        self.models_dir = f'{base_dir}/models'
        self.model = None
        self.feature_cols = None
        self.best_iteration = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_data(self):
        """Load prepared data from parquet files"""
        
        print("="*50)
        print("Loading XGBoost training data...")
        print("="*50)
        
        train_df = pd.read_parquet(f'{self.base_dir}/xgboost_train_2024.parquet')
        val_df = pd.read_parquet(f'{self.base_dir}/xgboost_val_2024.parquet')
        test_df = pd.read_parquet(f'{self.base_dir}/xgboost_test_2024.parquet')
        
        # Load feature config
        with open(f'{self.base_dir}/xgboost_features_config.json', 'r') as f:
            config = json.load(f)
        
        self.feature_cols = config['features']
        
        print(f"✅ Features: {len(self.feature_cols)}")
        print(f"✅ Train: {len(train_df):,} samples")
        print(f"✅ Val: {len(val_df):,} samples")
        print(f"✅ Test: {len(test_df):,} samples")
        
        # Print target statistics
        print("\nTarget Statistics (net_flow_2h):")
        print(f"  Train - Mean: {train_df['target_net_flow_2h'].mean():.2f}, Std: {train_df['target_net_flow_2h'].std():.2f}")
        print(f"  Val   - Mean: {val_df['target_net_flow_2h'].mean():.2f}, Std: {val_df['target_net_flow_2h'].std():.2f}")
        print(f"  Test  - Mean: {test_df['target_net_flow_2h'].mean():.2f}, Std: {test_df['target_net_flow_2h'].std():.2f}")
        
        return train_df, val_df, test_df
    
    def prepare_datasets(self, train_df, val_df, test_df):
        """Prepare XGBoost DMatrix objects"""
        
        print("\nPreparing datasets...")
        
        # Extract features and target
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_net_flow_2h']
        w_train = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None
        
        X_val = val_df[self.feature_cols]
        y_val = val_df['target_net_flow_2h']
        
        X_test = test_df[self.feature_cols]
        y_test = test_df['target_net_flow_2h']
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        print(f"✅ DMatrix objects created")
        
        return dtrain, dval, dtest, X_test, y_test, test_df
    
    def train_model(self, dtrain, dval):
        """Train XGBoost with GPU support"""
        
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Model parameters - tuned for better performance
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,  # Learning rate
            'max_depth': 8,  # Deeper trees for complex patterns
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,  # Minimum loss reduction
            'lambda': 1,  # L2 regularization
            'alpha': 0.1,  # L1 regularization
            'seed': 42
        }
        
        # GPU settings
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
            print("🚀 Using GPU acceleration")
        else:
            params['tree_method'] = 'hist'
            print("💻 Using CPU (consider using GPU for faster training)")
        
        # Training
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        print("\nStarting training...")
        print("-" * 30)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        self.best_iteration = self.model.best_iteration
        print(f"\n✅ Training complete!")
        print(f"📊 Best iteration: {self.best_iteration}")
        
        return self.model
    
    def evaluate_model(self, dtest, X_test, y_test, test_df):
        """Evaluate model performance"""
        
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(dtest)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (avoiding division by zero)
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'mean_error': float(np.mean(y_pred - y_test)),
            'std_error': float(np.std(y_pred - y_test))
        }
        
        print("📈 Test Set Performance:")
        print(f"  RMSE: {rmse:.3f} bikes")
        print(f"  MAE:  {mae:.3f} bikes")
        print(f"  R²:   {r2:.3f}")
        print(f"  MAPE: {mape:.1f}%")
        
        # Analyze predictions by magnitude
        print("\n📊 Prediction Analysis:")
        
        # Check if model is making varied predictions
        pred_std = np.std(y_pred)
        actual_std = np.std(y_test)
        print(f"  Prediction std: {pred_std:.2f}")
        print(f"  Actual std:     {actual_std:.2f}")
        print(f"  Ratio:          {pred_std/actual_std:.2f}")
        
        # Distribution of predictions
        print("\n  Prediction distribution:")
        ranges = [(-np.inf, -10), (-10, -5), (-5, -2), (-2, 2), (2, 5), (5, 10), (10, np.inf)]
        labels = ['< -10', '-10 to -5', '-5 to -2', '-2 to 2', '2 to 5', '5 to 10', '> 10']
        
        for (low, high), label in zip(ranges, labels):
            mask = (y_pred >= low) & (y_pred < high)
            count = mask.sum()
            pct = count / len(y_pred) * 100
            print(f"    {label:12s}: {count:6,} ({pct:5.1f}%)")
        
        # Plot predictions vs actual
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(y_test, y_pred, alpha=0.3, s=1, c=np.abs(y_test), cmap='viridis')
        ax.plot([-50, 50], [-50, 50], 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('Actual Net Flow (bikes)')
        ax.set_ylabel('Predicted Net Flow (bikes)')
        ax.set_title(f'Predictions vs Actual (R² = {r2:.3f})')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='|Actual|')
        
        # Error distribution
        ax = axes[0, 1]
        errors = y_pred - y_test
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(0, color='red', linestyle='--', label='Zero error')
        ax.set_xlabel('Prediction Error (bikes)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution (MAE = {mae:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actual vs Predicted distributions
        ax = axes[1, 0]
        ax.hist(y_test, bins=50, alpha=0.5, label='Actual', color='blue', edgecolor='black')
        ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
        ax.set_xlabel('Net Flow (bikes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error by hour of day
        ax = axes[1, 1]
        if 'hour' in test_df.columns:
            hourly_errors = test_df.groupby('hour').apply(
                lambda x: np.abs(y_pred[x.index] - y_test[x.index]).mean()
            )
            ax.plot(hourly_errors.index, hourly_errors.values, marker='o', linewidth=2)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Mean Absolute Error (bikes)')
            ax.set_title('Error by Hour of Day')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Hour data not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/xgboost_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return metrics
    
    def analyze_feature_importance(self):
        """Analyze and plot feature importance"""
        
        print("\n" + "="*50)
        print("Feature Importance Analysis")
        print("="*50)
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        # Normalize importance
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum()
        )
        
        print(f"\n📊 Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance_normalized']*100:5.2f}%")
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance_normalized'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Normalized Importance')
        plt.title('Top 20 Feature Importance (by Gain)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar, value in zip(bars, top_features['importance_normalized']):
            plt.text(value, bar.get_y() + bar.get_height()/2, 
                    f'{value*100:.1f}%', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def save_model(self, metrics, importance_df):
        """Save model and configuration for API compatibility"""
        
        print("\n" + "="*50)
        print("Saving Model")
        print("="*50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for API compatibility (API uses joblib.load)
        pickle_path = f'{self.models_dir}/xgb_target_net_flow_2h_{timestamp}.pkl'
        joblib.dump(self.model, pickle_path)
        print(f"✅ Model saved: {pickle_path}")
        
        # Also save in native XGBoost format
        xgb_path = f'{self.models_dir}/xgb_target_net_flow_2h_{timestamp}.xgb'
        self.model.save_model(xgb_path)
        print(f"✅ XGBoost format: {xgb_path}")
        
        # Save configuration (MUST match API expected format exactly)
        config = {
            "model_info": {
                "model_type": "XGBoost regression",
                "target": "target_net_flow_2h",
                "features": self.feature_cols,
                "num_features": len(self.feature_cols),
                "model_params": {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "eta": 0.05,
                    "max_depth": 8,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "seed": 42,
                    "tree_method": "gpu_hist" if self.use_gpu else "hist"
                },
                "best_iteration": int(self.best_iteration),
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "metrics": metrics
        }
        
        config_path = f'{self.models_dir}/xgb_target_net_flow_2h_config_{timestamp}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved: {config_path}")
        
        # Save feature importance
        importance_path = f'{self.models_dir}/xgb_importance_{timestamp}.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"✅ Feature importance: {importance_path}")
        
        print("\n" + "="*50)
        print("📝 API Integration Instructions:")
        print("="*50)
        print("Update realtime_prediction/config/config.py with:")
        print(f"  XGB_MODEL_PATH = 'models/xgb_target_net_flow_2h_{timestamp}.pkl'")
        print(f"  XGB_CONFIG_PATH = 'models/xgb_target_net_flow_2h_config_{timestamp}.json'")
        print("\nThen restart the API to load the new model.")
        
        return timestamp
    
    def run_training_pipeline(self):
        """Complete training pipeline"""
        
        print("\n" + "🚀 "*20)
        print("XGBOOST TRAINING PIPELINE")
        print("🚀 "*20)
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Prepare datasets
        dtrain, dval, dtest, X_test, y_test, test_df_full = self.prepare_datasets(
            train_df, val_df, test_df
        )
        
        # Train model
        self.train_model(dtrain, dval)
        
        # Evaluate
        metrics = self.evaluate_model(dtest, X_test, y_test, test_df_full)
        
        # Feature importance
        importance_df = self.analyze_feature_importance()
        
        # Save everything
        timestamp = self.save_model(metrics, importance_df)
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETE!")
        print("="*50)
        print(f"📊 Final Performance:")
        print(f"   R² Score: {metrics['r2']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f} bikes")
        print(f"   RMSE: {metrics['rmse']:.3f} bikes")
        
        if metrics['r2'] < 0.5:
            print("\n⚠️  Warning: R² is below 0.5. Consider:")
            print("   - Adding more training data")
            print("   - Feature engineering improvements")
            print("   - Hyperparameter tuning")
        elif metrics['r2'] > 0.7:
            print("\n🎉 Excellent model performance!")
        
        return self.model, metrics, timestamp

def check_gpu():
    """Check if GPU is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected")
            return True
    except:
        pass
    
    print("⚠️  No GPU detected - training will use CPU (slower)")
    return False

if __name__ == "__main__":
    # For Google Colab, uncomment these lines:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # BASE_DIR = '/content/drive/MyDrive/seoul_bikes'
    
    # For local training:
    BASE_DIR = '.'
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    # Create trainer
    trainer = XGBoostTrainer(use_gpu=use_gpu, base_dir=BASE_DIR)
    
    # Run training
    try:
        model, metrics, timestamp = trainer.run_training_pipeline()
        
        # Save a summary file
        summary = {
            'timestamp': timestamp,
            'performance': metrics,
            'gpu_used': use_gpu,
            'best_iteration': trainer.best_iteration
        }
        
        with open(f'{BASE_DIR}/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise