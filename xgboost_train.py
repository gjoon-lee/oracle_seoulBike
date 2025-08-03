"""
Working NetFlow Model Training for Seoul Bike System
Fixed version with proper XGBoost parameters
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

def prepare_netflow_data():
    """Prepare data with NetFlow targets"""
    print("📊 Loading and preparing data...")
    
    # Load data
    train_df = pd.read_csv('bike_features_train.csv')
    test_df = pd.read_csv('bike_features_test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Get feature columns (exclude identifiers and targets)
    exclude_cols = [
        'station_id', 'flow_date', 'flow_hour', 'station_type',
        'bikes_departed', 'bikes_arrived', 'net_flow', 'total_demand'
    ] + [col for col in train_df.columns if 'target_' in col]
    
    # Get numeric features only
    feature_cols = []
    for col in train_df.columns:
        if col not in exclude_cols and train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if column has any non-null values
            if train_df[col].notna().any():
                feature_cols.append(col)
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Create NetFlow if it doesn't exist
    if 'net_flow' not in train_df.columns:
        train_df['net_flow'] = train_df['bikes_arrived'] - train_df['bikes_departed']
    if 'net_flow' not in test_df.columns:
        test_df['net_flow'] = test_df['bikes_arrived'] - test_df['bikes_departed']
    
    # Sort by station and time
    train_df = train_df.sort_values(['station_id', 'flow_date', 'flow_hour'])
    test_df = test_df.sort_values(['station_id', 'flow_date', 'flow_hour'])
    
    # Create target: NetFlow 2 hours ahead
    train_df['target_net_flow_2h'] = train_df.groupby('station_id')['net_flow'].shift(-2)
    test_df['target_net_flow_2h'] = test_df.groupby('station_id')['net_flow'].shift(-2)
    
    # Remove rows with NaN targets
    train_mask = train_df['target_net_flow_2h'].notna()
    test_mask = test_df['target_net_flow_2h'].notna()
    
    # Prepare final datasets
    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, 'target_net_flow_2h']
    X_test = test_df.loc[test_mask, feature_cols]
    y_test = test_df.loc[test_mask, 'target_net_flow_2h']
    
    # Store metadata
    test_metadata = test_df.loc[test_mask, ['station_id', 'flow_date', 'flow_hour', 'net_flow']]
    
    print(f"\nFinal dataset sizes:")
    print(f"Train: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")
    print(f"\nTarget statistics:")
    print(f"Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"Test - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    return X_train, y_train, X_test, y_test, feature_cols, test_metadata

def train_netflow_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model for NetFlow prediction"""
    print("\n🚀 Training NetFlow XGBoost model...")
    print("-" * 60)
    
    # XGBoost parameters - include early_stopping_rounds here for newer versions
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'auto',
        'early_stopping_rounds': 20  # Add it here for newer XGBoost versions
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    # Train with evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # For newer XGBoost versions, early_stopping_rounds is in the constructor
    # So we just pass eval_set to fit()
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # Handle different XGBoost versions for best_iteration
    try:
        print(f"\nBest iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score:.4f}")
    except AttributeError:
        # For newer versions
        print(f"\nModel trained for {model.n_estimators} iterations")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, test_metadata):
    """Comprehensive model evaluation"""
    print("\n📈 Model Evaluation")
    print("=" * 60)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    
    print(f"\nPerformance Metrics:")
    print(f"{'Metric':<15} {'Train':<10} {'Test':<10}")
    print("-" * 35)
    print(f"{'MAE':<15} {metrics['train_mae']:<10.3f} {metrics['test_mae']:<10.3f}")
    print(f"{'RMSE':<15} {metrics['train_rmse']:<10.3f} {metrics['test_rmse']:<10.3f}")
    print(f"{'R²':<15} {metrics['train_r2']:<10.3f} {metrics['test_r2']:<10.3f}")
    
    # Analyze by hour
    test_analysis = test_metadata.copy()
    test_analysis['prediction'] = test_pred
    test_analysis['actual'] = y_test.values
    test_analysis['error'] = np.abs(test_analysis['actual'] - test_analysis['prediction'])
    
    hourly_mae = test_analysis.groupby('flow_hour')['error'].mean()
    
    print("\n⏰ Error by Hour (Top 5 worst):")
    print(hourly_mae.sort_values(ascending=False).head())
    
    return metrics, test_pred, test_analysis

def plot_results(model, feature_cols, y_test, test_pred, test_analysis):
    """Create visualization plots"""
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Predicted
    ax = axes[0, 0]
    scatter = ax.scatter(y_test, test_pred, alpha=0.5, s=20, c=np.abs(y_test - test_pred), cmap='viridis')
    ax.plot([-20, 20], [-20, 20], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Net Flow')
    ax.set_ylabel('Predicted Net Flow')
    ax.set_title('NetFlow Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Absolute Error')
    
    # 2. Error distribution
    ax = axes[0, 1]
    errors = y_test - test_pred
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (MAE: {mean_absolute_error(y_test, test_pred):.2f})')
    ax.grid(True, alpha=0.3)
    
    # 3. Feature importance (top 15)
    ax = axes[1, 0]
    importance_data = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    ax.barh(range(len(importance_data)), importance_data['importance'], color='coral')
    ax.set_yticks(range(len(importance_data)))
    ax.set_yticklabels(importance_data['feature'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Most Important Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Hourly MAE
    ax = axes[1, 1]
    hourly_mae = test_analysis.groupby('flow_hour')['error'].mean()
    bars = ax.bar(hourly_mae.index, hourly_mae.values, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error by Hour')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight worst hours
    worst_hours = hourly_mae.nlargest(3)
    for hour in worst_hours.index:
        bars[hour].set_color('red')
    
    plt.tight_layout()
    plt.savefig('netflow_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_data

def save_model_artifacts(model, feature_cols, metrics, importance_data):
    """Save model and related artifacts"""
    print("\n💾 Saving model artifacts...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save model
    model_path = f'models/netflow_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # 2. Save configuration
    config = {
        'model_type': 'netflow_prediction',
        'timestamp': timestamp,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'model_params': model.get_params(),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'top_features': importance_data.head(10).to_dict('records')
    }
    
    config_path = f'models/netflow_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    # 3. Save feature importance
    importance_path = f'models/feature_importance_{timestamp}.csv'
    importance_data.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    
    return model_path, config_path

def demonstrate_production_usage():
    """Show how to use the model in production"""
    print("\n🎯 Production Usage Example")
    print("=" * 60)
    print("""
# In production, you would:

1. Load the model:
   ```python
   model = joblib.load('models/netflow_model.pkl')
   ```

2. Get current bike count from API:
   ```python
   current_bikes = api.get_available_bikes(station_id)  # e.g., 8
   station_capacity = api.get_station_capacity(station_id)  # e.g., 25
   ```

3. Prepare features for current time:
   ```python
   features = prepare_current_features(station_id, datetime.now())
   ```

4. Predict net flow for next 2 hours:
   ```python
   net_flow_2h = model.predict(features)[0]  # e.g., -5
   ```

5. Calculate future availability:
   ```python
   future_bikes = current_bikes + net_flow_2h  # 8 + (-5) = 3
   ```

6. Generate alerts:
   ```python
   if future_bikes <= 2:
       send_alert(f"Station {station_id} will be EMPTY in 2 hours!")
   elif future_bikes >= station_capacity - 2:
       send_alert(f"Station {station_id} will be FULL in 2 hours!")
   ```
""")

# Main execution
if __name__ == "__main__":
    print("🚴 Seoul Bike NetFlow Prediction Model Training")
    print("=" * 60)
    
    try:
        # 1. Prepare data
        X_train, y_train, X_test, y_test, feature_cols, test_metadata = prepare_netflow_data()
        
        # 2. Train model
        model = train_netflow_model(X_train, y_train, X_test, y_test)
        
        # 3. Evaluate model
        metrics, test_pred, test_analysis = evaluate_model(
            model, X_train, y_train, X_test, y_test, test_metadata
        )
        
        # 4. Create visualizations
        importance_data = plot_results(model, feature_cols, y_test, test_pred, test_analysis)
        
        # 5. Save artifacts
        model_path, config_path = save_model_artifacts(model, feature_cols, metrics, importance_data)
        
        # 6. Show usage example
        demonstrate_production_usage()
        
        print("\n✅ Training completed successfully!")
        print(f"\nModel performance summary:")
        print(f"- Test MAE: {metrics['test_mae']:.2f} bikes")
        print(f"- Test R²: {metrics['test_r2']:.3f}")
        print(f"\nModel saved to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()