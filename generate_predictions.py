#!/usr/bin/env python3
"""
Generate Daily Predictions using PPO Models
- Makes predictions for each stock for the next 90 days (3 months)
- Uses the trained PPO models from saved_models_with_xgb
- Saves predictions to a daily CSV file with date stamp
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ppo_realtime_multi import DynamicTradingEnv, fetch_live_features, calculate_features_simple

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models_with_xgb")
ENV_DIR = os.path.join(BASE_DIR, "saved_envs")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "daily_predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
FORECAST_DAYS = 90  # 3 months prediction

def prepare_forecast_data(df, days=90):
    """Prepare data for forecasting by extending the dataframe with future dates"""
    if df.empty:
        print("⚠️ Empty dataframe, cannot prepare forecast data")
        return None
        
    # Get the latest date in the dataframe
    latest_date = pd.to_datetime(df['Date']).max()
    
    # Create a dataframe with future dates
    future_dates = [latest_date + timedelta(days=i+1) for i in range(days)]
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Add the symbol
    if 'Symbol' in df.columns:
        symbol = df['Symbol'].iloc[0]
        future_df['Symbol'] = symbol
    else:
        print("⚠️ No Symbol column found in dataframe")
        future_df['Symbol'] = 'UNKNOWN'
    
    # Add placeholder data - we'll use the last known values
    last_row = df.iloc[-1].copy()
    for col in df.columns:
        if col not in ['Date', 'Symbol']:
            future_df[col] = last_row[col]
    
    # Ensure Date is datetime type
    future_df['Date'] = pd.to_datetime(future_df['Date'])
    
    return future_df

def predict_stock(symbol, real_data, models_dir=MODEL_DIR, env_dir=ENV_DIR):
    """Generate predictions for a single stock"""
    model_path = os.path.join(models_dir, f"ppo_rl_xgb_{symbol.lower()}.zip")
    env_path = os.path.join(env_dir, f"vecnormalize_{symbol.lower()}.pkl")
    
    if not os.path.exists(model_path):
        print(f"⚠️ No model found for {symbol}")
        return None
        
    if not os.path.exists(env_path):
        print(f"⚠️ No environment found for {symbol}")
        return None
    
    try:
        # Filter data for this symbol
        stock_data = real_data[real_data['Symbol'] == symbol].copy()
        if stock_data.empty:
            print(f"⚠️ No data found for {symbol}")
            return None
            
        # Prepare forecast data
        future_data = prepare_forecast_data(stock_data, days=FORECAST_DAYS)
        if future_data is None:
            print(f"⚠️ Failed to create forecast data for {symbol}")
            return None
            
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        env = DummyVecEnv([lambda: DynamicTradingEnv(future_data, future_data, None, None, None)])
        env = VecNormalize.load(env_path, env)
        env.training = False  # Disable training mode
        env.norm_reward = False  # Disable reward normalization
        
        # Generate predictions
        predictions = []
        obs = env.reset()
        
        for day in range(FORECAST_DAYS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            
            # Get the predicted position and confidence
            position = 1 if action == 2 else -1 if action == 0 else 0
            position_name = "BUY" if action == 2 else "SELL" if action == 0 else "HOLD"
            
            # Add prediction to list
            predictions.append({
                'Date': future_data['Date'].iloc[day],
                'Symbol': symbol,
                'Predicted_Action': position_name,
                'Position': position,
                'Action_Code': int(action),
                'Close': future_data['Close'].iloc[day]
            })
            
            if done:
                break
                
        # Convert predictions to dataframe
        predictions_df = pd.DataFrame(predictions)
        print(f"✅ Generated {len(predictions_df)} predictions for {symbol}")
        return predictions_df
        
    except Exception as e:
        print(f"❌ Error predicting {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_predictions():
    """Generate predictions for all stocks"""
    print(f"\n{'='*80}\n📈 GENERATING PREDICTIONS FOR NEXT {FORECAST_DAYS} DAYS\n{'='*80}")
    
    # Get current date for filename
    today = date.today().strftime('%Y%m%d')
    output_file = os.path.join(PREDICTIONS_DIR, f"predictions_{today}.csv")
    
    # Check if predictions were already generated today
    if os.path.exists(output_file):
        print(f"⚠️ Predictions for today already exist at {output_file}")
        print("📊 Loading existing predictions...")
        return pd.read_csv(output_file)
    
    # Fetch latest data
    print("📊 Fetching latest market data...")
    real_data = fetch_live_features()
    
    if real_data is None or real_data.empty:
        print("❌ Failed to fetch market data")
        return None
    
    # Get list of available models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
    symbols = [f.replace('ppo_rl_xgb_', '').replace('.zip', '').upper() for f in model_files]
    
    if not symbols:
        print("⚠️ No models found in directory")
        return None
    
    print(f"🔍 Found {len(symbols)} models: {', '.join(symbols)}")
    
    # Generate predictions for each stock
    all_predictions = []
    
    for symbol in symbols:
        print(f"\n🔮 Generating predictions for {symbol}...")
        stock_predictions = predict_stock(symbol, real_data)
        if stock_predictions is not None:
            all_predictions.append(stock_predictions)
    
    if not all_predictions:
        print("❌ No predictions generated")
        return None
    
    # Combine all predictions
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions
    combined_predictions.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(combined_predictions)} predictions to {output_file}")
    
    # Generate summary
    summary = combined_predictions.groupby(['Symbol', 'Predicted_Action']).size().unstack().fillna(0)
    print("\n📊 Prediction Summary:")
    print(summary)
    
    # Calculate recommendation score for each stock
    # +1 for each BUY, -1 for each SELL
    recommendations = combined_predictions.groupby('Symbol').apply(
        lambda x: (x['Position'].sum() / len(x))
    ).sort_values(ascending=False)
    
    print("\n💼 Stock Recommendations (range: -1 to +1):")
    for stock, score in recommendations.items():
        recommendation = "Strong Buy" if score > 0.5 else "Buy" if score > 0 else "Sell" if score > -0.5 else "Strong Sell"
        print(f"{stock}: {score:.2f} - {recommendation}")
    
    # Create a summary file with recommendations
    summary_file = os.path.join(PREDICTIONS_DIR, f"recommendations_{today}.csv")
    pd.DataFrame({
        'Symbol': recommendations.index,
        'Score': recommendations.values,
        'Recommendation': ['Strong Buy' if s > 0.5 else 'Buy' if s > 0 else 'Sell' if s > -0.5 else 'Strong Sell' for s in recommendations.values]
    }).to_csv(summary_file, index=False)
    print(f"\n✅ Saved recommendations to {summary_file}")
    
    return combined_predictions

if __name__ == "__main__":
    start_time = time.time()
    predictions = run_all_predictions()
    elapsed_time = time.time() - start_time
    print(f"\n✅ PREDICTIONS COMPLETED in {elapsed_time:.2f} seconds")
    
    if predictions is not None:
        print(f"📊 Generated {len(predictions)} predictions for {predictions['Symbol'].nunique()} stocks") 