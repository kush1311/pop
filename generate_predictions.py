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

def prepare_forecast_data(df, days=90, lookback_days=20):
    """
    Prepare data for forecasting by extending the dataframe with future dates
    Uses the last lookback_days to create patterns for decision making
    """
    if df.empty:
        print("⚠️ Empty dataframe, cannot prepare forecast data")
        return None
        
    # Get the latest date in the dataframe
    latest_date = pd.to_datetime(df['Date']).max()
    
    # Ensure we have enough historical data (at least lookback_days)
    if len(df) < lookback_days:
        print(f"⚠️ Not enough historical data, need at least {lookback_days} days")
        # If we don't have enough data, use what we have
        lookback_days = len(df)
        
    # Get the last lookback_days of data to use as pattern
    historical_window = df.iloc[-lookback_days:].copy()
    
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
    
    # Add features based on historical patterns
    # For price data, we'll use a simple model that continues the recent trend
    if len(historical_window) >= 2:
        # Calculate average daily change over the lookback period
        avg_daily_change = (historical_window['Close'].iloc[-1] - historical_window['Close'].iloc[0]) / (len(historical_window) - 1)
        
        # Initialize with the last known price
        last_price = historical_window['Close'].iloc[-1]
        
        # Generate future prices
        future_prices = []
        for i in range(days):
            # Add some randomness based on historical volatility
            volatility = historical_window['Close'].pct_change().std()
            noise = np.random.normal(0, volatility * last_price) if not pd.isna(volatility) else 0
            
            # Calculate new price
            new_price = last_price + avg_daily_change + noise
            future_prices.append(max(0.1, new_price))  # Ensure price doesn't go negative
            last_price = new_price
    else:
        # Not enough data for trend, just use last price
        future_prices = [df['Close'].iloc[-1]] * days
    
    # Add the forecasted prices
    future_df['Close'] = future_prices
    
    # Generate other price columns based on the Close price
    future_df['Open'] = future_df['Close'] * (1 + np.random.normal(0, 0.005, size=days))
    future_df['High'] = future_df['Close'] * (1 + abs(np.random.normal(0, 0.01, size=days)))
    future_df['Low'] = future_df['Close'] * (1 - abs(np.random.normal(0, 0.01, size=days)))
    
    # Copy over other indicators from historical data
    for col in df.columns:
        if col not in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']:
            future_df[col] = historical_window[col].iloc[-1]
    
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
        
        # Check if we have enough data for meaningful predictions
        if len(stock_data) < 5:  # At least 5 days of data
            print(f"⚠️ Not enough historical data for {symbol}, need at least 5 days")
            return None
            
        print(f"📊 Using last {len(stock_data)} days of data for {symbol} to predict next {FORECAST_DAYS} days")
            
        # Prepare forecast data with 20-day lookback pattern
        future_data = prepare_forecast_data(stock_data, days=FORECAST_DAYS, lookback_days=20)
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
                'Close': future_data['Close'].iloc[day],
                'Forecasted_Price': round(future_data['Close'].iloc[day], 2)
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