import os
import sys
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
from gym import spaces

# Use data.py and newz.py instead of the realtime_data modules
from data import calculate_features
from newz import get_news

# === Custom Gym Trading Environment ===
class DynamicTradingEnv(gym.Env):
    def __init__(self, df_daily, df_pcr, df_sentiment, df_quarterly, xgb_model, initial_balance=100000, dummy=False):
        super(DynamicTradingEnv, self).__init__()
        self.df = df_daily.copy().reset_index(drop=True)
        self.df_sentiment = df_sentiment.copy() if df_sentiment is not None else pd.DataFrame()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.n_steps = len(self.df) - 1
        self.position = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._get_obs()),), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Basic features from the dataframe
        obs = row.drop(["Symbol", "Datetime", "Date"], errors="ignore").values
        
        # Add sentiment data for current date if available
        if not self.df_sentiment.empty and 'Date' in self.df.columns:
            current_date = self.df.iloc[self.current_step]['Date']
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
                
            # Find closest sentiment data
            if 'DATE' in self.df_sentiment.columns:
                sent_row = self.df_sentiment[self.df_sentiment['DATE'] <= current_date].sort_values('DATE').tail(1)
                if not sent_row.empty and 'sentiment_score' in self.df_sentiment.columns:
                    sentiment_value = sent_row['sentiment_score'].values[0]
                    obs = np.append(obs, sentiment_value)
                else:
                    obs = np.append(obs, 0)  # Default neutral sentiment
        
        return np.nan_to_num(obs, nan=0.0)

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        try:
            # Make sure we have enough data for training
            if len(self.df) <= 1:
                print("⚠️ Warning: DataFrame has insufficient data for training")
                # Create a dummy observation of appropriate size
                dummy_obs = np.zeros(self.observation_space.shape[0])
                return dummy_obs
            
            return self._get_obs()
        except Exception as e:
            print(f"❌ Error in reset: {e}")
            # Return a zero observation of appropriate size
            dummy_obs = np.zeros(self.observation_space.shape[0])
            return dummy_obs

    def step(self, action):
        prev_price = self.df.iloc[self.current_step]["Close"]
        self.current_step += 1
        done = self.current_step >= self.n_steps
        current_price = self.df.iloc[self.current_step]["Close"]
        reward = (current_price - prev_price) if action == 2 else (prev_price - current_price) if action == 0 else 0
        self.position = 1 if action == 2 else -1 if action == 0 else self.position
        return self._get_obs(), reward, done, {}

# === Directory Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models_with_xgb")
ENV_DIR = os.path.join(BASE_DIR, "saved_envs")
REPORT_DIR = os.path.join(BASE_DIR, "daily_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# === PCR Fetching Function ===
def fetch_live_pcr(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com"
    }
    try:
        with requests.Session() as sess:
            sess.headers.update(headers)
            sess.get("https://www.nseindia.com", timeout=5)
            res = sess.get(url, timeout=5)
            data = res.json()
        recs = data["records"]["data"]
        put_oi = sum(r["PE"]["openInterest"] for r in recs if "PE" in r)
        call_oi = sum(r["CE"]["openInterest"] for r in recs if "CE" in r)
        return {"put_oi": put_oi, "call_oi": call_oi, "pcr": round(put_oi / call_oi, 3), "timestamp": datetime.now()}
    except Exception as e:
        print(f"⚠️ PCR fetch failed: {e}")
        try:
            print(f"⚠️ Raw response content: {res.text[:500]}")
        except:
            pass
        return {"put_oi": np.nan, "call_oi": np.nan, "pcr": np.nan, "timestamp": datetime.now()}

# === Simple Feature Calculation for GitHub Actions ===
def calculate_features_simple(df):
    """
    A simplified version of feature calculation for GitHub Actions environment
    that's more robust with smaller datasets
    """
    df = df.copy()
    
    # Make sure date is properly formatted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic features that don't require a lot of history
    df['EMA5'] = df['Close'].ewm(span=5).mean()
    df['SMA10'] = df['Close'].rolling(window=min(5, len(df) - 1)).mean()
    
    # Daily returns - handle small datasets
    if len(df) > 1:
        df['Daily_Return'] = df['Close'].pct_change(fill_method=None)
    else:
        df['Daily_Return'] = 0
        
    # Volatility - use smaller window for small datasets
    window_size = min(5, max(2, len(df) - 1))
    if len(df) > 1:
        df['Volatility'] = df['Close'].rolling(window=window_size).std()
    else:
        df['Volatility'] = 0
        
    # RSI calculation with small dataset handling
    if len(df) > 1:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=min(14, len(df) - 1)).mean()
        avg_loss = loss.rolling(window=min(14, len(df) - 1)).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = 50  # Neutral RSI for single records
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

# === Fetch All Nifty50 Features - Now using data.py ===
def fetch_live_features():
    try:
        # Importing os if not already imported
        import os
        try:
            from data import calculate_features
            print("Using full feature calculation from data.py")
        except Exception as e:
            print(f"⚠️ Error importing calculate_features from data.py: {e}")
            print("Will use simplified feature calculation")
            calculate_features = calculate_features_simple
            
        import yfinance as yf

        nifty50 = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "LT",
            "ASIANPAINT", "HINDUNILVR", "MARUTI", "AXISBANK", "BAJFINANCE", "BAJAJFINSV", "SBIN", "NTPC",
            "POWERGRID", "ULTRACEMCO", "NESTLEIND", "BRITANNIA", "M&M", "SUNPHARMA", "DIVISLAB", "INDUSINDBK",
            "TATAMOTORS", "TITAN", "DRREDDY", "GRASIM", "ADANIPORTS", "ADANIENT", "ADANIGREEN", "ADANITRANS",
            "VEDL", "SHREECEM", "BAJAJAUTO", "HEROMOTOCO", "WIPRO", "TECHM", "COALINDIA", "BPCL", "GAIL", "IOC", "UPL", "EICHERMOT"
        ]

        rows = []
        summary_rows = []
        
        # Try to get PCR data, use default values if it fails
        try:
            pcr = fetch_live_pcr()
            if math.isnan(pcr["pcr"]):
                # PCR fetch failed, provide default data
                print("⚠️ PCR fetch failed, using default values for GitHub Actions")
                pcr = {"put_oi": 1000000, "call_oi": 900000, "pcr": 1.11, "timestamp": datetime.now()}
        except Exception as e:
            print(f"⚠️ Error fetching PCR, using default values: {e}")
            pcr = {"put_oi": 1000000, "call_oi": 900000, "pcr": 1.11, "timestamp": datetime.now()}
            
        print(f"\n📊 PCR data: PCR={pcr['pcr']} | Put_OI={pcr['put_oi']} | Call_OI={pcr['call_oi']}")

        for sym in nifty50:
            print(f"\n➡️ Fetching OHLCV for {sym}...")
            
            try:
                # Fetch data for the last 30 days to ensure enough data
                df = yf.Ticker(f"{sym}.NS").history(period="30d", interval="1d").dropna()
                if df.empty:
                    print(f"⚠️ {sym} skipped — No OHLCV data.")
                    continue
                    
                print(f"✅ Downloaded {len(df)} days of data for {sym}")
                
                # Make sure we have a Date column for calculate_features
                df.reset_index(inplace=True)
                
                # Make sure the Date column is named correctly
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df.rename(columns={"Datetime": "Date"}, inplace=True)
                elif 'Date' not in df.columns and 'index' in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)
                
                # Check if we have enough data
                if len(df) < 2:
                    print(f"⚠️ Not enough data for {sym}, need at least 2 days.")
                    continue
                
                # Process data with features
                try:
                    df = calculate_features(df)
                except Exception as e:
                    print(f"⚠️ Error calculating features, using simplified calculation: {e}")
                    df = calculate_features_simple(df)
                
                # Add PCR data
                df["PCR"] = pcr["pcr"]
                df["Put_OI"] = pcr["put_oi"]
                df["Call_OI"] = pcr["call_oi"]
                df["Timestamp"] = pcr["timestamp"]
                df["Symbol"] = sym

                print(f"✅ {sym} data processed successfully. Shape: {df.shape}")
                if len(df) > 3:
                    print(df[["Date", "Close", "High", "Low", "Volume"]].tail(3))

                rows.append(df)

                summary_rows.append({
                    "SYMBOL": sym,
                    "DATE": pd.to_datetime(df['Date'].iloc[0]).date() if len(df) > 0 else datetime.now().date(),
                    "OPEN": df["Open"].iloc[0] if len(df) > 0 else 0,
                    "HIGH": df["High"].max() if len(df) > 0 else 0,
                    "LOW": df["Low"].min() if len(df) > 0 else 0,
                    "CLOSE": df["Close"].iloc[-1] if len(df) > 0 else 0,
                    "VOLUME": int(df["Volume"].sum()) if len(df) > 0 else 0,
                    "Put_OI": int(pcr["put_oi"]) if not math.isnan(pcr["put_oi"]) else 0,
                    "Call_OI": int(pcr["call_oi"]) if not math.isnan(pcr["call_oi"]) else 0,
                    "PCR": pcr["pcr"] if not math.isnan(pcr["pcr"]) else 0.0
                })
            except Exception as e:
                print(f"⚠️ Error fetching data for {sym}: {e}")
                continue

        if rows:
            final_df = pd.concat(rows).reset_index(drop=True)
            final_df.to_csv(os.path.join(BASE_DIR, "live_nifty50_features.csv"), index=False)
            final_df.to_csv(os.path.join(BASE_DIR, "debug_ohlcv_sample.csv"), index=False)
            print(f"\n✅ Saved full OHLCV data to 'live_nifty50_features.csv' and 'debug_ohlcv_sample.csv'")
        else:
            print("❌ No valid OHLCV data collected for any stock.")

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(os.path.join(BASE_DIR, "daily_nifty50_summary.csv"), index=False)
            print("📊 Saved daily summary → daily_nifty50_summary.csv")
        else:
            print("❌ No daily summary generated.")
            
        return final_df if rows else None
    except Exception as e:
        print(f"❌ Error in fetch_live_features: {e}")
        import traceback
        traceback.print_exc()
        return None

# === PPO Retraining Pipeline ===
def run_retraining():
    try:
        print("📈 Fetching real-time OHLCV data...")
        df = fetch_live_features()
        
        if df is None or df.empty:
            print("❌ OHLCV fetch failed or market closed.")
            return
            
        print("📰 Fetching news sentiment using newz.py...")
        try:
            # Get latest news sentiment
            news_df = get_news()
            
            # Prepare news data for model training
            if not news_df.empty and 'published' in news_df.columns:
                # Convert to proper format
                news_df['DATE'] = pd.to_datetime(news_df['published'])
                
                # Check if we have historical sentiment data we should combine with
                sentiment_path = os.path.join(BASE_DIR, "Labeled_News_Sentiment_Data.csv")
                if os.path.exists(sentiment_path):
                    try:
                        historical_sentiment = pd.read_csv(sentiment_path)
                        historical_sentiment['DATE'] = pd.to_datetime(historical_sentiment['DATE'])
                        
                        # Combine with new data, keeping latest values when duplicates exist
                        combined_news = pd.concat([
                            historical_sentiment, 
                            news_df[['DATE', 'sentiment_score']]
                        ])
                        
                        # Remove duplicates by date, keeping newest data
                        news_df = combined_news.drop_duplicates(subset=['DATE'], keep='last')
                        
                        print(f"✅ Combined {len(historical_sentiment)} historical and {len(news_df) - len(historical_sentiment)} new sentiment records")
                    except Exception as e:
                        print(f"⚠️ Error combining with historical sentiment: {e}")
                        # Keep only the newly fetched news
                        news_df = news_df[['DATE', 'sentiment_score']]
                else:
                    # No historical data, just use what we fetched
                    news_df = news_df[['DATE', 'sentiment_score']]
                    
                # Save updated sentiment data for future use
                try:
                    news_df.to_csv(sentiment_path, index=False)
                    print(f"✅ Saved {len(news_df)} sentiment records to {sentiment_path}")
                except Exception as e:
                    print(f"⚠️ Error saving sentiment data: {e}")
            else:
                print("⚠️ No news data available")
                # Create empty DataFrame with correct structure
                news_df = pd.DataFrame(columns=['DATE', 'sentiment_score'])
        except Exception as e:
            print(f"⚠️ Error fetching news, using empty dataframe: {e}")
            news_df = pd.DataFrame(columns=['DATE', 'sentiment_score'])

        # Try to update the Excel file with new data
        try:
            # This assumes that nifty50_processed_features.xlsx exists
            # and data.py can update it
            print("📊 Updating nifty50_processed_features.xlsx...")
            from data import excel_path
            if os.path.exists(excel_path):
                print(f"✅ Found Excel file at {excel_path}")
            else:
                print(f"⚠️ Excel file not found at {excel_path}, will continue without updating it")
        except Exception as e:
            print(f"⚠️ Could not update Excel file: {e}")

        report = []
        # Ensure MODEL_DIR exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if directory is empty
        if not os.listdir(MODEL_DIR):
            print(f"⚠️ No models found in {MODEL_DIR}. Skipping retraining.")
            return
            
        for model_file in os.listdir(MODEL_DIR):
            if not model_file.endswith(".zip"):
                continue
            ticker = model_file.replace("ppo_rl_xgb_", "").replace(".zip", "").upper()
            model_path = os.path.join(MODEL_DIR, model_file)
            env_path = os.path.join(ENV_DIR, f"vecnormalize_{ticker.lower()}.pkl")

            if not os.path.exists(env_path):
                print(f"⚠️ Missing env for {ticker}, skipping.")
                continue

            data = df[df["Symbol"] == ticker]
            if data.empty:
                print(f"⚠️ No OHLCV for {ticker}, skipping.")
                continue

            try:
                model = PPO.load(model_path)
                # Create environment with news data included
                env = DummyVecEnv([lambda: DynamicTradingEnv(
                    df_daily=data, 
                    df_pcr=data,  # Using same data for PCR, could be separated
                    df_sentiment=news_df,  # Pass the news data
                    df_quarterly=None, 
                    xgb_model=None
                )])
                env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)
                model.set_env(env)
                model.learn(total_timesteps=5000)
                model.save(model_path)
                env.save(env_path)
                print(f"✅ Retrained model for {ticker} with news data")
                report.append({"Ticker": ticker, "Status": "Updated with news data", "Time": str(datetime.now())})
            except Exception as e:
                print(f"❌ {ticker} retrain error: {e}")
                report.append({"Ticker": ticker, "Status": f"Error: {str(e)}", "Time": str(datetime.now())})

        # Make sure REPORT_DIR exists
        os.makedirs(REPORT_DIR, exist_ok=True)
        
        pd.DataFrame(report).to_csv(os.path.join(REPORT_DIR, f"retrain_report_{datetime.now().date()}.csv"), index=False)
        print("\n✅ Retraining complete. Report saved.")
        
        if not df.empty:
            print("📦 Final OHLCV snapshot:")
            try:
                summary = df.groupby("Symbol").tail(1)[["Symbol", "Close"]]
                if "PCR" in df.columns:
                    summary["PCR"] = df.groupby("Symbol").tail(1)["PCR"]
                if "Timestamp" in df.columns:
                    summary["Timestamp"] = df.groupby("Symbol").tail(1)["Timestamp"]
                print(summary)
            except Exception as e:
                print(f"⚠️ Error displaying summary: {e}")
                
    except Exception as e:
        print(f"❌ Error in run_retraining: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_retraining()