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
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.n_steps = len(self.df) - 1
        self.position = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._get_obs()),), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = row.drop(["Symbol", "Datetime"], errors="ignore").values
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

# === Fetch All Nifty50 Features - Now using data.py ===
def fetch_live_features():
    try:
        # Importing os if not already imported
        import os
        from data import calculate_features
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
                # Fetch data for the last 7 days
                df = yf.Ticker(f"{sym}.NS").history(period="7d", interval="1d").dropna()
                if df.empty:
                    print(f"⚠️ {sym} skipped — No OHLCV data.")
                    continue
                    
                # Make sure we have a Date column for calculate_features
                df.reset_index(inplace=True)
                
                # Make sure the Date column is named correctly
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df.rename(columns={"Datetime": "Date"}, inplace=True)
                elif 'Date' not in df.columns and 'index' in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)
                
                # Process data with features
                df = calculate_features(df)
                
                # Add PCR data
                df["PCR"] = pcr["pcr"]
                df["Put_OI"] = pcr["put_oi"]
                df["Call_OI"] = pcr["call_oi"]
                df["Timestamp"] = pcr["timestamp"]
                df["Symbol"] = sym

                print(f"✅ {sym} - Last 3 OHLCV rows:")
                print(df[["Close", "High", "Low", "Volume"]].tail(3))

                rows.append(df)

                summary_rows.append({
                    "SYMBOL": sym,
                    "DATE": pd.to_datetime(df['Date'].iloc[0]).date(),
                    "OPEN": df["Open"].iloc[0],
                    "HIGH": df["High"].max(),
                    "LOW": df["Low"].min(),
                    "CLOSE": df["Close"].iloc[-1],
                    "VOLUME": int(df["Volume"].sum()),
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
            news_df = get_news()
        except Exception as e:
            print(f"⚠️ Error fetching news, using empty dataframe: {e}")
            news_df = pd.DataFrame()

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
                env = DummyVecEnv([lambda: DynamicTradingEnv(data, data, news_df, None, None)])
                env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)
                model.set_env(env)
                model.learn(total_timesteps=5000)
                model.save(model_path)
                env.save(env_path)
                print(f"✅ Retrained model for {ticker}")
                report.append({"Ticker": ticker, "Status": "Updated", "Time": str(datetime.now())})
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
