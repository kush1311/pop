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
        return self._get_obs()

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
        pcr = fetch_live_pcr()
        print(f"\n📊 Fetched PCR once: PCR={pcr['pcr']} | Put_OI={pcr['put_oi']} | Call_OI={pcr['call_oi']}")

        for sym in nifty50:
            print(f"\n➡️ Fetching OHLCV for {sym}...")
            
            # Fetch data for the last 7 days
            df = yf.Ticker(f"{sym}.NS").history(period="7d", interval="1m").dropna()
            if df.empty:
                print(f"⚠️ {sym} skipped — No OHLCV data.")
                continue
                
            df.index = df.index.tz_convert("Asia/Kolkata")
            
            # Use calculate_features from data.py to process the data
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)
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
        return None

# === PPO Retraining Pipeline ===
def run_retraining():
    print("📈 Fetching real-time OHLCV data...")
    df = fetch_live_features()
    
    if df is None or df.empty:
        print("❌ OHLCV fetch failed or market closed.")
        return
        
    print("📰 Fetching news sentiment using newz.py...")
    news_df = get_news()

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

    pd.DataFrame(report).to_csv(os.path.join(REPORT_DIR, f"retrain_report_{datetime.now().date()}.csv"), index=False)
    print("\n✅ Retraining complete. Report saved.")
    print("📦 Final OHLCV snapshot:")
    print(df.groupby("Symbol").tail(1)[["Symbol", "Close", "PCR", "Timestamp"]])

if __name__ == "__main__":
    run_retraining()
