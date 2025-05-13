import time
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
 
def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    try:
        df = (
            yf.Ticker(f"{ticker}.NS")
              .history(period="7d", interval="1m", auto_adjust=False)
              .dropna()
        )
        if df.empty:
            print(f"⚠️ {ticker}: no OHLCV data, skipping.")
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
        return df
    except Exception as e:
        print(f"❌ {ticker}: OHLCV error ({e}), skipping.")
        return None
 
def fetch_live_pcr(symbol: str = "NIFTY", retries: int = 3, pause: float = 1.0) -> dict:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com",
    }
    for attempt in range(1, retries + 1):
        try:
            with requests.Session() as sess:
                sess.headers.update(headers)
                sess.get("https://www.nseindia.com", timeout=5)
                res = sess.get(url, timeout=5)
                data = res.json()
            recs = data["records"]["data"]
            put_oi = sum(r["PE"]["openInterest"] for r in recs if "PE" in r)
            call_oi = sum(r["CE"]["openInterest"] for r in recs if "CE" in r)
            pcr = round(put_oi / call_oi, 3) if call_oi else np.nan
            return {
                "timestamp": datetime.now(),
                "put_oi": put_oi,
                "call_oi": call_oi,
                "pcr": pcr
            }
        except Exception as e:
            print(f"⚠️ PCR attempt {attempt} failed: {e}")
            time.sleep(pause)
    print("❌ PCR fetch failed after retries → NaN values")
    return {"timestamp": datetime.now(), "put_oi": np.nan, "call_oi": np.nan, "pcr": np.nan}
 
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + roll_up / roll_down)
 
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
 
    df["BB_mid"] = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * std20
    df["BB_lo"] = df["BB_mid"] - 2 * std20
 
    df["prev_close"] = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["prev_close"]).abs()
    tr3 = (df["Low"] - df["prev_close"]).abs()
    df["ATR14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
 
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["cum_vol"] = df["Volume"].cumsum()
    df["cum_tp_vol"] = (tp * df["Volume"]).cumsum()
    df["VWAP"] = df["cum_tp_vol"] / df["cum_vol"]
 
    for m in (20, 50, 100):
        df[f"MA{m}"] = df["Close"].rolling(m).mean()
 
    return df.drop(columns=["prev_close", "cum_vol", "cum_tp_vol"])
 
def main():
    nifty50 = [
        "RELIANCE", "TCS", "HDFC", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
        "BHARTIARTL", "ITC", "LT", "ASIANPAINT", "HINDUNILVR", "MARUTI", "AXISBANK",
        "BAJFINANCE", "BAJAJFINSV", "SBIN", "NTPC", "POWERGRID", "ULTRACEMCO",
        "NESTLEIND", "BRITANNIA", "M&M", "SUNPHARMA", "DIVISLAB", "INDUSINDBK",
        "TATAMOTORS", "TITAN", "DRREDDY", "GRASIM", "ADANIPORTS", "ADANIENT",
        "ADANIGREEN", "ADANITRANS", "VEDL", "SHREECEM", "BAJAJAUTO", "HEROMOTOCO",
        "WIPRO", "TECHM", "COALINDIA", "BPCL", "GAIL", "IOC", "UPL", "EICHERMOT"
    ]
 
    summary_rows = []
    pcr = fetch_live_pcr()
    print(f"📊 Fetched PCR: PCR={pcr['pcr']} | Put={pcr['put_oi']} | Call={pcr['call_oi']}")
 
    for sym in nifty50:
        df = fetch_ohlcv(sym)
        if df is None:
            continue
 
        df = compute_features(df)
        summary_rows.append({
            "SYMBOL": sym,
            "DATE": df.index[0].date(),
            "OPEN": df["Open"].iloc[0],
            "HIGH": df["High"].max(),
            "LOW": df["Low"].min(),
            "CLOSE": df["Close"].iloc[-1],
            "VOLUME": int(df["Volume"].sum()),
            "Put_OI": int(pcr["put_oi"]),
            "Call_OI": int(pcr["call_oi"]),
            "PCR": pcr["pcr"]
        })
 
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("daily_nifty50_summary.csv", index=False)
        print("✅ Saved summary to daily_nifty50_summary.csv")
    else:
        print("❌ No daily summary generated.")
 
if __name__ == "__main__":
    main()
