import os
import sys
import time
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "nifty50_processed_features.xlsx")
backup_path = os.path.join(BASE_DIR, "nifty50_processed_features_backup.xlsx")

# Nifty 50 stocks
NIFTY50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "KOTAKBANK", "BHARTIARTL",
    "ITC", "LT", "ASIANPAINT", "HINDUNILVR", "MARUTI", "AXISBANK", "BAJFINANCE",
    "BAJAJFINSV", "SBIN", "NTPC", "POWERGRID", "ULTRACEMCO", "NESTLEIND", "BRITANNIA",
    "M&M", "SUNPHARMA", "DIVISLAB", "INDUSINDBK", "TATAMOTORS", "TITAN", "DRREDDY",
    "GRASIM", "ADANIPORTS", "ADANIENT", "ADANIGREEN", "ADANITRANS", "VEDL", "SHREECEM",
    "BAJAJAUTO", "HEROMOTOCO", "WIPRO", "TECHM", "COALINDIA", "BPCL", "GAIL", "IOC",
    "UPL", "EICHERMOT"
]

def calculate_features(df):
    """Calculate technical indicators and features for the dataset"""
    if df.empty:
        print("❌ Empty dataframe provided to calculate_features")
        return df
        
    try:
        # Make sure we have the right columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print("❌ Missing required columns in dataframe")
            return df
            
        # Basic price features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        
        # Moving averages
        df['SMA5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA5'] = EMAIndicator(df['Close'], window=5).ema_indicator()
        df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        
        # Volume features
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # VWAP
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Volatility
        df['Volatility'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Fill NaN values
        df = df.fillna(method='ffill')
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        print(f"❌ Error calculating features: {str(e)}")
        return df

def fetch_stock_data(symbol, period="1y", interval="1d"):
    """Fetch OHLCV data for a single stock"""
    try:
        # Add .NS suffix for NSE stocks
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"⚠️ No data returned for {symbol}")
            return None
            
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Add symbol column
        df['Symbol'] = symbol
        
        print(f"✅ Fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {str(e)}")
        return None

def update_excel_file():
    """Update the Excel file with latest market data"""
    try:
        print(f"\n{'='*80}\n📊 Updating market data Excel file\n{'='*80}")
        
        # Create backup of existing file
        if os.path.exists(excel_path):
            print("📦 Creating backup of existing Excel file...")
            try:
                pd.read_excel(excel_path).to_excel(backup_path, index=False)
                print("✅ Backup created successfully")
            except Exception as e:
                print(f"⚠️ Error creating backup: {str(e)}")
        
        # Fetch and process data for all stocks
        all_data = []
        for symbol in NIFTY50_STOCKS:
            print(f"\n🔄 Processing {symbol}...")
            df = fetch_stock_data(symbol)
            if df is not None:
                df = calculate_features(df)
                all_data.append(df)
                print(f"✅ Added {len(df)} processed records for {symbol}")
            else:
                print(f"⚠️ Skipping {symbol} due to data fetch error")
        
        if not all_data:
            print("❌ No data collected for any stock")
            return False
            
        # Combine all data
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save to Excel
        print(f"\n💾 Saving {len(final_df)} records to Excel...")
        final_df.to_excel(excel_path, index=False)
        print(f"✅ Data saved successfully to {excel_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating Excel file: {str(e)}")
        return False

if __name__ == "__main__":
    update_excel_file()