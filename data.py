import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from ta.trend import MACD, EMAIndicator, ADXIndicator, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
 
# Constants - Use environment variable or default path
excel_path = os.environ.get(
    "NIFTY50_EXCEL_PATH", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty50_processed_features.xlsx")
)
# Check if old path is available
if not os.path.exists(excel_path) and os.path.exists(r"C:\Users\Admin\Desktop\Darsh\Share Market\nifty50_processed_features.xlsx"):
    excel_path = r"C:\Users\Admin\Desktop\Darsh\Share Market\nifty50_processed_features.xlsx"
    
nifty50_tickers = [
    "RELIANCE", "TCS", "HDFC", "INFY", "HDFCBANK", "ICICIBANK",
    "KOTAKBANK", "BHARTIARTL", "ITC", "LT", "ASIANPAINT", "HINDUNILVR",
    "MARUTI", "AXISBANK", "BAJFINANCE", "BAJAJFINSV", "SBIN", "NTPC",
    "POWERGRID", "ULTRACEMCO", "NESTLEIND", "BRITANNIA", "M&M",
    "SUNPHARMA", "DIVISLAB", "INDUSINDBK", "TATAMOTORS", "TITAN",
    "DRREDDY", "GRASIM", "ADANIPORTS", "ADANIENT", "ADANIGREEN", "ADANITRANS",
    "VEDL", "SHREECEM", "BAJAJ-AUTO", "HEROMOTOCO", "WIPRO", "TECHM",
    "COALINDIA", "BPCL", "GAIL", "IOC", "UPL", "EICHERMOT"
]
tickers_yf = [ticker + ".NS" for ticker in nifty50_tickers]
ticker_map = dict(zip(nifty50_tickers, tickers_yf))
 
# Feature calculation function
def calculate_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
 
    df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['MACD_HIST'] = macd.macd_diff()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'])
    df['BOLL_UPPER'] = bb.bollinger_hband()
    df['BOLL_LOWER'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['PIVOT'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['PIVOT'] - df['Low']
    df['S1'] = 2 * df['PIVOT'] - df['High']
    df['R2'] = df['PIVOT'] + (df['High'] - df['Low'])
    df['S2'] = df['PIVOT'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['PIVOT'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['PIVOT'])
    diff = df['High'] - df['Low']
    df['FIB_R1'] = df['Close'] + 0.236 * diff
    df['FIB_R2'] = df['Close'] + 0.382 * diff
    df['FIB_R3'] = df['Close'] + 0.618 * diff
    df['FIB_S1'] = df['Close'] - 0.236 * diff
    df['FIB_S2'] = df['Close'] - 0.382 * diff
    df['FIB_S3'] = df['Close'] - 0.618 * diff
    h_l = df['High'] - df['Low']
    df['CAM_R1'] = df['Close'] + h_l * 1.1 / 12
    df['CAM_R2'] = df['Close'] + h_l * 1.1 / 6
    df['CAM_R3'] = df['Close'] + h_l * 1.1 / 4
    df['CAM_R4'] = df['Close'] + h_l * 1.1 / 2
    df['CAM_S1'] = df['Close'] - h_l * 1.1 / 12
    df['CAM_S2'] = df['Close'] - h_l * 1.1 / 6
    df['CAM_S3'] = df['Close'] - h_l * 1.1 / 4
    df['CAM_S4'] = df['Close'] - h_l * 1.1 / 2
    adx = ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    df['ADX_Change'] = df['ADX'].diff()
    sar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['SAR'] = sar.psar()
    df['SAR_Diff'] = df['Close'] - df['SAR']
    df['Rolling_ADX_5'] = df['ADX'].rolling(window=5).mean()
    df['Rolling_SAR_5'] = df['SAR_Diff'].rolling(window=5).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    df['Lag_1D'] = df['Close'].shift(1)
    df['Lag_3D'] = df['Close'].shift(3)
    df['Lag_5D'] = df['Close'].shift(5)
    df['Lag_10D'] = df['Close'].shift(10)
    df['RSI.1'] = df['RSI']
    df['RSI_flag'] = np.where(df['RSI'] > 70, 'Overbought',
                              np.where(df['RSI'] < 30, 'Oversold', 'Neutral'))
    df['LABEL'] = df['RSI_flag'].map({
        'Oversold': 'BUY',
        'Overbought': 'SELL',
        'Neutral': 'HOLD'
    })
    df.reset_index(inplace=True)
    return df
 
# Load existing Excel file
# If the file doesn't exist, create a new empty one with basic structure
if not os.path.exists(excel_path):
    print(f"⚠️ Excel file not found at {excel_path}, creating a new one")
    # Create base structure with sheets for each ticker
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    empty_df = pd.DataFrame(columns=['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for ticker in nifty50_tickers:
        empty_df.to_excel(writer, sheet_name=ticker, index=False)
    writer.close()
    print(f"✅ Created new Excel file at {excel_path}")

try:
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
except Exception as e:
    print(f"❌ Error reading Excel file: {e}")
    print("Creating empty sheet names...")
    sheet_names = nifty50_tickers
 
# Process each ticker
updated_data = {}
today = datetime.today().date()
 
for sheet in sheet_names:
    print(f"\n📈 Updating: {sheet}")
    try:
        existing_df = xls.parse(sheet)
        if existing_df.empty or 'Date' not in existing_df.columns:
            print(f"  ⚠️ No data or 'Date' column missing in {sheet}. Skipping.")
            continue
       
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        last_date = existing_df['Date'].max().date()
        fetch_from = last_date + timedelta(days=1)
       
        if fetch_from > today:
            print("  ✅ Already up to date.")
            updated_data[sheet] = existing_df
            continue
 
        print(f"  ⏳ Fetching new data from {fetch_from} to {today}")
        ticker_yf = ticker_map[sheet]
        new_df = yf.download(ticker_yf, start=fetch_from, end=today + timedelta(days=1), interval='1d')
        new_df.reset_index(inplace=True)
 
        if new_df.empty:
            print("  ⚠️ No new data available.")
            updated_data[sheet] = existing_df
            continue
 
        new_df['Symbol'] = sheet
        new_df = new_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
        combined_df = pd.concat([existing_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']], new_df])
        combined_df.drop_duplicates(subset='Date', keep='last', inplace=True)
        combined_df = calculate_features(combined_df)
        updated_data[sheet] = combined_df
        print(f"  ✅ Updated rows: {len(new_df)}")
    except Exception as e:
        print(f"  ❌ Error updating {sheet}: {e}")
 
# Save back to the same Excel file
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    for sheet, df in updated_data.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
 
print(f"\n✅ All sheets updated in: {excel_path}")

# Main execution guard
if __name__ == "__main__":
    print(f"📊 Running data.py to update {excel_path}")
    # Core logic is already in the module level so we don't need to do anything else