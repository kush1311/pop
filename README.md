# PPO Reinforcement Learning for Nifty50 Trading

This repository contains a Proximal Policy Optimization (PPO) Reinforcement Learning system for stock trading. It automatically fetches real-time market data, news sentiment, and retrains models daily using GitHub Actions.

## 🤖 System Overview

The system performs daily continuous learning through these components:

1. **Data Collection**:
   - Market OHLCV data collection via `data.py`
   - News sentiment analysis via `newz.py`
   - PCR (Put-Call Ratio) data from NSE

2. **Model Training**:
   - Daily retraining of PPO models
   - Integration of sentiment data
   - Excel storage of historical features

3. **Automation**:
   - Daily updates via GitHub Actions (weekdays at 4:00 PM IST)
   - Artifact storage of daily results

## 📁 File Structure

- `train_ppo_realtime_multi.py` - Main PPO training module
- `data.py` - OHLCV data collection and feature calculation
- `newz.py` - Financial news sentiment analysis
- `run_daily_update.py` - Pipeline orchestration script
- `requirements.txt` - Python dependencies
- `.github/workflows/realtime_daily_update.yml` - GitHub Actions workflow

## 🔄 Continuous Learning Process

1. The GitHub Actions workflow triggers daily at 10:30 AM UTC (4:00 PM IST) on weekdays
2. `run_daily_update.py` executes:
   - Updates the Excel file with latest market data via `data.py`
   - Runs PPO retraining with the new data via `train_ppo_realtime_multi.py`
3. Trained models and artifacts are saved to the repository

## 🛠️ Setup and Usage

### Manual Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the daily update process
python run_daily_update.py
```

### GitHub Actions

The system will run automatically via GitHub Actions. You can also trigger a manual run:

1. Go to the "Actions" tab in your repository
2. Select "Daily PPO Update" workflow
3. Click "Run workflow"

### Required Secrets

The following GitHub Secrets should be configured:
- `NEWSDATA_API_KEY`
- `FINNHUB_API_KEY`
- `GNEWS_API_KEY`

## 📊 Outputs

- `live_nifty50_features.csv` - Daily OHLCV data
- `daily_nifty50_summary.csv` - Summary statistics
- `nifty50_processed_features.xlsx` - Historical feature database
- `daily_reports/*.csv` - Retraining logs

## 📚 Dependencies

- stable-baselines3 - PPO algorithm implementation
- gym - RL environment
- pandas, numpy - Data manipulation
- yfinance - Market data access
- textblob - Sentiment analysis
- ta - Technical indicators 