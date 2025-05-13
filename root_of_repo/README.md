# 📈 Daily PPO Retraining for Nifty50 – Live Automation

## ⚙️ How It Works

1. **Daily GitHub Action triggers** (at 3:30 PM IST after market close)
2. **Fetch today's 1-min live OHLCV data** and **PCR data**
3. **Fetch latest news sentiment** from APIs
4. **Update each stock's PPO model** for 5,000 additional steps
5. **Save updated model + normalization statistics**
6. **Generate daily retraining report CSV**

---

## 🔥 Features

- Auto-updates all Nifty50 models without manual effort
- Skips training safely if market is closed
- Saves retrain logs as downloadable `.csv`
- Fully modular: Easy to extend to more stocks or different models
- Clean GitHub Actions automation

---

## 🛠 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
