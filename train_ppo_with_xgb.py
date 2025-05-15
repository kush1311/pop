import os
import pandas as pd
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from trading_env import DynamicTradingEnv
from sklearn.metrics import accuracy_score
from newz import get_news  # Import the news fetching function

print("✅ Starting train_ppo_with_xgb script with improved normalization and tuning")

# 1) Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
XGB_MODEL_DIR = os.path.join(PROJECT_ROOT, "models_per_sheet")

TECH_PATH      = os.path.join(BASE_DIR, "tech+rsi+price based features.xlsx")
PCR_PATH       = os.path.join(BASE_DIR, "updated_Nifty50_PCR_Features_with_label.xlsx")
SENTIMENT_PATH = os.path.join(BASE_DIR, "Labeled_News_Sentiment_Data.csv")
QUARTERLY_PATH = os.path.join(BASE_DIR, "Nifty50_Quarterly_Features.xlsx")

# 2) Load shared data
print("📥 Loading sentiment and quarterly datasets...")
# Load historical sentiment data
try:
    sentiment_df = pd.read_csv(SENTIMENT_PATH)
    print(f"✅ Loaded historical sentiment data: {len(sentiment_df)} records")
except Exception as e:
    print(f"⚠️ Error loading historical sentiment: {e}")
    sentiment_df = pd.DataFrame()

# Try to fetch latest news data and merge with historical data
try:
    print("📰 Fetching latest news sentiment data...")
    latest_news_df = get_news()
    
    # If we have both historical and new data, combine them
    if not sentiment_df.empty and not latest_news_df.empty:
        # Convert date columns to datetime for proper merging
        if 'DATE' in sentiment_df.columns and 'published' in latest_news_df.columns:
            sentiment_df['DATE'] = pd.to_datetime(sentiment_df['DATE'])
            latest_news_df['DATE'] = pd.to_datetime(latest_news_df['published'])
            
            # Make sure necessary columns exist in both dataframes
            required_cols = ['sentiment_score']
            for col in required_cols:
                if col not in sentiment_df.columns:
                    sentiment_df[col] = 0
                    
            # Merge the datasets, prioritizing newer data
            combined_news = pd.concat([sentiment_df, latest_news_df[['DATE', 'sentiment_score']]])
            sentiment_df = combined_news.drop_duplicates(subset=['DATE'], keep='last')
            print(f"✅ Combined with latest news data: {len(sentiment_df)} total records")
    elif not latest_news_df.empty:
        # If we only have new data, use that
        latest_news_df['DATE'] = pd.to_datetime(latest_news_df['published'])
        sentiment_df = latest_news_df[['DATE', 'sentiment_score']]
        print(f"✅ Using only latest news data: {len(sentiment_df)} records")
        
except Exception as e:
    print(f"⚠️ Error fetching latest news: {e}")
    # Continue with existing sentiment_df

quarterly_df = pd.read_excel(QUARTERLY_PATH)

# 3) Discover sheets and XGB models
tech_sheets = pd.ExcelFile(TECH_PATH).sheet_names
pcr_sheets  = pd.ExcelFile(PCR_PATH).sheet_names

# Map XGBoost model files named label_encoder_<TICKER>.pkl
xgb_models = {}
if os.path.isdir(XGB_MODEL_DIR):
    for fname in os.listdir(XGB_MODEL_DIR):
        low = fname.lower()
        if low.endswith('.pkl') and low.startswith('label_encoder_'):
            ticker = fname.replace('label_encoder_', '').replace('.pkl', '').strip().upper()
            xgb_models[ticker] = fname
print("🔍 Available XGBoost models:", list(xgb_models.keys()))

# 4) Create output directories
output_models_dir = os.path.join(BASE_DIR, 'saved_models_with_xgb')
output_logs_dir   = os.path.join(BASE_DIR, 'ppo_logs')
output_env_dir    = os.path.join(BASE_DIR, 'saved_envs')
for d in [output_models_dir, output_logs_dir, output_env_dir]:
    os.makedirs(d, exist_ok=True)

# 5) Train loop per ticker
for raw_ticker in tech_sheets:
    ticker = raw_ticker.strip().upper()
    print(f"\n🧩 Checking ticker: '{raw_ticker}' -> normalized '{ticker}'")

    # PCR sheet must exist
    if ticker not in pcr_sheets:
        print(f"❌ Skipping {ticker}: Missing PCR sheet")
        continue
    # XGB model must exist
    if ticker not in xgb_models:
        print(f"❌ Skipping {ticker}: Missing XGBoost model (found: {list(xgb_models.keys())})")
        continue

    # 6) Load data and XGB model
    try:
        tech_df   = pd.read_excel(TECH_PATH, sheet_name=raw_ticker)
        pcr_df    = pd.read_excel(PCR_PATH, sheet_name=raw_ticker)
        xgb_path  = os.path.join(XGB_MODEL_DIR, xgb_models[ticker])
        xgb_model = joblib.load(xgb_path)
        print(f"✅ Loaded XGBoost model for {ticker}: {xgb_models[ticker]}")
    except Exception as e:
        print(f"❌ Error loading data/model for {ticker}: {e}")
        continue

    # 7) Split train/test for XGB evaluation
    split_idx  = int(len(tech_df) * 0.8)
    tech_train = tech_df.iloc[:split_idx].reset_index(drop=True)
    pcr_train  = pcr_df.iloc[:split_idx].reset_index(drop=True)
    tech_test  = tech_df.iloc[split_idx:].reset_index(drop=True)

    # 8) Create VecEnv and normalize
    base_env = DummyVecEnv([lambda: DynamicTradingEnv(
        df_daily      = tech_train,
        df_pcr        = pcr_train,
        df_sentiment  = sentiment_df,  # Use the combined sentiment data
        df_quarterly  = quarterly_df,
        xgb_model     = xgb_model,
        initial_balance = 100000
    )])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 9) Learning rate schedule
    lr_schedule = get_schedule_fn(1e-4)

    # 10) Instantiate PPO with tuned hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=os.path.join(output_logs_dir, ticker.lower())
    )

    # 11) Train PPO
    print(f"🚀 Training PPO for {ticker} for 300,000 timesteps...")
    model.learn(total_timesteps=300_000)

    # 12) Save PPO model and VecNormalize stats
    model_save_path = os.path.join(output_models_dir, f"ppo_rl_xgb_{ticker.lower()}.zip")
    model.save(model_save_path)
    print(f"✅ Saved PPO model: {model_save_path}")

    env_save_path = os.path.join(output_env_dir, f"vecnormalize_{ticker.lower()}.pkl")
    env.save(env_save_path)
    print(f"✅ Saved VecNormalize stats: {env_save_path}")

    # 13) Evaluate XGB accuracy on test split
    try:
        x_test = tech_test.drop(columns=['DATE','SYMBOL','LABEL','label','Meaning','Market Mood'], errors='ignore')
        if 'LABEL' in tech_test.columns:
            y_true = tech_test['LABEL']
            y_pred = xgb_model.predict(x_test)
            acc    = accuracy_score(y_true, y_pred)
            print(f"📊 XGBoost test accuracy for {ticker}: {acc:.4f}")
    except Exception as e:
        print(f"⚠️ Accuracy eval failed for {ticker}: {e}")

print("\n🏁 All done. PPO + XGBoost agents trained, normalized envs saved.")
print("📦 Models are in the 'saved_models_with_xgb' folder.") 