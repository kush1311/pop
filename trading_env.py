import gym
import numpy as np
import pandas as pd
 
class DynamicTradingEnv(gym.Env):
    """
    A trading environment for RL agents that includes:
    - Daily technical features
    - PCR, sentiment, and quarterly financial data
    - Optional XGBoost model output as additional state info
    """
    metadata = {"render_modes": ["human"]}
 
    def __init__(self, df_daily, df_pcr, df_sentiment, df_quarterly, xgb_model=None, initial_balance=100000):
        super().__init__()
 
        # Core inputs
        self.df_daily     = df_daily.copy()
        self.df_pcr       = df_pcr.copy()
        self.df_sentiment = df_sentiment.copy()
        self.df_quarterly = df_quarterly.copy()
        self.xgb_model    = xgb_model
 
        self.initial_balance = initial_balance
        self.current_step = 0
 
        # Ensure dates are proper datetime objects
        for df in [self.df_daily, self.df_pcr, self.df_sentiment, self.df_quarterly]:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.dropna(subset=['DATE'], inplace=True)
 
        # Sort and clean df_daily
        self.df_daily = self.df_daily.sort_values("DATE").reset_index(drop=True)
        self.dates = self.df_daily['DATE'].reset_index(drop=True)
        self.df_daily['CLOSE'] = self.df_daily['CLOSE'].ffill().bfill()
        self.price_series = self.df_daily['CLOSE'].values
 
        drop_cols = ['DATE', 'SYMBOL', 'LABEL', 'label', 'Meaning', 'Market Mood']
        self.df_daily = self.df_daily.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
 
        # Columns to extract from auxiliary data
        self.pcr_cols = [c for c in self.df_pcr.columns if c not in ('DATE', 'SYMBOL') and pd.api.types.is_numeric_dtype(self.df_pcr[c])]
        self.sentiment_cols = [c for c in self.df_sentiment.columns if c not in ('DATE', 'SYMBOL') and pd.api.types.is_numeric_dtype(self.df_sentiment[c])]
        self.quarterly_cols = [c for c in self.df_quarterly.columns if c not in ('DATE', 'SYMBOL') and pd.api.types.is_numeric_dtype(self.df_quarterly[c])]
 
        # Total state vector length
        self.state_size = (
            self.df_daily.shape[1] +
            len(self.pcr_cols) +
            len(self.sentiment_cols) +
            len(self.quarterly_cols) +
            (3 if self.xgb_model is not None else 0) +  # [buy, hold, sell] proba
            1  # position
        )
 
        # Gym format
        self.action_space = gym.spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
 
        self.reset()
 
    def _get_state(self):
        ref_date = self.dates[self.current_step]
        base = self.df_daily.iloc[self.current_step].to_numpy(dtype=np.float32).ravel()
 
        pcr_row = self.df_pcr[self.df_pcr['DATE'] == ref_date]
        pcr_data = pcr_row[self.pcr_cols].to_numpy(dtype=np.float32).ravel() if not pcr_row.empty else np.zeros(len(self.pcr_cols), dtype=np.float32)
 
        sent_row = self.df_sentiment[self.df_sentiment['DATE'] == ref_date]
        sent_data = sent_row[self.sentiment_cols].to_numpy(dtype=np.float32).ravel() if not sent_row.empty else np.zeros(len(self.sentiment_cols), dtype=np.float32)
 
        q_row = self.df_quarterly[self.df_quarterly['DATE'] <= ref_date].sort_values("DATE").tail(1)
        q_data = q_row[self.quarterly_cols].to_numpy(dtype=np.float32).ravel() if not q_row.empty else np.zeros(len(self.quarterly_cols), dtype=np.float32)
 
        if self.xgb_model is not None:
            try:
                xgb_input = self.df_daily.iloc[[self.current_step]]
                xgb_pred = self.xgb_model.predict_proba(xgb_input)[0].astype(np.float32)
            except Exception:
                xgb_pred = np.zeros(3, dtype=np.float32)
        else:
            xgb_pred = np.zeros(3, dtype=np.float32)
 
        state = np.concatenate([base, pcr_data, sent_data, q_data, xgb_pred, [self.position]])
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
 
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        return self._get_state()
 
    def step(self, action):
        price = self.price_series[self.current_step]
        reward = 0
 
        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = price
            elif self.position == -1:
                reward = self.entry_price - price
                self.total_profit += reward
                self.position = 0
 
        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = price
            elif self.position == 1:
                reward = price - self.entry_price
                self.total_profit += reward
                self.position = 0
 
        self.current_step += 1
        done = self.current_step >= (len(self.df_daily) - 1)
        next_state = self._get_state()
        info = {
            'step': self.current_step,
            'total_profit': self.total_profit,
            'position': self.position
        }
        return next_state, reward, done, info
 
    def render(self, mode='human'):
        print(f"Step {self.current_step} | Pos {self.position} | P&L {self.total_profit:.2f}")
 
 
