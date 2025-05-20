# Stock Prediction Model with Reinforcement Learning

This project implements a reinforcement learning-based stock prediction model for NIFTY 50 stocks, focusing on RELIANCE as the initial company. The model uses technical indicators, price data, sentiment analysis, and options data to predict whether to BUY, SELL, or HOLD a stock.

## Features

- Reinforcement learning (PPO algorithm) for trading decisions
- Technical indicators (RSI, MACD, Bollinger Bands, Golden/Death Cross, etc.)
- Sentiment analysis from news data
- Options data analysis (Put-Call Ratio)
- Quarterly financial data integration
- Interactive visualizations
- Multi-horizon predictions (15, 30, 45, 60, 90 days)
- Stop-loss recommendations based on volatility and support/resistance levels
- Confidence-backed predictions with detailed rationale
- Realistic trading simulation with slippage and transaction costs
- Automated stop-loss and take-profit mechanisms

## Project Structure

- `stock_env.py`: Custom Gymnasium environment for stock trading simulation
- `training.py`: Main script for data loading, feature engineering, and model training
- `run_model.py`: Script to run a trained model and get predictions
- `visualizations/`: Directory containing generated visualizations
- `models/`: Directory for saved models
- `logs/`: Training logs for TensorBoard
- `prediction_reports/`: Directory for saved prediction reports

## Data Files

The model uses the following data files:
- `tech+rsi+price based features.xlsx`: Technical indicators and price data
- `updated_Nifty50_PCR_Features_with_label.xlsx`: Put-Call Ratio data
- `Nifty50_Quarterly_Features.xlsx`: Quarterly financial data
- `Labeled_News_Sentiment_Data.csv`: News sentiment data

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
stable-baselines3
gymnasium
```

## Usage

### Training a Model

To train a new model:

```bash
python training.py
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train a reinforcement learning model
4. Evaluate the model
5. Make predictions
6. Save visualizations

### Running a Trained Model

To run a trained model and get predictions:

```bash
python run_model.py --company RELIANCE --visualize --save_report
```

Options:
- `--company`: Company to predict (default: RELIANCE)
- `--model_path`: Path to a specific model file (default: use latest model)
- `--visualize`: Show visualizations
- `--save_report`: Save prediction report to a text file

## Visualizations

The model generates several visualizations:
- Stock price with buy/sell markers and stop-loss/take-profit levels
- Portfolio value over time with cash and asset breakdown
- Performance metrics including drawdown and rolling returns
- Action distribution and trading statistics
- Prediction summary with target prices and expected returns
- Technical indicators with RSI and MACD

## Model Logic

The reinforcement learning agent:
1. Observes market data and technical indicators
2. Takes actions (BUY, SELL, HOLD)
3. Receives rewards based on portfolio performance
4. Implements automatic stop-loss and take-profit mechanisms
5. Learns optimal trading strategy over time

The final prediction includes:
- Recommended action (BUY, SELL, HOLD)
- Current price
- Target prices for different time horizons
- Expected returns for each horizon
- Stop-loss recommendation
- Confidence level
- Detailed rationale for the recommendation
- Volatility information

## Extending to Other Stocks

To use the model for other NIFTY 50 stocks, modify the `COMPANY` constant in `training.py` or use the `--company` flag with `run_model.py`.

## Future Improvements

- Hyperparameter optimization
- Multi-stock portfolio management
- Market regime detection
- Integration with live trading APIs
- Adaptive stop-loss strategies
- Deep reinforcement learning models (A2C, TD3)
- Ensemble methods combining multiple models
- Incorporating macroeconomic indicators
- Sector-specific models
