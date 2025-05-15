#!/usr/bin/env python3
"""
Create Report Script for PPO Trading System
- Generates detailed markdown reports from prediction data
- Creates sector-specific analysis for banking, IT, energy, and auto sectors
- Provides short, medium, and long-term signals
- Presents buy/sell recommendations with confidence scores
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib
# Set backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

# Constants and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(BASE_DIR, "daily_predictions")
REPORTS_DIR = os.path.join(BASE_DIR, "daily_reports")
VISUALIZATIONS_DIR = os.path.join(REPORTS_DIR, "visualizations")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Sector classification for Nifty 50 stocks
SECTORS = {
    "Banking": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV"],
    "IT": ["TCS", "INFY", "WIPRO", "TECHM"],
    "Energy": ["RELIANCE", "NTPC", "POWERGRID", "BPCL", "GAIL", "IOC", "ADANIENT", "ADANIGREEN", "ADANITRANS"],
    "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJAUTO", "HEROMOTOCO", "EICHERMOT"],
    "Pharma": ["SUNPHARMA", "DIVISLAB", "DRREDDY"],
    "FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA"],
    "Manufacturing": ["LT", "ASIANPAINT", "ULTRACEMCO", "GRASIM", "SHREECEM"],
    "Other": ["BHARTIARTL", "TITAN", "ADANIPORTS", "VEDL", "COALINDIA", "UPL"]
}

def get_latest_predictions():
    """Get the most recent predictions file"""
    prediction_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith("predictions_") and f.endswith(".csv")]
    
    if not prediction_files:
        print("❌ No prediction files found in directory")
        return None
        
    # Sort by date (filenames are in format predictions_YYYYMMDD.csv)
    prediction_files.sort(reverse=True)
    latest_file = os.path.join(PREDICTIONS_DIR, prediction_files[0])
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} predictions from {latest_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading prediction file: {e}")
        return None

def get_recommendation_summary():
    """Get the most recent recommendation summary file"""
    recommendation_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith("recommendations_") and f.endswith(".csv")]
    
    if not recommendation_files:
        print("❌ No recommendation summary files found in directory")
        return None
        
    # Sort by date
    recommendation_files.sort(reverse=True)
    latest_file = os.path.join(PREDICTIONS_DIR, recommendation_files[0])
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} recommendations from {latest_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading recommendation file: {e}")
        return None

def create_overall_summary(recommendations_df):
    """Create an overall market summary based on recommendations"""
    if recommendations_df is None or recommendations_df.empty:
        return "## Market Summary\n\nNo recommendations available for market summary."
    
    total_stocks = len(recommendations_df)
    buy_stocks = len(recommendations_df[recommendations_df['Recommendation'].str.contains('Buy')])
    sell_stocks = len(recommendations_df[recommendations_df['Recommendation'].str.contains('Sell')])
    hold_stocks = total_stocks - buy_stocks - sell_stocks
    
    market_score = recommendations_df['Overall_Score'].mean()
    market_sentiment = "Bullish" if market_score > 0.2 else "Slightly Bullish" if market_score > 0 else "Neutral" if market_score == 0 else "Slightly Bearish" if market_score > -0.2 else "Bearish"
    
    # Short, medium, long term outlook
    short_term = recommendations_df['Short_Term_Score'].mean()
    medium_term = recommendations_df['Medium_Term_Score'].mean()
    long_term = recommendations_df['Long_Term_Score'].mean()
    
    short_signal = "Bullish" if short_term > 0.2 else "Slightly Bullish" if short_term > 0 else "Neutral" if short_term == 0 else "Slightly Bearish" if short_term > -0.2 else "Bearish"
    medium_signal = "Bullish" if medium_term > 0.2 else "Slightly Bullish" if medium_term > 0 else "Neutral" if medium_term == 0 else "Slightly Bearish" if medium_term > -0.2 else "Bearish"
    long_signal = "Bullish" if long_term > 0.2 else "Slightly Bullish" if long_term > 0 else "Neutral" if long_term == 0 else "Slightly Bearish" if long_term > -0.2 else "Bearish"
    
    summary = f"""## Market Summary

**Date:** {date.today().strftime('%B %d, %Y')}

### Overall Market Sentiment: {market_sentiment}

- **Market Score:** {market_score:.2f}
- **Buy Signals:** {buy_stocks} stocks ({buy_stocks/total_stocks*100:.1f}%)
- **Sell Signals:** {sell_stocks} stocks ({sell_stocks/total_stocks*100:.1f}%)
- **Hold Signals:** {hold_stocks} stocks ({hold_stocks/total_stocks*100:.1f}%)

### Time-based Analysis

- **Short-term (30 days):** {short_signal} ({short_term:.2f})
- **Medium-term (30-60 days):** {medium_signal} ({medium_term:.2f})
- **Long-term (60-90 days):** {long_signal} ({long_term:.2f})

"""
    return summary

def create_sector_analysis(recommendations_df):
    """Create sector-specific analysis"""
    if recommendations_df is None or recommendations_df.empty:
        return "## Sector Analysis\n\nNo recommendations available for sector analysis."
    
    sector_analysis = "## Sector Analysis\n\n"
    
    # Prepare data for each sector
    sector_data = {}
    
    for sector, symbols in SECTORS.items():
        sector_stocks = recommendations_df[recommendations_df['Symbol'].isin(symbols)]
        
        if sector_stocks.empty:
            continue
            
        # Calculate sector metrics
        sector_score = sector_stocks['Overall_Score'].mean()
        buy_stocks = len(sector_stocks[sector_stocks['Recommendation'].str.contains('Buy')])
        sell_stocks = len(sector_stocks[sector_stocks['Recommendation'].str.contains('Sell')])
        total_stocks = len(sector_stocks)
        
        # Get top buy and sell recommendations in this sector
        top_buys = sector_stocks.sort_values('Overall_Score', ascending=False).head(2)
        top_sells = sector_stocks.sort_values('Overall_Score', ascending=True).head(2)
        
        # Short, medium, long term outlook for sector
        short_term = sector_stocks['Short_Term_Score'].mean()
        medium_term = sector_stocks['Medium_Term_Score'].mean()
        long_term = sector_stocks['Long_Term_Score'].mean()
        
        # Store data for this sector
        sector_data[sector] = {
            'score': sector_score,
            'buy_pct': buy_stocks / total_stocks * 100 if total_stocks > 0 else 0,
            'sell_pct': sell_stocks / total_stocks * 100 if total_stocks > 0 else 0,
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'top_buys': top_buys,
            'top_sells': top_sells,
            'total_stocks': total_stocks
        }
    
    # Sort sectors by score for presentation
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Generate markdown for each sector
    for sector, data in sorted_sectors:
        sentiment = "Bullish" if data['score'] > 0.2 else "Slightly Bullish" if data['score'] > 0 else "Neutral" if data['score'] == 0 else "Slightly Bearish" if data['score'] > -0.2 else "Bearish"
        
        sector_analysis += f"### {sector} Sector: {sentiment}\n\n"
        sector_analysis += f"- **Sector Score:** {data['score']:.2f}\n"
        sector_analysis += f"- **Buy Signals:** {data['buy_pct']:.1f}% of stocks\n"
        sector_analysis += f"- **Sell Signals:** {data['sell_pct']:.1f}% of stocks\n"
        sector_analysis += f"- **Time Analysis:** Short-term ({data['short_term']:.2f}), Medium-term ({data['medium_term']:.2f}), Long-term ({data['long_term']:.2f})\n\n"
        
        # Add top buy recommendations for this sector
        if not data['top_buys'].empty:
            sector_analysis += "**Top Buy Recommendations:**\n\n"
            buy_table = []
            for _, row in data['top_buys'].iterrows():
                buy_table.append([
                    row['Symbol'], 
                    row['Recommendation'], 
                    f"{row['Overall_Score']:.2f}", 
                    row['Short_Term_Signal'], 
                    row['Medium_Term_Signal'], 
                    row['Long_Term_Signal']
                ])
            sector_analysis += tabulate(buy_table, headers=["Symbol", "Recommendation", "Score", "Short", "Medium", "Long"], tablefmt="pipe") + "\n\n"
        
        # Add top sell recommendations for this sector
        if not data['top_sells'].empty:
            sector_analysis += "**Top Sell Recommendations:**\n\n"
            sell_table = []
            for _, row in data['top_sells'].iterrows():
                sell_table.append([
                    row['Symbol'], 
                    row['Recommendation'], 
                    f"{row['Overall_Score']:.2f}", 
                    row['Short_Term_Signal'], 
                    row['Medium_Term_Signal'], 
                    row['Long_Term_Signal']
                ])
            sector_analysis += tabulate(sell_table, headers=["Symbol", "Recommendation", "Score", "Short", "Medium", "Long"], tablefmt="pipe") + "\n\n"
    
    return sector_analysis

def create_top_recommendations(recommendations_df, top_n=10):
    """Create a section with top buy and sell recommendations"""
    if recommendations_df is None or recommendations_df.empty:
        return "## Top Recommendations\n\nNo recommendations available."
    
    # Sort for top buys and sells
    top_buys = recommendations_df.sort_values('Overall_Score', ascending=False).head(top_n)
    top_sells = recommendations_df.sort_values('Overall_Score', ascending=True).head(top_n)
    
    # Create markdown section
    recommendation_md = "## Top Recommendations\n\n"
    
    # Top buys table
    recommendation_md += "### Top Buy Recommendations\n\n"
    buy_table = []
    for _, row in top_buys.iterrows():
        buy_table.append([
            row['Symbol'], 
            row['Recommendation'], 
            f"{row['Overall_Score']:.2f}", 
            f"{row['Buy_Count']} days", 
            row['Short_Term_Signal'], 
            row['Medium_Term_Signal'], 
            row['Long_Term_Signal']
        ])
    recommendation_md += tabulate(buy_table, headers=["Symbol", "Signal", "Score", "Buy Days", "Short", "Medium", "Long"], tablefmt="pipe") + "\n\n"
    
    # Top sells table
    recommendation_md += "### Top Sell Recommendations\n\n"
    sell_table = []
    for _, row in top_sells.iterrows():
        sell_table.append([
            row['Symbol'], 
            row['Recommendation'], 
            f"{row['Overall_Score']:.2f}", 
            f"{row['Sell_Count']} days", 
            row['Short_Term_Signal'], 
            row['Medium_Term_Signal'], 
            row['Long_Term_Signal']
        ])
    recommendation_md += tabulate(sell_table, headers=["Symbol", "Signal", "Score", "Sell Days", "Short", "Medium", "Long"], tablefmt="pipe") + "\n\n"
    
    return recommendation_md

def generate_plots(recommendations_df):
    """Generate plots for the report and return their filenames"""
    if recommendations_df is None or recommendations_df.empty:
        print("❌ No recommendations available for plot generation")
        return []
    
    plot_files = []
    
    # Set style
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    try:
        print("📊 Generating score distribution plot...")
        # 1. Overall score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(recommendations_df['Overall_Score'], kde=True, bins=20)
        plt.title('Distribution of Stock Recommendation Scores')
        plt.xlabel('Recommendation Score (-1 to +1)')
        plt.ylabel('Number of Stocks')
        plt.axvline(x=0, color='black', linestyle='--')
        score_hist_file = os.path.join(VISUALIZATIONS_DIR, 'score_distribution.png')
        plt.savefig(score_hist_file, dpi=100, bbox_inches='tight')
        plt.close()
        plot_files.append(score_hist_file)
        print(f"✅ Created score distribution plot at {score_hist_file}")
        
        print("📊 Generating sector comparison plot...")
        # 2. Sector comparison
        sector_scores = []
        for sector, symbols in SECTORS.items():
            sector_stocks = recommendations_df[recommendations_df['Symbol'].isin(symbols)]
            if not sector_stocks.empty:
                sector_scores.append({
                    'Sector': sector,
                    'Score': sector_stocks['Overall_Score'].mean(),
                    'StockCount': len(sector_stocks)
                })
        
        if sector_scores:
            sector_df = pd.DataFrame(sector_scores)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Sector', y='Score', data=sector_df)
            plt.title('Average Recommendation Score by Sector')
            plt.xlabel('Sector')
            plt.ylabel('Average Score (-1 to +1)')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xticks(rotation=45)
            for i, row in enumerate(sector_df.itertuples()):
                ax.text(i, row.Score + (0.1 if row.Score >= 0 else -0.1), 
                        f"{row.StockCount} stocks", ha='center')
            sector_comparison_file = os.path.join(VISUALIZATIONS_DIR, 'sector_comparison.png')
            plt.tight_layout()
            plt.savefig(sector_comparison_file, dpi=100, bbox_inches='tight')
            plt.close()
            plot_files.append(sector_comparison_file)
            print(f"✅ Created sector comparison plot at {sector_comparison_file}")
            
        print("📊 Generating time horizon comparison plot...")
        # 3. Time horizon comparison (short, medium, long term)
        plt.figure(figsize=(10, 6))
        time_data = pd.DataFrame({
            'Horizon': ['Short-term', 'Medium-term', 'Long-term'],
            'Score': [
                recommendations_df['Short_Term_Score'].mean(),
                recommendations_df['Medium_Term_Score'].mean(),
                recommendations_df['Long_Term_Score'].mean()
            ]
        })
        sns.barplot(x='Horizon', y='Score', data=time_data)
        plt.title('Average Score by Time Horizon')
        plt.xlabel('Time Horizon')
        plt.ylabel('Average Score (-1 to +1)')
        plt.axhline(y=0, color='black', linestyle='--')
        time_comparison_file = os.path.join(VISUALIZATIONS_DIR, 'time_comparison.png')
        plt.savefig(time_comparison_file, dpi=100, bbox_inches='tight')
        plt.close()
        plot_files.append(time_comparison_file)
        print(f"✅ Created time horizon plot at {time_comparison_file}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return plot_files

def create_markdown_report(recommendations_df, predictions_df):
    """Create a complete markdown report"""
    if recommendations_df is None or recommendations_df.empty:
        return "# Stock Market Analysis Report\n\nNo recommendations available for report generation."
    
    # Generate header
    report = f"""# Stock Market Analysis Report

**Date:** {date.today().strftime('%B %d, %Y')}
**Generated by:** PPO Reinforcement Learning System
**Analysis Period:** Next 90 days forecast

"""
    
    # Add sections
    report += create_overall_summary(recommendations_df)
    report += "\n\n"
    report += create_top_recommendations(recommendations_df)
    report += "\n\n"
    report += create_sector_analysis(recommendations_df)
    
    # Generate and add plots
    plot_files = generate_plots(recommendations_df)
    
    # Add plots to the report
    if plot_files:
        report += "\n\n## Market Visualizations\n\n"
        report += "### Score Distribution\n\n"
        report += "![Score Distribution](visualizations/score_distribution.png)\n\n"
        report += "### Sector Comparison\n\n"
        report += "![Sector Comparison](visualizations/sector_comparison.png)\n\n"
        report += "### Time Horizon Analysis\n\n"
        report += "![Time Horizon Analysis](visualizations/time_comparison.png)\n\n"
    
    # Add footer
    report += "\n\n---\n\n"
    report += "*This report is generated by an automated PPO reinforcement learning system. "
    report += "All recommendations should be considered as suggestions and not financial advice. "
    report += "Always conduct your own research before making investment decisions.*"
    
    return report

def generate_report():
    """Main function to generate the report"""
    print(f"{'='*80}\n📊 GENERATING MARKET REPORT\n{'='*80}")
    
    # Get the latest data
    predictions_df = get_latest_predictions()
    recommendations_df = get_recommendation_summary()
    
    # If no recommendation summary file exists, create one from the predictions
    if recommendations_df is None and predictions_df is not None:
        print("⚠️ No recommendation summary file found. Creating one from predictions...")
        try:
            # Ensure we have the necessary columns in the predictions DataFrame
            if 'Day' not in predictions_df.columns:
                # Create a Day column based on Date
                try:
                    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
                    min_date = predictions_df['Date'].min()
                    predictions_df['Day'] = (predictions_df['Date'] - min_date).dt.days
                except Exception as e:
                    print(f"⚠️ Could not create Day from Date column: {e}")
                    # Create a sequential Day column if date conversion fails
                    predictions_df['Day'] = predictions_df.groupby('Symbol').cumcount()
            
            # Create a basic recommendation file from the predictions
            symbols = predictions_df['Symbol'].unique()
            recommendations = []
            
            for symbol in symbols:
                symbol_preds = predictions_df[predictions_df['Symbol'] == symbol]
                
                # Calculate buy/sell counts and scores
                buy_count = len(symbol_preds[symbol_preds['Predicted_Action'] == 'BUY'])
                sell_count = len(symbol_preds[symbol_preds['Predicted_Action'] == 'SELL'])
                hold_count = len(symbol_preds[symbol_preds['Predicted_Action'] == 'HOLD'])
                
                # Ensure day values are numeric
                symbol_preds['Day'] = pd.to_numeric(symbol_preds['Day'], errors='coerce').fillna(0)
                
                # Calculate time-based scores - using day ranges
                day_max = symbol_preds['Day'].max()
                day_third = day_max / 3
                
                short_term_preds = symbol_preds[symbol_preds['Day'] <= day_third]
                medium_term_preds = symbol_preds[(symbol_preds['Day'] > day_third) & (symbol_preds['Day'] <= 2*day_third)]
                long_term_preds = symbol_preds[symbol_preds['Day'] > 2*day_third]
                
                # Score calculation: (buy_count - sell_count) / total_count
                total_count = len(symbol_preds)
                overall_score = (buy_count - sell_count) / total_count if total_count > 0 else 0
                
                # Time-based scores
                short_term_score = 0
                medium_term_score = 0
                long_term_score = 0
                
                if len(short_term_preds) > 0:
                    short_buy = len(short_term_preds[short_term_preds['Predicted_Action'] == 'BUY'])
                    short_sell = len(short_term_preds[short_term_preds['Predicted_Action'] == 'SELL'])
                    short_term_score = (short_buy - short_sell) / len(short_term_preds)
                
                if len(medium_term_preds) > 0:
                    med_buy = len(medium_term_preds[medium_term_preds['Predicted_Action'] == 'BUY'])
                    med_sell = len(medium_term_preds[medium_term_preds['Predicted_Action'] == 'SELL'])
                    medium_term_score = (med_buy - med_sell) / len(medium_term_preds)
                
                if len(long_term_preds) > 0:
                    long_buy = len(long_term_preds[long_term_preds['Predicted_Action'] == 'BUY'])
                    long_sell = len(long_term_preds[long_term_preds['Predicted_Action'] == 'SELL'])
                    long_term_score = (long_buy - long_sell) / len(long_term_preds)
                
                # Determine recommendations based on scores
                recommendation = "Strong Buy" if overall_score > 0.6 else \
                                 "Buy" if overall_score > 0.2 else \
                                 "Neutral" if overall_score >= -0.2 else \
                                 "Sell" if overall_score >= -0.6 else "Strong Sell"
                
                short_signal = "Buy" if short_term_score > 0.2 else \
                              "Hold" if short_term_score >= -0.2 else "Sell"
                
                medium_signal = "Buy" if medium_term_score > 0.2 else \
                               "Hold" if medium_term_score >= -0.2 else "Sell"
                
                long_signal = "Buy" if long_term_score > 0.2 else \
                             "Hold" if long_term_score >= -0.2 else "Sell"
                
                # Create recommendation entry
                recommendations.append({
                    'Symbol': symbol,
                    'Recommendation': recommendation,
                    'Overall_Score': overall_score,
                    'Buy_Count': buy_count,
                    'Sell_Count': sell_count,
                    'Hold_Count': hold_count,
                    'Short_Term_Score': short_term_score,
                    'Medium_Term_Score': medium_term_score,
                    'Long_Term_Score': long_term_score,
                    'Short_Term_Signal': short_signal,
                    'Medium_Term_Signal': medium_signal,
                    'Long_Term_Signal': long_signal
                })
            
            # Create the DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            
            # Save the recommendations file
            today = date.today().strftime('%Y%m%d')
            recommendations_file = os.path.join(PREDICTIONS_DIR, f"recommendations_{today}.csv")
            recommendations_df.to_csv(recommendations_file, index=False)
            print(f"✅ Created recommendations file: {recommendations_file}")
            
        except Exception as e:
            print(f"❌ Error creating recommendations from predictions: {e}")
            import traceback
            traceback.print_exc()
    
    if predictions_df is None or recommendations_df is None:
        print("❌ Cannot generate report without predictions or recommendations")
        return False
    
    # Create the markdown report
    report_md = create_markdown_report(recommendations_df, predictions_df)
    
    # Save the report
    today = date.today().strftime('%Y%m%d')
    report_file = os.path.join(REPORTS_DIR, f"market_report_{today}.md")
    
    with open(report_file, 'w') as f:
        f.write(report_md)
    
    print(f"✅ Report generated successfully: {report_file}")
    
    # Also save a CSV report with key metrics
    try:
        # Create a summary table
        summary_data = []
        for sector, symbols in SECTORS.items():
            sector_stocks = recommendations_df[recommendations_df['Symbol'].isin(symbols)]
            if not sector_stocks.empty:
                sector_score = sector_stocks['Overall_Score'].mean()
                sector_short = sector_stocks['Short_Term_Score'].mean()
                sector_medium = sector_stocks['Medium_Term_Score'].mean()
                sector_long = sector_stocks['Long_Term_Score'].mean()
                
                summary_data.append({
                    'Date': date.today(),
                    'Sector': sector,
                    'Overall_Score': sector_score,
                    'Short_Term_Score': sector_short,
                    'Medium_Term_Score': sector_medium,
                    'Long_Term_Score': sector_long,
                    'Stock_Count': len(sector_stocks)
                })
        
        # Add overall market summary
        summary_data.append({
            'Date': date.today(),
            'Sector': 'MARKET',
            'Overall_Score': recommendations_df['Overall_Score'].mean(),
            'Short_Term_Score': recommendations_df['Short_Term_Score'].mean(),
            'Medium_Term_Score': recommendations_df['Medium_Term_Score'].mean(),
            'Long_Term_Score': recommendations_df['Long_Term_Score'].mean(),
            'Stock_Count': len(recommendations_df)
        })
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(REPORTS_DIR, f"market_summary_{today}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary CSV saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error saving summary CSV: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    generate_report() 