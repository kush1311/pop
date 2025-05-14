#!/usr/bin/env python3
"""
Simple test script to generate predictions for all stocks
without writing to the standard output file
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import function from generate_predictions but don't run it immediately
from generate_predictions import fetch_live_features

def create_test_predictions():
    """Generate test predictions for all stocks"""
    print(f"📈 Generating test predictions for all stocks")
    
    # Create output directory
    output_dir = "test_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    today = date.today().strftime('%Y%m%d')
    output_file = os.path.join(output_dir, f"predictions_{today}.csv")
    
    # Fetch latest data
    print("📊 Fetching latest market data...")
    real_data = fetch_live_features()
    
    if real_data is None or real_data.empty:
        print("❌ Failed to fetch market data")
        return None
    
    # Get all symbols from real_data
    all_symbols = real_data['Symbol'].unique() if real_data is not None and not real_data.empty else ["SAMPLE1", "SAMPLE2"]
    print(f"📈 Will generate predictions for {len(all_symbols)} stocks")
    
    # Create demonstration predictions
    demo_predictions = []
    forecast_days = 90
    
    for symbol in all_symbols:
        # For each stock, create predictions for the next forecast_days days
        for day in range(forecast_days):
            action = np.random.choice([0, 1, 2])  # Random action
            position_name = "BUY" if action == 2 else "SELL" if action == 0 else "HOLD"
            
            # Get the last known price for this symbol if available
            last_price = None
            if real_data is not None and not real_data.empty:
                symbol_data = real_data[real_data['Symbol'] == symbol]
                if not symbol_data.empty and 'Close' in symbol_data.columns:
                    last_price = symbol_data['Close'].iloc[-1]
            
            # Use last price if available, otherwise generate random price
            if last_price:
                # Add some random variation to simulate future price
                price = last_price * (1 + np.random.normal(0, 0.02))  # 2% daily volatility
            else:
                price = 1000 + np.random.normal(0, 10)  # Random price
            
            demo_predictions.append({
                'Date': date.today() + timedelta(days=day),
                'Symbol': symbol,
                'Predicted_Action': position_name,
                'Position': 1 if action == 2 else -1 if action == 0 else 0,
                'Action_Code': int(action),
                'Close': price,
                'Forecasted_Price': round(price, 2)
            })
    
    # Create demo dataframe
    combined_predictions = pd.DataFrame(demo_predictions)
    
    # Save to CSV
    combined_predictions.to_csv(output_file, index=False)
    print(f"✅ Created predictions file with {len(combined_predictions)} rows at {output_file}")
    
    # Create summary output
    summary_file = os.path.join(output_dir, f"summary_{today}.csv")
    
    # Calculate detailed metrics
    detailed_recommendations = []
    for symbol in combined_predictions['Symbol'].unique():
        symbol_data = combined_predictions[combined_predictions['Symbol'] == symbol]
        
        # Count actions
        buy_count = sum(symbol_data['Predicted_Action'] == 'BUY')
        sell_count = sum(symbol_data['Predicted_Action'] == 'SELL')
        hold_count = sum(symbol_data['Predicted_Action'] == 'HOLD')
        
        # Calculate score
        score = symbol_data['Position'].sum() / len(symbol_data)
        
        # Determine recommendation
        recommendation = "Strong Buy" if score > 0.5 else "Buy" if score > 0 else "Hold" if score == 0 else "Sell" if score > -0.5 else "Strong Sell"
        
        # Calculate short, medium, long term signals
        if len(symbol_data) >= 90:
            short_term = symbol_data.iloc[:30]['Position'].sum() / 30
            medium_term = symbol_data.iloc[30:60]['Position'].sum() / 30
            long_term = symbol_data.iloc[60:90]['Position'].sum() / 30
        else:
            # Handle cases with less than 90 days of predictions
            third = len(symbol_data) // 3
            short_term = symbol_data.iloc[:third]['Position'].sum() / third if third > 0 else 0
            medium_term = symbol_data.iloc[third:2*third]['Position'].sum() / third if third > 0 else 0
            long_term = symbol_data.iloc[2*third:]['Position'].sum() / (len(symbol_data) - 2*third) if (len(symbol_data) - 2*third) > 0 else 0
        
        # Store all data
        detailed_recommendations.append({
            'Symbol': symbol,
            'Buy_Count': buy_count,
            'Sell_Count': sell_count,
            'Hold_Count': hold_count,
            'Overall_Score': score,
            'Short_Term_Score': short_term,  # First 30 days
            'Medium_Term_Score': medium_term,  # Next 30 days
            'Long_Term_Score': long_term,  # Last 30 days
            'Recommendation': recommendation,
            'Short_Term_Signal': "Buy" if short_term > 0 else "Hold" if short_term == 0 else "Sell",
            'Medium_Term_Signal': "Buy" if medium_term > 0 else "Hold" if medium_term == 0 else "Sell",
            'Long_Term_Signal': "Buy" if long_term > 0 else "Hold" if long_term == 0 else "Sell"
        })
    
    # Convert to DataFrame and save
    pd.DataFrame(detailed_recommendations).to_csv(summary_file, index=False)
    print(f"✅ Saved detailed recommendations to {summary_file}")
    
    return combined_predictions, detailed_recommendations

if __name__ == "__main__":
    predictions, recommendations = create_test_predictions()
    
    # Display top 5 BUY recommendations
    if recommendations:
        top_buys = sorted(recommendations, key=lambda x: x['Overall_Score'], reverse=True)[:5]
        print("\n💰 TOP 5 BUY RECOMMENDATIONS:")
        for rec in top_buys:
            print(f"{rec['Symbol']}: {rec['Recommendation']} (Score: {rec['Overall_Score']:.2f}, Buy: {rec['Buy_Count']}, Sell: {rec['Sell_Count']}, Hold: {rec['Hold_Count']})")
        
        # Display top 5 SELL recommendations
        top_sells = sorted(recommendations, key=lambda x: x['Overall_Score'])[:5]
        print("\n📉 TOP 5 SELL RECOMMENDATIONS:")
        for rec in top_sells:
            print(f"{rec['Symbol']}: {rec['Recommendation']} (Score: {rec['Overall_Score']:.2f}, Buy: {rec['Buy_Count']}, Sell: {rec['Sell_Count']}, Hold: {rec['Hold_Count']})") 