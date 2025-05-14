#!/usr/bin/env python3
"""
Daily update script for PPO RL trading model
This script:
1. Updates the Nifty50 data using data.py
2. Runs the PPO retraining using train_ppo_realtime_multi.py
3. Generates predictions for next 3 months using generate_predictions.py
4. Creates detailed market reports with the create_report.py module
"""

import os
import sys
import time
import traceback
from datetime import datetime

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "saved_models_with_xgb",
        "saved_envs",
        "daily_reports",
        "daily_reports/visualizations",
        "daily_predictions"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

def run_update():
    """Main function to run the daily update process"""
    start_time = time.time()
    print(f"\n{'='*80}\n📊 DAILY PPO UPDATE - {datetime.now()}\n{'='*80}")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Step 1: Update the excel file with latest data
        print("\n🔄 STEP 1: Updating market data from data.py")
        import data
        print("✅ Data update complete")
        
        # Step 2: Run PPO retraining
        print("\n🔄 STEP 2: Running PPO retraining")
        from train_ppo_realtime_multi import run_retraining
        run_retraining()
        print("✅ Model retraining complete")
        
        # Step 3: Generate predictions
        print("\n🔄 STEP 3: Generating predictions for next 3 months")
        predictions = None
        try:
            from generate_predictions import run_all_predictions
            predictions = run_all_predictions()
            if predictions is not None:
                print(f"✅ Successfully generated predictions for {predictions['Symbol'].nunique()} stocks")
                
                # Save predictions summary
                try:
                    summary_file = os.path.join("daily_predictions", f"summary_{datetime.now().strftime('%Y%m%d')}.csv")
                    predictions.groupby('Symbol').agg({
                        'Predicted_Action': 'value_counts',
                        'Confidence': ['mean', 'std']
                    }).to_csv(summary_file)
                    print(f"✅ Saved predictions summary to {summary_file}")
                except Exception as e:
                    print(f"⚠️ Failed to save predictions summary: {e}")
            else:
                print("⚠️ No predictions were generated")
        except Exception as e:
            print(f"\n❌ ERROR during predictions: {str(e)}")
            traceback.print_exc()
            print("⚠️ Continuing with the rest of the process despite prediction error")
        
        # Step 4: Create market report
        print("\n🔄 STEP 4: Creating detailed market report")
        try:
            # First check if we have the required files
            if not os.path.exists("daily_predictions"):
                print("⚠️ daily_predictions directory not found")
                return False
                
            prediction_files = os.listdir("daily_predictions")
            if not prediction_files:
                print("⚠️ No prediction files found in daily_predictions/")
                return False
                
            print(f"📊 Found prediction files: {prediction_files}")
            
            from create_report import generate_report
            report_success = generate_report()
            
            if report_success:
                print("✅ Market report generation complete")
                # List generated files
                report_files = []
                for root, dirs, files in os.walk("daily_reports"):
                    for file in files:
                        report_files.append(os.path.join(root, file))
                print("\n📄 Generated report files:")
                for file in report_files:
                    print(f"  - {file}")
            else:
                print("⚠️ Failed to generate market report")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR during report generation: {str(e)}")
            traceback.print_exc()
            print("⚠️ Continuing despite report generation error")
            return False
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ DAILY UPDATE COMPLETED in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\n⚠️ Daily update failed!")
        return False

if __name__ == "__main__":
    success = run_update()
    sys.exit(0 if success else 1) 