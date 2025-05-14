#!/usr/bin/env python3
"""
Daily update script for PPO RL trading model
This script:
1. Updates the Nifty50 data using data.py
2. Runs the PPO retraining using train_ppo_realtime_multi.py
3. Generates predictions for next 3 months using generate_predictions.py
"""

import os
import sys
import time
import traceback
from datetime import datetime

def run_update():
    """Main function to run the daily update process"""
    start_time = time.time()
    print(f"\n{'='*80}\n📊 DAILY PPO UPDATE - {datetime.now()}\n{'='*80}")
    
    try:
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
        try:
            from generate_predictions import run_all_predictions
            predictions = run_all_predictions()
            if predictions is not None:
                print(f"✅ Successfully generated predictions for {predictions['Symbol'].nunique()} stocks")
            else:
                print("⚠️ No predictions were generated")
        except Exception as e:
            print(f"\n❌ ERROR during predictions: {str(e)}")
            traceback.print_exc()
            print("⚠️ Continuing with the rest of the process despite prediction error")
        
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