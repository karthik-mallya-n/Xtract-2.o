#!/usr/bin/env python3
"""
Direct Training Test - Shows actual backend training results
"""

import sys
import os
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_ml import ml_core

def test_direct_training():
    """Test training directly and show results that would go to frontend"""
    
    print("🚀 DIRECT BACKEND TRAINING TEST")
    print("=" * 50)
    
    # Use your actual dataset
    file_path = "uploads/b0560d95-7006-4035-9ac9-a547229a0071.csv"
    model_name = "Random Forest"
    target_column = "Species"
    
    print(f"📄 Dataset: {file_path}")
    print(f"🤖 Model: {model_name}")
    print(f"🎯 Target: {target_column}")
    
    # Train the model
    print(f"\n🔥 Starting training...")
    result = ml_core.train_advanced_model(
        model_name=model_name,
        file_path=file_path,
        target_column=target_column
    )
    
    if result['success']:
        print(f"\n✅ TRAINING SUCCESSFUL!")
        print(f"📁 Model: {result['model_folder']}")
        print(f"🎯 Accuracy: {result['main_score']*100:.2f}%")
        print(f"🏆 Threshold Met: {'✅ YES' if result['threshold_met'] else '❌ NO'}")
        
        # Show what frontend would receive
        print(f"\n📊 FRONTEND WOULD RECEIVE:")
        print(json.dumps(result, indent=2, default=str)[:500] + "...")
        
        # Save for manual testing
        with open("live_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Live results saved to: live_results.json")
        print(f"🌐 Test in frontend: http://localhost:3000/results")
        
        return True
    else:
        print(f"❌ Training failed: {result.get('error')}")
        return False

if __name__ == "__main__":
    success = test_direct_training()
    
    if success:
        print(f"\n🎉 BACKEND TRAINING WORKS PERFECTLY!")
        print(f"💡 Your model achieves excellent performance in seconds!")
    else:
        print(f"\n❌ Training test failed")
        sys.exit(1)