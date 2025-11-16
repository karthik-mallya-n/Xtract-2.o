#!/usr/bin/env python3
"""
Test CatBoost training to ensure single execution and clean output
"""

import sys
import os
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_ml import ml_core

def test_catboost_training():
    """Test CatBoost training with clean output"""
    
    print("🧪 TESTING CATBOOST TRAINING")
    print("=" * 50)
    
    # Test with Iris dataset
    file_path = "uploads/b0560d95-7006-4035-9ac9-a547229a0071.csv"
    model_name = "catboost-classifier"
    target_column = "Species"
    
    print(f"📄 Dataset: {file_path}")
    print(f"🤖 Model: {model_name}")  
    print(f"🎯 Target: {target_column}")
    
    # Record start time
    start_time = time.time()
    
    # Train the model
    print(f"\n🚀 Starting training...")
    result = ml_core.train_advanced_model(
        model_name=model_name,
        file_path=file_path,
        target_column=target_column
    )
    
    # Record end time
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    
    # Verify results
    if result['success']:
        accuracy = result['main_score']
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"📁 Model saved: {result['model_folder']}")
        
        # Check if high accuracy achieved
        if accuracy >= 0.90:
            print("🏆 SUCCESS: Achieved 90%+ accuracy!")
        else:
            print(f"⚠️ Accuracy below 90%: {accuracy*100:.1f}%")
            
        return True
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = test_catboost_training()
    if success:
        print(f"\n✅ CatBoost test completed successfully!")
    else:
        print(f"\n❌ CatBoost test failed!")
        sys.exit(1)