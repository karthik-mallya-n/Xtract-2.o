"""
Test the current training system to see what's failing
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from core_ml import ml_core

def test_current_training():
    print("🧪 Testing Current Training System")
    print("=" * 60)
    
    # Create test dataset
    test_data = pd.DataFrame({
        'credit_score': np.random.randint(300, 850, 100),
        'income': np.random.randint(30000, 120000, 100),
        'age': np.random.randint(18, 80, 100),
        'loan_approved': np.random.choice(['Yes', 'No'], 100)
    })
    
    test_file = 'test_training_data.csv'
    test_data.to_csv(test_file, index=False)
    print(f"📄 Created test dataset: {test_file}")
    
    # Test different models
    test_models = [
        "Random Forest Classifier",
        "XGBoost Classifier", 
        "LightGBM Classifier",
        "Logistic Regression"
    ]
    
    for model_name in test_models:
        print(f"\n🧪 Testing: {model_name}")
        print("-" * 40)
        
        try:
            result = ml_core.train_advanced_model(
                model_name=model_name,
                file_path=test_file,
                target_column='loan_approved'
            )
            
            if result['success']:
                print(f"✅ Training SUCCESS!")
                print(f"   📊 Score: {result['main_score']:.4f}")
                print(f"   🎯 Threshold met: {result['threshold_met']}")
                print(f"   📁 Model folder: {result.get('model_folder', 'N/A')}")
            else:
                print(f"❌ Training FAILED: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Cleanup
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"🧹 Cleaned up: {test_file}")

if __name__ == "__main__":
    test_current_training()