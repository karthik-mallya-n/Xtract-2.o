#!/usr/bin/env python3
"""
Quick test to verify the model implementation works
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_ml import ml_core
import pandas as pd

# Create a simple test dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'feature3': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Save to CSV
test_file = 'quick_test_dataset.csv'
df.to_csv(test_file, index=False)

print("🧪 Quick Model Implementation Test")
print("=" * 50)

# Test a few key models
test_models = [
    "Random Forest",
    "XGBoost", 
    "Support Vector Machine",
    "Logistic Regression"
]

for model_name in test_models:
    print(f"\n🔧 Testing: {model_name}")
    try:
        result = ml_core.train_specific_model(
            file_path=test_file,
            model_name=model_name,
            user_data={'is_labeled': True, 'data_type': 'mixed'},
            target_column='target'
        )
        
        if result.get('success'):
            print(f"✅ {model_name} - SUCCESS")
            print(f"   Score: {result.get('performance', {}).get('accuracy', 'N/A')}")
        else:
            print(f"❌ {model_name} - FAILED")
            print(f"   Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"💥 {model_name} - EXCEPTION")
        print(f"   Error: {str(e)}")

# Clean up
if os.path.exists(test_file):
    os.remove(test_file)

print(f"\n✅ Quick test completed!")