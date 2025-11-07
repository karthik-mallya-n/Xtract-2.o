#!/usr/bin/env python3
"""
Test the classification scenario (Labeled + Categorical)
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core_ml import ml_core
    print("✅ Successfully imported ml_core")
except Exception as e:
    print(f"❌ Failed to import ml_core: {str(e)}")
    exit(1)

# Create a classification test dataset
print("\n📊 Creating classification test dataset...")
test_data = {
    'temperature': [20, 25, 30, 35, 15, 28, 32, 18, 26, 29] * 10,
    'humidity': [60, 70, 80, 90, 50, 65, 75, 55, 68, 72] * 10,
    'wind_speed': [10, 15, 20, 25, 5, 12, 18, 8, 14, 16] * 10,
    'weather': ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Rainy'] * 10  # Categorical target
}

df = pd.DataFrame(test_data)
test_file_path = 'test_classification_dataset.csv'
df.to_csv(test_file_path, index=False)
print(f"✅ Classification dataset created: {df.shape}")

# Test user answers for classification scenario (Labeled + Categorical)
user_answers = {
    'data_type': 'categorical',  # Target is categorical (weather)
    'is_labeled': 'labeled',     # We have a target variable
    'problem_type': 'classification',
    'data_size': 'medium',
    'accuracy_priority': 'high'
}

print(f"\n🎯 User Answers: {user_answers}")

try:
    # Analyze the dataset
    print("\n📊 Analyzing dataset...")
    dataset_analysis = ml_core.analyze_dataset(test_file_path)
    
    # Test the recommendation system for classification
    print("\n🤖 Testing classification recommendations...")
    llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
    
    if llm_response.get('success'):
        print("✅ Classification LLM request successful!")
        recs = llm_response.get('recommendations', {})
        
        # Display scenario and semantic analysis
        if 'scenario_detected' in recs:
            scenario = recs['scenario_detected']
            print(f"\n🎯 SCENARIO: {scenario.get('type', 'Unknown')}")
            print(f"📝 TASK: {scenario.get('task', 'Unknown')}")
        
        # Display ranked classification models
        if 'recommended_models' in recs:
            models = recs['recommended_models']
            print(f"\n📊 RANKED CLASSIFICATION MODELS:")
            for model in models[:5]:  # Show top 5
                print(f"  #{model.get('rank', '?')}. {model.get('name', 'Unknown')}")
                print(f"      🎯 Expected Accuracy: {model.get('expected_accuracy', 'Unknown')}")
                print()
        
        if 'primary_recommendation' in recs:
            primary = recs['primary_recommendation']
            print(f"🏆 PRIMARY: {primary.get('model', 'Unknown')}")

except Exception as e:
    print(f"❌ Classification test failed: {str(e)}")

finally:
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n🧹 Cleaned up: {test_file_path}")

print("\n✅ Classification test completed!")