#!/usr/bin/env python3
"""
Test the Labeled + Categorical scenario to verify ALL classification models are returned
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

# Create a clear classification dataset (Labeled + Categorical)
print("\n📊 Creating Labeled + Categorical test dataset...")
test_data = {
    'credit_score': [800, 750, 680, 620, 590, 780, 720, 650, 580, 770, 700, 600, 550, 820, 740] * 10,
    'income': [80000, 70000, 50000, 40000, 30000, 75000, 60000, 45000, 25000, 85000, 55000, 35000, 20000, 90000, 65000] * 10,
    'debt_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.55, 0.12, 0.3, 0.45, 0.6, 0.08, 0.28] * 10,
    'employment_years': [10, 8, 5, 3, 1, 12, 7, 4, 2, 15, 6, 3, 1, 20, 9] * 10,
    'loan_approved': ['Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes'] * 10  # Categorical target
}

df = pd.DataFrame(test_data)
test_file_path = 'test_all_classification_models.csv'
df.to_csv(test_file_path, index=False)
print(f"✅ Classification dataset created: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print(f"🎯 Target: loan_approved (categorical)")

# User answers for classification scenario (Labeled + Categorical)
user_answers = {
    'data_type': 'categorical',  # Target is categorical
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
    
    # Test the recommendation system for ALL classification models
    print("\n🤖 Getting ALL classification model recommendations...")
    llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
    
    if llm_response.get('success'):
        print("✅ Classification LLM request successful!")
        recs = llm_response.get('recommendations', {})
        
        # Display scenario detection
        if 'scenario_detected' in recs:
            scenario = recs['scenario_detected']
            print(f"\n🎯 DETECTED SCENARIO: {scenario.get('type', 'Unknown')}")
            print(f"📝 TASK: {scenario.get('task', 'Unknown')}")
        
        # Display ALL classification models
        if 'recommended_models' in recs:
            models = recs['recommended_models']
            print(f"\n📊 ALL CLASSIFICATION MODELS (Total: {len(models)} models):")
            print("="*80)
            
            expected_classification_models = [
                'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'Naive Bayes',
                'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 
                'CatBoost', 'Neural Network', 'MLP Classifier'
            ]
            
            for model in models:
                rank = model.get('rank', '?')
                name = model.get('name', 'Unknown')
                accuracy = model.get('expected_accuracy', 'Unknown')
                print(f"#{rank:2}. {name:35} | Accuracy: {accuracy}")
            
            print("="*80)
            print(f"✅ Total models returned: {len(models)}")
            
            # Check if we got all expected models
            returned_model_names = [model.get('name', '').lower() for model in models]
            found_models = []
            missing_models = []
            
            for expected in expected_classification_models:
                found = any(expected.lower() in name for name in returned_model_names)
                if found:
                    found_models.append(expected)
                else:
                    missing_models.append(expected)
            
            print(f"\n📊 ANALYSIS:")
            print(f"✅ Found models: {len(found_models)}")
            print(f"❓ Missing models: {len(missing_models)}")
            
            if missing_models:
                print(f"Missing: {missing_models}")
        
        # Display primary recommendation
        if 'primary_recommendation' in recs:
            primary = recs['primary_recommendation']
            print(f"\n🏆 PRIMARY: {primary.get('model', 'Unknown')}")
    else:
        print(f"❌ LLM request failed: {llm_response.get('error', 'Unknown error')}")

except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n🧹 Cleaned up: {test_file_path}")

print("\n✅ ALL CLASSIFICATION MODELS test completed!")