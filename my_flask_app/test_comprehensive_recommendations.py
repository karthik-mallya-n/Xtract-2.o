#!/usr/bin/env python3
"""
Test the updated recommendation system with the new comprehensive model categorization
"""

import sys
import os
import pandas as pd
import json

# Add the current directory to the path so we can import core_ml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core_ml import ml_core
    print("✅ Successfully imported ml_core")
except Exception as e:
    print(f"❌ Failed to import ml_core: {str(e)}")
    exit(1)

# Create a test dataset
print("\n📊 Creating test dataset...")
test_data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
    'income': [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000, 150000, 165000] * 10,
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD', 'High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'] * 10,
    'purchase_amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] * 10  # Continuous target
}

df = pd.DataFrame(test_data)
test_file_path = 'test_recommendation_dataset.csv'
df.to_csv(test_file_path, index=False)
print(f"✅ Test dataset created: {df.shape}")
print(f"📁 Saved to: {test_file_path}")

# Test user answers for regression scenario (Labeled + Continuous)
user_answers = {
    'data_type': 'continuous',  # Target is continuous (purchase_amount)
    'is_labeled': 'labeled',    # We have a target variable
    'problem_type': 'regression',
    'data_size': 'medium',
    'accuracy_priority': 'high'
}

print(f"\n🎯 User Answers: {user_answers}")

try:
    # Analyze the dataset
    print("\n📊 Analyzing dataset...")
    dataset_analysis = ml_core.analyze_dataset(test_file_path)
    print(f"✅ Dataset analysis complete")
    print(f"   📈 Rows: {dataset_analysis['total_rows']}")
    print(f"   📊 Columns: {dataset_analysis['total_columns']}")
    print(f"   🔢 Numeric: {dataset_analysis['numeric_columns']}")
    print(f"   📝 Categorical: {dataset_analysis['categorical_columns']}")
    
    # Test the new recommendation system
    print("\n🤖 Testing new recommendation system...")
    llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
    
    if llm_response.get('success'):
        print("✅ LLM request successful!")
        recs = llm_response.get('recommendations', {})
        
        # Display the comprehensive results
        if 'scenario_detected' in recs:
            scenario = recs['scenario_detected']
            print(f"\n🎯 DETECTED SCENARIO: {scenario.get('type', 'Unknown')}")
            print(f"📝 Task Type: {scenario.get('task', 'Unknown')}")
        
        if 'semantic_analysis' in recs:
            semantic = recs['semantic_analysis']
            print(f"\n🔍 SEMANTIC ANALYSIS:")
            print(f"🏢 Domain: {semantic.get('domain', 'Unknown')}")
            print(f"💡 Insights: {semantic.get('key_insights', 'Unknown')}")
        
        if 'recommended_models' in recs:
            models = recs['recommended_models']
            print(f"\n📊 RANKED MODELS ({len(models)} models):")
            for model in models:
                print(f"  #{model.get('rank', '?')}. {model.get('name', 'Unknown')}")
                print(f"      🎯 Accuracy: {model.get('expected_accuracy', 'Unknown')}")
                print(f"      💭 Reasoning: {model.get('reasoning', 'No reasoning')[:100]}...")
                print()
        
        if 'primary_recommendation' in recs:
            primary = recs['primary_recommendation']
            print(f"🏆 PRIMARY RECOMMENDATION: {primary.get('model', 'Unknown')}")
            print(f"🎯 Confidence: {primary.get('confidence', 'Unknown')}")
    else:
        print(f"❌ LLM request failed: {llm_response.get('error', 'Unknown error')}")
        print(f"📄 Raw response: {llm_response.get('raw_response', 'No response')[:200]}...")

except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n🧹 Cleaned up: {test_file_path}")

print("\n✅ Test completed!")