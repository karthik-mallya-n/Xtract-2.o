#!/usr/bin/env python3
"""
Test the API response to verify no duplicate model IDs and proper structure
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

# Create a test dataset
print("\n📊 Creating test dataset...")
test_data = {
    'credit_score': [700, 650, 800, 580, 720, 600, 750, 680] * 20,
    'income': [50000, 35000, 80000, 25000, 60000, 30000, 70000, 45000] * 20,
    'loan_approved': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'] * 20
}

df = pd.DataFrame(test_data)
test_file_path = 'test_unique_ids.csv'
df.to_csv(test_file_path, index=False)
print(f"✅ Dataset created: {df.shape}")

# Simulate what the API endpoint does
user_answers = {
    'data_type': 'categorical',
    'is_labeled': 'labeled',
    'problem_type': 'classification',
    'data_size': 'medium',
    'accuracy_priority': 'high'
}

try:
    # Get dataset analysis
    print("\n📊 Analyzing dataset...")
    dataset_analysis = ml_core.analyze_dataset(test_file_path)
    
    # Get LLM response
    print("\n🤖 Getting LLM recommendations...")
    llm_response = ml_core.make_llm_request(user_answers, dataset_analysis)
    
    if llm_response.get('success'):
        recs = llm_response.get('recommendations', {})
        
        # Simulate the API processing (like in app.py)
        scenario_info = recs.get('scenario_detected', {})
        semantic_analysis = recs.get('semantic_analysis', {})
        ranked_models = recs.get('recommended_models', [])
        primary_recommendation = recs.get('primary_recommendation', {})
        
        if ranked_models:
            print(f"\n📋 Processing {len(ranked_models)} models...")
            
            recommended_models = []
            alternative_models = []
            
            # Process models exactly like in app.py
            for i, model in enumerate(ranked_models):
                model_name = model.get('name', f'Model {i+1}')
                unique_id = f"model_{i+1}_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')}"
                
                model_data = {
                    'id': unique_id,
                    'name': model_name,
                    'description': model.get('reasoning', model.get('advantages', '')),
                    'accuracy_estimate': model.get('expected_accuracy', 'Unknown'),
                    'reasoning': model.get('reasoning', ''),
                    'advantages': model.get('advantages', ''),
                    'rank': model.get('rank', i+1)
                }
                
                recommended_models.append(model_data)
            
            # Check for duplicates
            print("\n🔍 CHECKING FOR DUPLICATES:")
            print("="*60)
            
            all_ids = [model['id'] for model in recommended_models]
            all_names = [model['name'] for model in recommended_models]
            
            # Check ID uniqueness
            unique_ids = set(all_ids)
            if len(unique_ids) == len(all_ids):
                print("✅ All model IDs are unique")
            else:
                print(f"❌ Duplicate IDs found! {len(all_ids)} total, {len(unique_ids)} unique")
                duplicates = [id for id in all_ids if all_ids.count(id) > 1]
                print(f"   Duplicate IDs: {set(duplicates)}")
            
            # Check name uniqueness
            unique_names = set(all_names)
            if len(unique_names) == len(all_names):
                print("✅ All model names are unique")
            else:
                print(f"❌ Duplicate names found! {len(all_names)} total, {len(unique_names)} unique")
                duplicates = [name for name in all_names if all_names.count(name) > 1]
                print(f"   Duplicate names: {set(duplicates)}")
            
            print("\n📊 MODEL SUMMARY:")
            print("="*60)
            for model in recommended_models[:10]:  # Show first 10
                print(f"ID: {model['id']}")
                print(f"Name: {model['name']}")
                print(f"Rank: {model['rank']}")
                print(f"Accuracy: {model['accuracy_estimate']}")
                print("-" * 40)
            
            if len(recommended_models) > 10:
                print(f"... and {len(recommended_models) - 10} more models")
            
            print(f"\n✅ FINAL RESULT:")
            print(f"   📊 Total models: {len(recommended_models)}")
            print(f"   🆔 Unique IDs: {len(unique_ids)}")
            print(f"   📝 Unique names: {len(unique_names)}")
            print(f"   🔄 No duplicates: {'✅ Yes' if len(unique_ids) == len(all_ids) and len(unique_names) == len(all_names) else '❌ No'}")
        
        else:
            print("❌ No ranked models found in response")
    
    else:
        print(f"❌ LLM request failed: {llm_response.get('error')}")

except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n🧹 Cleaned up: {test_file_path}")

print("\n✅ Duplicate check test completed!")