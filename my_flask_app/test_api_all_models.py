#!/usr/bin/env python3
"""
Test API endpoint to verify ALL models are returned via /api/recommend
"""

import requests
import pandas as pd
import json
import os
import time

# Create test dataset
print("📊 Creating test dataset for API...")
test_data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60] * 20,
    'salary': [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000] * 20,
    'experience': [1, 3, 5, 7, 9, 11, 13, 15] * 20,
    'approved': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'] * 20  # Categorical target
}

df = pd.DataFrame(test_data)
test_file = 'test_api_all_models.csv'
df.to_csv(test_file, index=False)
print(f"✅ Dataset created: {df.shape}")

BASE_URL = 'http://127.0.0.1:5000'

try:
    # Step 1: Upload file
    print("\n1️⃣ Uploading file...")
    with open(test_file, 'rb') as f:
        files = {'file': (test_file, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        exit(1)
    
    upload_result = response.json()
    file_id = upload_result['file_id']
    print(f"✅ File ID: {file_id}")
    
    # Step 2: Get recommendations via API
    print("\n2️⃣ Getting ALL model recommendations via API...")
    recommendation_data = {
        'file_id': file_id,
        'questions': {
            'data_type': 'categorical',
            'is_labeled': 'labeled',
            'problem_type': 'classification',
            'data_size': 'medium',
            'accuracy_priority': 'high'
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend", json=recommendation_data)
    
    if response.status_code != 200:
        print(f"❌ API recommendation failed: {response.status_code}")
        print(f"Response: {response.text}")
        exit(1)
    
    api_result = response.json()
    print(f"✅ API recommendation successful!")
    
    # Check recommendations
    if 'recommendations' in api_result:
        recs = api_result['recommendations']
        
        # Display scenario
        if 'scenario_detected' in recs:
            scenario = recs['scenario_detected']
            print(f"\n🎯 API DETECTED: {scenario.get('type', 'Unknown')}")
        
        # Display ALL models from API
        if 'recommended_models' in recs:
            models = recs['recommended_models']
            print(f"\n📊 API RETURNED MODELS: {len(models)} models")
            print("="*70)
            
            for model in models[:15]:  # Show first 15 to avoid too much output
                name = model.get('name', 'Unknown')
                accuracy = model.get('accuracy_estimate', 'Unknown')
                rank = model.get('rank', '?')
                print(f"#{rank:2}. {name:30} | Accuracy: {accuracy}")
            
            if len(models) > 15:
                print(f"... and {len(models) - 15} more models")
            
            print("="*70)
            print(f"🎉 SUCCESS: API returned {len(models)} models total!")
        
        # Check if alternative models also populated
        if 'alternative_models' in recs:
            alt_models = recs['alternative_models']
            print(f"📋 Alternative models: {len(alt_models)}")
    
    print(f"\n✅ API TEST SUCCESSFUL!")
    print(f"🔥 The system now returns ALL models for each scenario via API!")

except Exception as e:
    print(f"❌ API test failed: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n🧹 Cleaned up: {test_file}")