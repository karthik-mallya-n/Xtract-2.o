#!/usr/bin/env python3
"""
Simple API test to check if training endpoint returns feature_info
"""

import requests
import json
import os

# Configuration
BASE_URL = "http://127.0.0.1:5000/api"
TEST_FILE_PATH = r"e:\New Codes\MP 2.o\02\test_dataset.csv"

def simple_api_test():
    """Test just the API response structure"""
    
    print("🔍 SIMPLE API TEST")
    print("=" * 50)
    
    # Upload file
    selected_columns = ["Weekly_Sales", "Temperature", "Holiday_Flag"]  # Explicitly 3 columns, no Store
    print(f"📋 Sending selected_columns: {selected_columns}")
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'unlabeled',
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Upload successful: {file_id}")
    
    # Start training
    training_data = {'file_id': file_id, 'model_name': 'K-Means'}
    response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Training failed: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    print(f"✅ Training completed")
    print(f"📊 Response keys: {list(result.keys())}")
    print(f"📊 Success: {result.get('success')}")
    
    # Check feature info
    feature_info = result.get('feature_info', {})
    print(f"📊 Feature info keys: {list(feature_info.keys())}")
    
    # Key checks
    selected_cols_result = feature_info.get('selected_columns', [])
    feature_names_result = feature_info.get('feature_names', [])
    training_feature_names = result.get('result', {}).get('training_details', {}).get('feature_names', [])
    
    print(f"\n🔍 COLUMN ANALYSIS:")
    print(f"📋 Original selected: {selected_columns}")
    print(f"📋 Feature info selected_columns: {selected_cols_result}")
    print(f"📋 Feature info feature_names: {feature_names_result}")  
    print(f"📋 Training details feature_names: {training_feature_names}")
    
    # Validation
    if set(selected_cols_result) == set(selected_columns):
        print(f"✅ Selected columns preserved correctly!")
    else:
        print(f"❌ Selected columns mismatch!")
        print(f"   Expected: {sorted(selected_columns)}")
        print(f"   Got: {sorted(selected_cols_result)}")
    
    if len(training_feature_names) == len(selected_columns):
        print(f"✅ Training used correct number of features!")
    else:
        print(f"❌ Training feature count mismatch!")
        print(f"   Expected: {len(selected_columns)} features")
        print(f"   Got: {len(training_feature_names)} features")

if __name__ == "__main__":
    simple_api_test()