#!/usr/bin/env python3
"""
Test script for supervised learning column filtering
"""

import requests
import json
import os

# Configuration
BASE_URL = "http://127.0.0.1:5000/api"
TEST_FILE_PATH = r"e:\New Codes\MP 2.o\02\test_dataset.csv"

def test_supervised_column_filtering():
    """Test column filtering for supervised learning (regression)"""
    
    print("🎯 SUPERVISED LEARNING COLUMN FILTERING TEST")
    print("=" * 60)
    
    # Upload file for supervised learning
    selected_columns = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    target_column = "CPI"  # Use CPI as target for regression
    
    print(f"📋 Selected columns: {selected_columns}")
    print(f"🎯 Target column: {target_column}")
    print(f"📊 Expected features: {[col for col in selected_columns if col != target_column]}")
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'continuous',
            'target_column': target_column,
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    file_id = response.json()['file_id']
    print(f"✅ Upload successful: {file_id}")
    
    # Start supervised training
    training_data = {'file_id': file_id, 'model_name': 'Linear Regression'}
    print(f"🚀 Training with: {training_data['model_name']}")
    
    response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=120)
    
    if response.status_code != 200:
        print(f"❌ Training failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    result = response.json()
    print(f"✅ Training completed")
    print(f"📊 Success: {result.get('success')}")
    
    # Analyze results
    feature_info = result.get('feature_info', {})
    print(f"\n📋 RESULT ANALYSIS:")
    print(f"📊 Feature info keys: {list(feature_info.keys())}")
    
    selected_cols_result = feature_info.get('selected_columns', [])
    feature_names_result = feature_info.get('feature_names', [])
    target_col_result = feature_info.get('target_column')
    
    print(f"📋 Selected columns in result: {selected_cols_result}")
    print(f"📋 Feature names in result: {feature_names_result}")
    print(f"🎯 Target column in result: {target_col_result}")
    
    # Validation
    expected_features = [col for col in selected_columns if col != target_column]
    
    print(f"\n🔍 VALIDATION:")
    
    # Test 1: Selected columns should match what was sent
    if set(selected_cols_result) == set(selected_columns):
        print(f"✅ TEST 1 PASSED: Selected columns preserved correctly")
    else:
        print(f"❌ TEST 1 FAILED: Selected columns mismatch")
        print(f"   Expected: {sorted(selected_columns)}")
        print(f"   Got: {sorted(selected_cols_result)}")
    
    # Test 2: Feature names should exclude target
    if set(feature_names_result) == set(expected_features):
        print(f"✅ TEST 2 PASSED: Feature names correct (target excluded)")
    else:
        print(f"❌ TEST 2 FAILED: Feature names incorrect")
        print(f"   Expected: {sorted(expected_features)}")
        print(f"   Got: {sorted(feature_names_result)}")
    
    # Test 3: Target column should be preserved
    if target_col_result == target_column:
        print(f"✅ TEST 3 PASSED: Target column preserved")
    else:
        print(f"❌ TEST 3 FAILED: Target column incorrect")
        print(f"   Expected: {target_column}")
        print(f"   Got: {target_col_result}")
    
    # Print full feature_info for debugging
    print(f"\n📋 FULL FEATURE INFO:")
    print(json.dumps(feature_info, indent=2))
    
    return True

if __name__ == "__main__":
    try:
        test_supervised_column_filtering()
    except Exception as e:
        print(f"💥 TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()