#!/usr/bin/env python3
"""
Quick supervised learning test
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000/api"
TEST_FILE_PATH = r"e:\New Codes\MP 2.o\02\test_dataset.csv"

def quick_test():
    print("🚀 QUICK SUPERVISED TEST")
    
    # Upload
    selected_columns = ["Weekly_Sales", "Temperature", "CPI"]  # Smaller set
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'continuous', 
            'target_column': 'CPI',
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Uploaded: {file_id}")
    
    # Quick training
    print(f"🚀 Starting training...")
    training_data = {'file_id': file_id, 'model_name': 'Linear Regression'}
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=30)
        print(f"✅ Training completed: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            feature_info = result.get('feature_info', {})
            print(f"📊 Selected columns: {feature_info.get('selected_columns', [])}")
            print(f"📊 Feature names: {feature_info.get('feature_names', [])}")
            print(f"🎯 Target: {feature_info.get('target_column')}")
        else:
            print(f"❌ Training failed: {response.text}")
            
    except requests.Timeout:
        print(f"⏱️ Training timed out after 30 seconds")

if __name__ == "__main__":
    quick_test()