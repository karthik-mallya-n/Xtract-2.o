#!/usr/bin/env python3
"""
Test regression vs classification metrics fix
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000/api"
TEST_FILE_PATH = r"e:\New Codes\MP 2.o\02\test_dataset.csv"

def test_regression_metrics():
    print("🧪 TESTING REGRESSION METRICS FIX")
    print("="*60)
    
    # Test 1: Regression with continuous data
    print("\n📊 TEST 1: REGRESSION (data_type: continuous)")
    print("-" * 50)
    
    selected_columns = ["Store", "Temperature", "Weekly_Sales"]
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'continuous',  # EXPLICIT regression choice
            'target_column': 'Weekly_Sales',
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Uploaded: {file_id}")
    
    # Test smaller dataset to avoid training issues
    training_data = {'file_id': file_id, 'model_name': 'Linear Regression'}
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=60)
        print(f"🚀 Training response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training successful: {result.get('success', False)}")
            
            feature_info = result.get('feature_info', {})
            performance = result.get('performance', {})
            
            print(f"\n📈 FEATURE INFO:")
            print(f"   🎯 Problem type: {feature_info.get('problem_type', 'Unknown')}")
            print(f"   📊 Target: {feature_info.get('target_column', 'Unknown')}")
            print(f"   🔢 Selected columns: {feature_info.get('selected_columns', [])}")
            
            print(f"\n📊 PERFORMANCE METRICS:")
            for metric, value in performance.items():
                print(f"   📈 {metric}: {value}")
            
            # Check if regression metrics are present
            expected_regression_metrics = ['r2_score', 'mse', 'rmse', 'mae']
            unexpected_classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            print(f"\n✅ REGRESSION METRICS CHECK:")
            for metric in expected_regression_metrics:
                if metric in performance:
                    print(f"   ✅ {metric}: {performance[metric]} (CORRECT)")
                else:
                    print(f"   ❌ {metric}: MISSING")
            
            print(f"\n❌ CLASSIFICATION METRICS CHECK (should be absent):")
            for metric in unexpected_classification_metrics:
                if metric in performance:
                    print(f"   ❌ {metric}: {performance[metric]} (SHOULD NOT BE PRESENT)")
                else:
                    print(f"   ✅ {metric}: Correctly absent")
            
        else:
            print(f"❌ Training failed: {response.text}")
            
    except requests.Timeout:
        print(f"⏱️ Training timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_classification_metrics():
    print("\n\n🧪 TESTING CLASSIFICATION METRICS")
    print("="*60)
    
    # Test 2: Classification with categorical data
    print("\n📊 TEST 2: CLASSIFICATION (data_type: categorical)")
    print("-" * 50)
    
    selected_columns = ["Weekly_Sales", "Temperature", "Holiday_Flag"]
    
    with open(TEST_FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'categorical',  # EXPLICIT classification choice
            'target_column': 'Holiday_Flag',  # Binary categorical target
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Uploaded: {file_id}")
    
    training_data = {'file_id': file_id, 'model_name': 'Random Forest'}
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=60)
        print(f"🚀 Training response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training successful: {result.get('success', False)}")
            
            feature_info = result.get('feature_info', {})
            performance = result.get('performance', {})
            
            print(f"\n📈 FEATURE INFO:")
            print(f"   🎯 Problem type: {feature_info.get('problem_type', 'Unknown')}")
            print(f"   📊 Target: {feature_info.get('target_column', 'Unknown')}")
            
            print(f"\n📊 PERFORMANCE METRICS:")
            for metric, value in performance.items():
                print(f"   📈 {metric}: {value}")
            
            # Check if classification metrics are present and different
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            print(f"\n✅ CLASSIFICATION METRICS CHECK:")
            values = []
            for metric in expected_metrics:
                if metric in performance:
                    value = performance[metric]
                    values.append(value)
                    print(f"   ✅ {metric}: {value}")
                else:
                    print(f"   ❌ {metric}: MISSING")
            
            # Check if all values are different (not the same)
            if len(set(values)) > 1:
                print(f"\n✅ METRICS ARE DIFFERENT (CORRECT): {values}")
            else:
                print(f"\n❌ ALL METRICS SAME (PROBLEM): {values}")
            
        else:
            print(f"❌ Training failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_regression_metrics()
    test_classification_metrics()