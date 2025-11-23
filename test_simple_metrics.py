#!/usr/bin/env python3
"""
Simple test for metrics with minimal data to avoid training failures
"""

import requests
import json
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:5000/api"

def create_simple_test_data():
    """Create a simple test dataset that will definitely work"""
    
    # Create simple regression data
    np.random.seed(42)
    n_samples = 50  # Larger sample size
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples), 
        'feature_3': np.random.normal(0, 1, n_samples),
        'target_regression': None,
        'target_classification': None
    }
    
    # Create correlated regression target
    data['target_regression'] = (
        2 * data['feature_1'] + 
        1.5 * data['feature_2'] + 
        0.5 * data['feature_3'] + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Create binary classification target
    data['target_classification'] = (data['target_regression'] > data['target_regression'].mean()).astype(int)
    
    df = pd.DataFrame(data)
    test_file = r"e:\New Codes\MP 2.o\02\simple_test_data.csv"
    df.to_csv(test_file, index=False)
    
    return test_file

def test_simple_regression():
    print("🚀 CREATING SIMPLE TEST DATA")
    test_file = create_simple_test_data()
    print(f"✅ Created: {test_file}")
    
    print("\n📊 TESTING SIMPLE REGRESSION")
    print("-" * 40)
    
    selected_columns = ["feature_1", "feature_2", "target_regression"]
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'continuous',  # Force regression
            'target_column': 'target_regression',
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Uploaded: {file_id}")
    
    training_data = {'file_id': file_id, 'model_name': 'Linear Regression'}
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=30)
        print(f"🚀 Training response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")
            
            feature_info = result.get('feature_info', {})
            performance = result.get('performance', {})
            
            print(f"\n📈 PROBLEM TYPE: {feature_info.get('problem_type', 'Unknown')}")
            print(f"🎯 TARGET: {feature_info.get('target_column', 'Unknown')}")
            
            print(f"\n📊 METRICS RETURNED:")
            for metric, value in performance.items():
                print(f"   📈 {metric}: {value:.4f}")
            
            # Analyze metrics
            if feature_info.get('problem_type') == 'regression':
                expected_metrics = ['r2_score', 'mse', 'rmse', 'mae']
                print(f"\n✅ REGRESSION METRICS CHECK:")
                for metric in expected_metrics:
                    if metric in performance:
                        print(f"   ✅ {metric}: PRESENT")
                    else:
                        print(f"   ❌ {metric}: MISSING")
            
        else:
            print(f"❌ Training failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simple_classification():
    test_file = r"e:\New Codes\MP 2.o\02\simple_test_data.csv"
    
    print("\n📊 TESTING SIMPLE CLASSIFICATION")
    print("-" * 40)
    
    selected_columns = ["feature_1", "feature_2", "target_classification"]
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'categorical',  # Force classification
            'target_column': 'target_classification',
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
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=30)
        print(f"🚀 Training response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")
            
            feature_info = result.get('feature_info', {})
            performance = result.get('performance', {})
            
            print(f"\n📈 PROBLEM TYPE: {feature_info.get('problem_type', 'Unknown')}")
            print(f"🎯 TARGET: {feature_info.get('target_column', 'Unknown')}")
            
            print(f"\n📊 METRICS RETURNED:")
            metric_values = []
            for metric, value in performance.items():
                print(f"   📈 {metric}: {value:.4f}")
                metric_values.append(value)
            
            # Check if metrics are different
            if len(set(metric_values)) > 1:
                print(f"\n✅ METRICS ARE DIFFERENT (CORRECT)")
            else:
                print(f"\n❌ ALL METRICS SAME (PROBLEM): {metric_values}")
            
        else:
            print(f"❌ Training failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_regression()
    test_simple_classification()