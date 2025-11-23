#!/usr/bin/env python3
"""
Test regression training and verify frontend metrics
"""

import requests
import json
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:5000/api"

def create_regression_test_data():
    """Create regression test data"""
    np.random.seed(42)
    n_samples = 100  # Good size for training
    
    data = {
        'feature_1': np.random.normal(5, 2, n_samples),
        'feature_2': np.random.normal(10, 3, n_samples), 
        'feature_3': np.random.normal(0, 1, n_samples),
    }
    
    # Create correlated regression target
    data['target'] = (
        2 * data['feature_1'] + 
        0.5 * data['feature_2'] + 
        1.5 * data['feature_3'] + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    df = pd.DataFrame(data)
    test_file = r"e:\New Codes\MP 2.o\02\regression_test.csv"
    df.to_csv(test_file, index=False)
    
    return test_file

def test_regression_frontend():
    print("🧪 TESTING REGRESSION FOR FRONTEND DISPLAY")
    print("=" * 60)
    
    test_file = create_regression_test_data()
    print(f"✅ Created: {test_file}")
    
    selected_columns = ["feature_1", "feature_2", "target"]
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        data = {
            'is_labeled': 'labeled',
            'data_type': 'continuous',  # Force regression
            'target_column': 'target',
            'selected_columns': json.dumps(selected_columns)
        }
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    file_id = response.json()['file_id']
    print(f"✅ Uploaded: {file_id}")
    
    # Train regression model
    training_data = {'file_id': file_id, 'model_name': 'Linear Regression'}
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=60)
        print(f"🚀 Training response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")
            
            feature_info = result.get('feature_info', {})
            performance = result.get('performance', {})
            
            print(f"\n📈 FEATURE INFO:")
            print(f"   🎯 Problem type: {feature_info.get('problem_type', 'Unknown')}")
            print(f"   📊 Target: {feature_info.get('target_column', 'Unknown')}")
            
            print(f"\n📊 REGRESSION METRICS FOR FRONTEND:")
            for metric, value in performance.items():
                print(f"   📈 {metric}: {value}")
            
            # Save for frontend testing
            print(f"\n💾 SAVING FOR FRONTEND TEST:")
            frontend_data = {
                'success': result.get('success', False),
                'model_name': 'Linear Regression',
                'main_score': performance.get('r2_score', 0),
                'threshold_met': True,
                'performance': performance,
                'feature_info': feature_info,
                'training_details': result.get('training_details', {}),
                'model_info': result.get('model_info', {})
            }
            
            # Store in localStorage format for frontend
            with open('frontend_test_data.json', 'w') as f:
                json.dump(frontend_data, f, indent=2)
            
            print(f"✅ Frontend test data saved to frontend_test_data.json")
            print(f"📊 Problem type: {feature_info.get('problem_type')}")
            print(f"📊 Has regression metrics: {any(k in performance for k in ['r2_score', 'mse', 'rmse', 'mae'])}")
            
        else:
            print(f"❌ Training failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_regression_frontend()