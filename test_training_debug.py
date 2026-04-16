#!/usr/bin/env python3
"""
Test script to verify the training pipeline respects user's data_type choice
"""
import requests
import json
import time

# API endpoints
UPLOAD_URL = "http://localhost:5000/upload"
ANALYZE_URL = "http://localhost:5000/analyze"
TRAIN_URL = "http://localhost:5000/train"
RESULTS_URL = "http://localhost:5000/results"

# Test data - Wine Quality Dataset
TEST_FILE = "uploads/3a506899-8cbd-4409-8a4a-f2f72818d29c.csv"

def test_training():
    print("\n" + "="*80)
    print("Testing Training Pipeline with User Data Type Choice")
    print("="*80 + "\n")
    
    try:
        # Step 1: Upload file
        print("[1/4] Uploading test file...")
        with open(TEST_FILE, 'rb') as f:
            files = {'file': f}
            data = {
                'is_labeled': 'labeled',
                'data_type': 'continuous',  # User explicitly says CONTINUOUS
                'target_column': 'quality',
                'selected_columns': json.dumps([
                    'fixed acidity', 'volatile acidity', 'citric acid', 
                    'residual sugar', 'chlorides', 'free sulfur dioxide', 
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'
                ])
            }
            response = requests.post(UPLOAD_URL, files=files, data=data)
            
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            return
        
        upload_result = response.json()
        file_id = upload_result.get('file_id')
        print(f"✅ File uploaded successfully. File ID: {file_id}\n")
        
        # Step 2: Train model
        print("[2/4] Starting model training with XGBoost Regressor...")
        print("   (User has selected: data_type='continuous' - should use REGRESSION)\n")
        
        train_data = {
            'file_id': file_id,
            'model_name': 'XGBoost Regressor'
        }
        
        response = requests.post(TRAIN_URL, json=train_data)
        
        if response.status_code != 200:
            print(f"❌ Training request failed: {response.text}")
            return
        
        train_result = response.json()
        print(f"✅ Training request submitted\n")
        
        # Step 3: Wait and get results
        print("[3/4] Waiting for training to complete...")
        time.sleep(5)  # Wait for training to complete
        
        results_response = requests.get(f"{RESULTS_URL}?file_id={file_id}")
        if results_response.status_code == 200:
            results = results_response.json()
            print(f"Result: {results.get('success')}")
            if not results.get('success'):
                print(f"Error: {results.get('error')}\n")
        
        # Step 4: Check Docker logs
        print("[4/4] Checking Docker logs for debug output...\n")
        import subprocess
        try:
            logs = subprocess.check_output(
                ['docker', 'logs', 'xtract-2o-backend-1', '--tail=200'],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            lines = logs.split('\n')
            capture = False
            for line in lines:
                if "COMPREHENSIVE TRAINING FOR: XGBOOST" in line:
                    capture = True
                if capture:
                    if "DEBUG: user_data" in line or "DEBUG: explicit_data_type" in line or \
                       "DEBUG: unique_targets" in line or "FORCING REGRESSION" in line or \
                       "Auto-detected problem type" in line or "Problem type: " in line:
                        print(line)
                    if "TRAINING FAILED" in line or "TRAINING COMPLETED" in line:
                        capture = False
        except Exception as e:
            print(f"Could not read Docker logs: {e}")
        
        print("\n" + "="*80)
        print("Test Complete")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()
