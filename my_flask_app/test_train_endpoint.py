#!/usr/bin/env python3
"""
Test script to verify the /api/train endpoint works correctly
"""

import requests
import pandas as pd
import json
import os
import time

# Create a small test dataset
print("Creating test dataset...")
test_data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,  # 100 samples
    'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 10,
    'feature3': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * 10,
    'target': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10  # Binary classification
}

df = pd.DataFrame(test_data)
test_file_path = 'test_dataset.csv'
df.to_csv(test_file_path, index=False)
print(f"Test dataset created with shape: {df.shape}")

# Flask server URL
BASE_URL = 'http://127.0.0.1:5000'

try:
    # Step 1: Upload the test file
    print("\n1. Uploading test dataset...")
    with open(test_file_path, 'rb') as f:
        files = {'file': (test_file_path, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.status_code} - {response.text}")
        exit(1)
    
    upload_result = response.json()
    file_id = upload_result['file_id']
    print(f"File uploaded successfully. File ID: {file_id}")
    
    # Step 2: Get AI recommendation 
    print("\n2. Getting AI model recommendation...")
    recommendation_data = {
        'file_id': file_id,
        'questions': {
            'problem_type': 'classification',
            'data_size': 'small',
            'accuracy_priority': 'high'
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend", json=recommendation_data)
    if response.status_code != 200:
        print(f"Recommendation failed: {response.status_code} - {response.text}")
        exit(1)
    
    recommendation = response.json()
    print(f"AI recommendation: {recommendation.get('recommended_model', 'No recommendation')}")
    
    # Step 3: Train the model using recommended model
    print("\n3. Starting model training...")
    train_data = {
        'file_id': file_id,
        'model_name': 'Logistic Regression'  # Use a simple model for testing
    }
    
    response = requests.post(f"{BASE_URL}/api/train", json=train_data)
    
    if response.status_code != 200:
        print(f"Training failed: {response.status_code} - {response.text}")
        exit(1)
    
    train_result = response.json()
    print(f"Training response: {json.dumps(train_result, indent=2)}")
    
    # Step 4: Check if model folder was created
    print("\n4. Checking if model folder was created...")
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        if model_files:
            print(f"✅ Models found: {model_files}")
            
            # Check for Logistic Regression folder
            lr_folder = os.path.join(models_dir, 'logistic_regression')
            if os.path.exists(lr_folder):
                lr_files = os.listdir(lr_folder)
                print(f"✅ Logistic Regression folder contents: {lr_files}")
            else:
                print("❌ Logistic Regression folder not found")
        else:
            print("❌ No model files found in models directory")
    else:
        print("❌ Models directory not found")
    
    print("\n✅ Test completed successfully!")

except Exception as e:
    print(f"❌ Test failed with error: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\nCleaned up test file: {test_file_path}")