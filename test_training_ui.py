#!/usr/bin/env python3
"""
Test script to verify training results are properly formatted for the UI
"""

import requests
import json
import time

def test_training_results():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Training Results Format")
    print("=" * 50)
    
    # First, upload a test file
    print("\n📤 Step 1: Uploading test dataset")
    try:
        with open('test_classification.csv', 'rb') as f:
            files = {'file': f}
            data = {
                'is_labeled': 'true',
                'data_type': 'classification'
            }
            response = requests.post(f"{base_url}/api/upload", files=files, data=data)
            
        if response.status_code == 200:
            upload_result = response.json()
            file_id = upload_result['file_id']
            print(f"✅ File uploaded successfully: {file_id}")
        else:
            print(f"❌ Upload failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Upload exception: {str(e)}")
        return
    
    # Test training with specific model
    print(f"\n🏋️ Step 2: Training Decision Tree model")
    try:
        response = requests.post(
            f"{base_url}/api/train-specific-model",
            json={
                "file_id": file_id,
                "model_name": "decision-tree-classifier"
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ TRAINING SUCCESS!")
            print(f"Training Results Structure:")
            print(f"- success: {result.get('success')}")
            print(f"- performance: {result.get('training_results', {}).get('performance', {})}")
            print(f"- training_details: {result.get('training_results', {}).get('training_details', {})}")
            print(f"- model_info: {result.get('training_results', {}).get('model_info', {})}")
            
            # Check if UI-friendly metrics are present
            perf = result.get('training_results', {}).get('performance', {})
            details = result.get('training_results', {}).get('training_details', {})
            
            print(f"\n📊 UI Metrics Check:")
            print(f"- Accuracy: {perf.get('accuracy', 'MISSING')}")
            print(f"- Precision: {perf.get('precision', 'MISSING')}")
            print(f"- Recall: {perf.get('recall', 'MISSING')}")
            print(f"- F1-Score: {perf.get('f1_score', 'MISSING')}")
            print(f"- Training Samples: {details.get('training_samples', 'MISSING')}")
            print(f"- Test Samples: {details.get('test_samples', 'MISSING')}")
            print(f"- Features: {details.get('features', 'MISSING')}")
            print(f"- Training Time: {details.get('training_time', 'MISSING')}")
        else:
            print("❌ TRAINING FAILED!")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ TRAINING EXCEPTION: {str(e)}")

if __name__ == "__main__":
    # Wait a moment for Flask to be ready
    time.sleep(1)
    test_training_results()