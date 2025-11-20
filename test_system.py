#!/usr/bin/env python3
"""
Test script to verify the ML system is working with comprehensive training and predictions
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_FILE = "walmart_sample.csv"

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"✅ Health Check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def upload_file():
    """Upload the test file"""
    try:
        with open(TEST_FILE, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            file_id = data.get('file_id')
            print(f"✅ File Upload: {response.status_code}, File ID: {file_id}")
            return file_id
        else:
            print(f"❌ File Upload Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ File Upload Error: {e}")
        return None

def test_training(file_id):
    """Test comprehensive training"""
    try:
        training_data = {
            "file_id": file_id,
            "model_name": "Neural Networks (MLP Regressor)",
            "target_column": "Unemployment",
            "user_data": {}
        }
        
        print(f"🚀 Starting Training...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/train", 
            json=training_data,
            timeout=600  # 10 minute timeout
        )
        
        training_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training Complete: {response.status_code}")
            print(f"⏱️  Training took: {training_time:.2f} seconds")
            print(f"📊 Performance: {data.get('performance', {})}")
            return True
        else:
            print(f"❌ Training Failed: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training Error: {e}")
        return False

def test_prediction(file_id):
    """Test prediction functionality"""
    try:
        # Test prediction with sample data
        prediction_data = {
            "file_id": file_id,
            "input_data": {
                "Store": 1,
                "Date": "05-02-2010",
                "Weekly_Sales": 1643690.9,
                "Holiday_Flag": 0,
                "Temperature": 42.31,
                "Fuel_Price": 2.572,
                "CPI": 211.096358
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/predict", json=prediction_data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction Success: {response.status_code}")
            print(f"🔮 Prediction: {data.get('prediction')}")
            print(f"📊 Model Info: {data.get('model_info', {})}")
            return True
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting ML System Tests")
    print("="*50)
    
    # Test 1: Health Check
    if not test_health():
        print("❌ Server is not running!")
        return
    
    # Test 2: File Upload  
    file_id = upload_file()
    if not file_id:
        print("❌ File upload failed!")
        return
    
    # Test 3: Training (this should show comprehensive logging and realistic timing)
    print("\n" + "="*50)
    print("🔥 TESTING COMPREHENSIVE TRAINING")
    print("="*50)
    
    if not test_training(file_id):
        print("❌ Training failed!")
        return
    
    # Test 4: Prediction
    print("\n" + "="*50) 
    print("🔮 TESTING PREDICTION")
    print("="*50)
    
    if not test_prediction(file_id):
        print("❌ Prediction failed!")
        return
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)

if __name__ == "__main__":
    main()