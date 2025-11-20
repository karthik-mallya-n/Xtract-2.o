#!/usr/bin/env python3
"""
Test script to verify prediction functionality works with the new format
"""

import requests
import json
import time

# Test both prediction formats
def test_prediction_formats():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Prediction API Formats")
    print("=" * 50)
    
    # Test Format 1: Features array (React frontend format)
    test_features = [1.5, -0.5, 2.1, 0.8, -1.2, 0.3, 1.8, -0.2, 1.1, 0.5, 
                    -0.8, 1.7, 0.2, -1.1, 0.9, 1.3, -0.4, 0.6, 1.0, -0.7]
    
    print("\n📊 Test 1: Features Array Format (React Frontend)")
    print("-" * 50)
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json={"features": test_features},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Model Info: {result.get('model_info', {})}")
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
    
    # Test Format 2: Original format (for completeness)
    print("\n📊 Test 2: Original Format (with input_data object)")
    print("-" * 50)
    try:
        input_data = {f"feature_{i}": val for i, val in enumerate(test_features)}
        response = requests.post(
            f"{base_url}/api/predict",
            json={
                "file_id": "test",  # This will use most recent model
                "input_data": input_data
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")

if __name__ == "__main__":
    # Wait a moment for Flask to fully start
    time.sleep(2)
    test_prediction_formats()