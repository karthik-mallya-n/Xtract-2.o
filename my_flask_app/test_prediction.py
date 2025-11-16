#!/usr/bin/env python3
"""
Test script for the fixed prediction API
"""

import requests
import json

def test_prediction():
    """Test prediction with Walmart dataset features"""
    
    # Test data with Walmart dataset features
    test_data = {
        'file_id': 'test_walmart',
        'input_data': {
            'Store': 1,
            'Date': '2010-02-05', 
            'Weekly_Sales': 1643690.90,
            'Holiday_Flag': 0,
            'Temperature': 42.31,
            'Fuel_Price': 2.572,
            'CPI': 211.096327
        }
    }
    
    try:
        print("🧪 Testing prediction API...")
        print(f"📊 Input data: {test_data['input_data']}")
        
        response = requests.post('http://localhost:5000/api/predict', json=test_data)
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"🎯 Prediction: {result.get('prediction', 'N/A')}")
            print(f"📈 Raw prediction: {result.get('raw_prediction', 'N/A')}")
            print(f"🤖 Model: {result.get('model_info', {}).get('model_name', 'Unknown')}")
            print(f"📊 Feature info: {result.get('feature_info', {})}")
        else:
            result = response.json()
            print("❌ Prediction failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_prediction()