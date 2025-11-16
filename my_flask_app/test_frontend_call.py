#!/usr/bin/env python3
"""
Test frontend-style API call to verify prediction endpoint
"""

import requests
import json

def test_frontend_api_call():
    """Test the API call in the same format the frontend will use"""
    
    # Test data in the same format as frontend sends
    payload = {
        'file_id': 'current_model',
        'input_data': {
            'Store': 1,
            'Date': 2010,  # Simplified date format
            'Weekly_Sales': 1643690.90,
            'Holiday_Flag': 0,
            'Temperature': 42.31,
            'Fuel_Price': 2.572,
            'CPI': 211.096327
        }
    }
    
    try:
        print("🌐 Testing frontend-style API call...")
        print(f"📊 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post('http://localhost:5000/api/predict', 
                                json=payload,
                                headers={'Content-Type': 'application/json'})
        
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"🎯 Prediction: {result.get('prediction')}")
            print(f"📈 Raw prediction: {result.get('raw_prediction')}")
            print(f"🔒 Confidence: {result.get('confidence')}")
            print(f"📊 Model: {result.get('model_info', {}).get('model_name')}")
            print(f"🔧 Feature info: {result.get('feature_info', {})}")
            
            # Check if frontend can parse this
            is_regression = result.get('feature_info', {}).get('problem_type') == 'regression'
            prediction_value = result.get('prediction', 'Unknown')
            
            print(f"\n🖥️  Frontend processing:")
            print(f"   Problem type: {'Regression' if is_regression else 'Classification'}")
            print(f"   Display value: {prediction_value}")
            
            if is_regression and prediction_value != 'Unknown':
                try:
                    num_pred = float(prediction_value)
                    formatted = f"{num_pred:.3f}"
                    print(f"   Formatted: {formatted}")
                except:
                    print(f"   Could not format as number")
        else:
            result = response.json()
            print("❌ Error!")
            print(f"Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_frontend_api_call()