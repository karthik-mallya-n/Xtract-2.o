#!/usr/bin/env python3
"""
Test to validate that the model is properly trained on Walmart dataset
"""

import requests
import json
import pandas as pd

def test_walmart_predictions():
    """Test multiple Walmart-like predictions to validate model training"""
    
    # Test cases with different Walmart store scenarios
    test_cases = [
        {
            "name": "Store 1 - Normal Day",
            "data": {
                'Store': 1,
                'Date': '2010-02-05', 
                'Weekly_Sales': 1643690.90,
                'Holiday_Flag': 0,
                'Temperature': 42.31,
                'Fuel_Price': 2.572,
                'CPI': 211.096327
            }
        },
        {
            "name": "Store 20 - Holiday Period",
            "data": {
                'Store': 20,
                'Date': '2010-12-25', 
                'Weekly_Sales': 2500000.00,
                'Holiday_Flag': 1,
                'Temperature': 35.0,
                'Fuel_Price': 3.0,
                'CPI': 215.0
            }
        },
        {
            "name": "Store 45 - Summer Period",
            "data": {
                'Store': 45,
                'Date': '2011-07-15', 
                'Weekly_Sales': 1800000.0,
                'Holiday_Flag': 0,
                'Temperature': 85.5,
                'Fuel_Price': 3.5,
                'CPI': 220.0
            }
        }
    ]
    
    print("🏪 Testing Walmart Dataset Model Performance")
    print("=" * 60)
    
    predictions = []
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        
        payload = {
            'file_id': 'test_walmart',
            'input_data': test_case['data']
        }
        
        try:
            response = requests.post('http://localhost:5000/api/predict', json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = float(result['prediction'])
                predictions.append(prediction)
                
                print(f"✅ Prediction: {prediction:.3f}")
                print(f"   Features: Store={test_case['data']['Store']}, Sales=${test_case['data']['Weekly_Sales']:,.0f}")
                print(f"   Holiday: {'Yes' if test_case['data']['Holiday_Flag'] else 'No'}, Temp: {test_case['data']['Temperature']}°F")
                
            else:
                print(f"❌ Error: {response.json().get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
    
    # Analyze predictions
    if predictions:
        print(f"\n📈 Prediction Analysis:")
        print(f"   Range: {min(predictions):.3f} - {max(predictions):.3f}")
        print(f"   Average: {sum(predictions)/len(predictions):.3f}")
        print(f"   Variation: {max(predictions) - min(predictions):.3f}")
        
        # Check if predictions are reasonable for unemployment rate
        reasonable = all(0 <= p <= 20 for p in predictions)  # Unemployment typically 0-20%
        print(f"   Reasonable range: {'✅ Yes' if reasonable else '❌ No'}")
        
        # Check if model shows variation (not all same prediction)
        has_variation = len(set(round(p, 2) for p in predictions)) > 1
        print(f"   Shows variation: {'✅ Yes' if has_variation else '❌ No (possibly overfit)'}")
        
        if reasonable and has_variation:
            print(f"\n🎉 Model appears to be properly trained on Walmart data!")
        else:
            print(f"\n⚠️  Model may need improvement")
    else:
        print(f"\n❌ No successful predictions obtained")

if __name__ == "__main__":
    test_walmart_predictions()