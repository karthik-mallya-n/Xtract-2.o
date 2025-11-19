#!/usr/bin/env python3
"""
Test with Real Upload File ID
"""

import requests
import json

def test_real_upload():
    """Test training with a real uploaded file"""
    
    print("🧪 TESTING WITH REAL UPLOADED FILE")
    print("=" * 50)
    
    # Use a real file ID from uploads folder
    file_id = "a0c0f494-6319-4a34-a708-f09796a60f92"  # Walmart dataset
    model_name = "Lasso Regression"  # As requested by user
    
    print(f"📊 File ID: {file_id}")
    print(f"🤖 Model: {model_name}")
    
    # API endpoint
    url = "http://localhost:5000/api/train"
    
    # Payload
    payload = {
        "file_id": file_id,
        "model_name": model_name
    }
    
    print(f"\n🌐 Calling: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS!")
            print(f"🎯 Model: {result.get('model_name')}")
            
            # Check feature information
            feature_info = result.get('feature_info', {})
            if feature_info:
                print(f"\n🎉 FEATURE INFO:")
                print(f"📊 Features: {feature_info.get('feature_names', [])}")
                print(f"🎯 Target: {feature_info.get('target_column', 'Unknown')}")
                print(f"🔍 Type: {feature_info.get('problem_type', 'Unknown')}")
                
                expected_features = ['Store', 'Date', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
                actual_features = feature_info.get('feature_names', [])
                
                if 'Weekly_Sales' in actual_features:
                    print(f"❌ ERROR: Target column 'Weekly_Sales' should not be in features!")
                    
                if any(feat in actual_features for feat in expected_features):
                    print(f"✅ CORRECT: Found Walmart dataset features!")
                else:
                    print(f"❌ WRONG: Got Iris features instead of Walmart features!")
                    
            else:
                print(f"❌ NO FEATURE INFO")
                
            # Check training results
            training_result = result.get('result', {})
            if training_result:
                print(f"\n📈 TRAINING RESULTS:")
                print(f"🎯 Score: {training_result.get('main_score', 0):.4f}")
                print(f"📁 Model: {training_result.get('model_folder', 'Unknown')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_real_upload()