#!/usr/bin/env python3
"""
Test Training API with Feature Information
"""

import requests
import json

def test_training_with_features():
    """Test the training API to verify it returns feature information"""
    
    print("🧪 TESTING TRAINING API WITH FEATURE INFO")
    print("=" * 60)
    
    # Use the file ID from your log
    file_id = "745b84df-f69f-4cbb-824f-952a29bf69fe"
    model_name = "CatBoost"  # From your log message
    
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
    print(f"📤 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\n📊 Status Code: {response.status_code}")
        
        try:
            result = response.json()
            
            print(f"\n✅ SUCCESS! Training API Response:")
            print(f"📋 Success: {result.get('success')}")
            print(f"🎯 Model: {result.get('model_name')}")
            print(f"📁 File ID: {result.get('file_id')}")
            
            # Check for feature information
            feature_info = result.get('feature_info', {})
            if feature_info:
                print(f"\n🎉 FEATURE INFO AVAILABLE!")
                print(f"📊 Feature Names: {feature_info.get('feature_names', [])}")
                print(f"🎯 Target Column: {feature_info.get('target_column', 'Unknown')}")
                print(f"🔍 Problem Type: {feature_info.get('problem_type', 'Unknown')}")
                
                feature_names = feature_info.get('feature_names', [])
                print(f"\n📋 Total Features: {len(feature_names)}")
                for i, feature in enumerate(feature_names, 1):
                    print(f"   {i}. {feature}")
            else:
                print(f"\n❌ NO FEATURE INFO in response")
            
            # Check training result
            training_result = result.get('result', {})
            if training_result:
                print(f"\n📈 TRAINING RESULTS:")
                print(f"🎯 Accuracy: {training_result.get('main_score', 0)*100:.2f}%")
                print(f"📁 Model Folder: {training_result.get('model_folder', 'Unknown')}")
            
        except json.JSONDecodeError:
            print(f"❌ Non-JSON Response:")
            print(response.text[:500])
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print("💡 Make sure Flask server is running on port 5000")
        
    except requests.exceptions.Timeout:
        print(f"❌ Timeout Error: Request took longer than 60 seconds")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_training_with_features()