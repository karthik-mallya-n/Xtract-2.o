#!/usr/bin/env python3
"""
Test Training API with real file ID
"""

import requests
import json

def test_training_api():
    """Test the training API with the file ID from your logs"""
    
    print("🧪 TESTING TRAINING API")
    print("=" * 50)
    
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
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        try:
            result = response.json()
            print(f"📋 Response JSON:")
            print(json.dumps(result, indent=2, default=str))
            
            if result.get('success'):
                print(f"\n✅ TRAINING API SUCCESS!")
                print(f"🎯 Model: {result.get('model_name', 'Unknown')}")
                if 'result' in result:
                    training_result = result['result']
                    print(f"📁 Model Folder: {training_result.get('model_folder', 'Unknown')}")
                    print(f"📊 Accuracy: {training_result.get('main_score', 0)*100:.2f}%")
            else:
                print(f"\n❌ TRAINING API FAILED!")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
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
    test_training_api()