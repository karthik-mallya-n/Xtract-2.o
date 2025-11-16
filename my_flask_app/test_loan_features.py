#!/usr/bin/env python3
"""
Test Dynamic Features with Different Dataset
"""

import requests
import json

def test_loan_dataset():
    """Test training with a loan approval dataset"""
    
    print("🧪 TESTING WITH LOAN DATASET")
    print("=" * 50)
    
    # Use a mock file_id for our loan dataset
    file_id = "loan-dataset-test"
    model_name = "Random Forest"
    
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
            
            print(f"\n✅ Training API Response:")
            print(f"📋 Success: {result.get('success')}")
            
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
                    
                print(f"\n💡 These features will appear in the prediction form:")
                for feature in feature_names:
                    # Show how labels would be formatted
                    import re
                    label = re.sub(r'([A-Z])', r' \1', feature)
                    label = label.strip().title().replace(' Id', ' ID')
                    print(f"   • {feature} → '{label}'")
            else:
                print(f"\n❌ NO FEATURE INFO in response")
            
        except json.JSONDecodeError:
            print(f"❌ Non-JSON Response:")
            print(response.text[:500])
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_loan_dataset()