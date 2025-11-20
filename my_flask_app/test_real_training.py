#!/usr/bin/env python3
"""
Test script to verify training details with an actual uploaded file.
"""
import requests
import json
import time

def test_training_with_real_file():
    """Test training with a real uploaded file to verify training details display"""
    print("🧪 Testing Training with Real File...")
    
    # Use one of the recent uploaded files
    training_payload = {
        'file_id': 'e6134e1c-d3f9-405e-ad33-5386b13e5d92',  # Recent 5KB file
        'model_name': 'Random Forest',
        'is_labeled': 'true',
        'test_split': 0.2
    }
    
    print(f"📤 Sending training request: {json.dumps(training_payload, indent=2)}")
    
    try:
        # Send training request
        response = requests.post(
            'http://localhost:5000/api/train-specific-model',
            json=training_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training completed successfully!")
            
            # Check training_details at top level
            if 'training_details' in result:
                training_details = result['training_details']
                print(f"✅ Found training_details at top level!")
                
                print(f"\n📊 Training Details:")
                for key, value in training_details.items():
                    print(f"  {key}: {value}")
                    
                # Check for common metrics
                important_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'model_name', 'training_time', 'performance']
                print(f"\n🔍 Important Metrics Check:")
                for metric in important_metrics:
                    if metric in training_details:
                        value = training_details[metric]
                        if value and value not in ['N/A', 'NaN', None]:
                            print(f"  ✅ {metric}: {value}")
                        else:
                            print(f"  ❌ {metric}: {value} (problematic value)")
                    else:
                        print(f"  ⚠️  {metric}: Not found")
            else:
                print(f"❌ training_details not found at top level!")
                print(f"🔍 Available top-level keys: {list(result.keys())}")
                
                # Check if it's nested somewhere else
                for key, value in result.items():
                    if isinstance(value, dict) and 'accuracy' in value:
                        print(f"🔍 Found metrics in '{key}': {value}")
                
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Training failed with status {response.status_code}")
            print(f"Error: {json.dumps(error_data, indent=2)}")
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Training Details with Real File...\n")
    test_training_with_real_file()
    print("\n✅ Test completed!")